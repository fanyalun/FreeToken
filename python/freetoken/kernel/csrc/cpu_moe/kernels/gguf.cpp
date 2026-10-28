#include <atomic>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "../format.h"
#include "cpu_kernels.h"
#include "../../ggml/include/ggml.h"

namespace freetoken::cpu_moe {
namespace {

std::atomic<const ft_ggml_cpu_api*> g_ggml{nullptr};

const ft_ggml_cpu_api& ggml_api() {
  const ft_ggml_cpu_api* api = g_ggml.load();
  if (api == nullptr) throw std::logic_error("CPU MoE gguf experts need a ggml-cpu variant; call _cpu_moe.set_ggml_cpu first");
  return *api;
}

using Dot2 = decltype(ft_ggml_cpu_api::dot2);

Dot2 paired_dot(const ft_ggml_cpu_api& api, int type) {
  static const bool enabled = [] {
    const char* value = std::getenv("FREETOKEN_CPU_GGUF_DOT2");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled && type == GGML_TYPE_Q4_K && api.abi >= 2 ? api.dot2 : nullptr;
}

void dot_pair(Dot2 dot2, const ft_ggml_row_kernel& kernel, int type, int n,
              float* out, const void* weight, const void* x0, const void* x1) {
  if (dot2(type, n, out, weight, x0, x1) == FT_GGML_OK) return;
  kernel.vec_dot(n, out, 0, weight, 0, x0, 0, 1);
  kernel.vec_dot(n, out + 1, 0, weight, 0, x1, 0, 1);
}

}  // namespace

void set_ggml_cpu(uintptr_t addr) {
  const auto* api = reinterpret_cast<const ft_ggml_cpu_api*>(addr);
  if (api == nullptr || (api->abi != 1 && api->abi != FT_GGML_CPU_ABI)) throw std::invalid_argument("set_ggml_cpu: not a ggml-cpu kernel table of this build");
  const ft_ggml_cpu_api* prev = nullptr;
  if (!g_ggml.compare_exchange_strong(prev, api) && prev != api)
    throw std::invalid_argument(std::string("set_ggml_cpu: this process already uses the ") + prev->name + " ggml-cpu variant");
  static std::once_flag tables;
  std::call_once(tables, [api] { api->init(); });
}

torch::Tensor gguf_linear(torch::Tensor x, torch::Tensor weight, int type, int inputs, bool argmax) {
  TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu() && x.is_contiguous() && weight.is_contiguous(),
              "GGUF CPU linear requires contiguous CPU tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && weight.dim() == 2 && x.size(1) == inputs,
              "GGUF CPU linear input geometry");
  const auto& api = ggml_api();
  const auto dot2 = paired_dot(api, type);
  ft_ggml_row_kernel kernel{};
  TORCH_CHECK(api.row_kernel(type, inputs, &kernel) == FT_GGML_OK, "unsupported GGUF CPU linear type or row width");
  TORCH_CHECK(static_cast<size_t>(weight.size(1) * weight.element_size()) == kernel.row_bytes,
              "GGUF CPU linear packed row size mismatch");
  const int rows = x.size(0), outputs = weight.size(0), tile = 32, blocks = (outputs + tile - 1) / tile;
  TORCH_CHECK(rows > 0 && outputs > 0, "GGUF CPU linear requires nonempty matrices");
  AlignedVector<uint8_t> quant(static_cast<size_t>(rows) * kernel.act_row_bytes);
  std::vector<float> row(inputs);
  const auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < inputs; ++i) row[i] = bf16_to_f32(xp[(size_t)r * inputs + i]);
    kernel.from_float(row.data(), quant.data() + (size_t)r * kernel.act_row_bytes, inputs);
  }
  auto output = torch::empty(argmax ? std::vector<int64_t>{rows} : std::vector<int64_t>{rows, outputs},
                             x.options().dtype(argmax ? at::kLong : at::kBFloat16));
  std::vector<float> maxima(argmax ? blocks * rows : 0, -INFINITY);
  std::vector<int64_t> indices(argmax ? blocks * rows : 0);
  const auto* wp = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  auto* yp = argmax ? nullptr : reinterpret_cast<bf16_t*>(output.data_ptr());
  at::parallel_for(0, blocks, 1, [&](int64_t first, int64_t last) {
    for (int b = first; b < last; ++b) {
      if (argmax) for (int r = 0; r < rows; ++r) indices[b * rows + r] = b * tile;
      for (int o = b * tile; o < std::min(outputs, (b + 1) * tile); ++o)
        for (int r = 0; r < rows;) {
          float values[2];
          const int count = dot2 && r + 1 < rows ? 2 : 1;
          const auto* weight_row = wp + (size_t)o * kernel.row_bytes;
          const auto* input = quant.data() + (size_t)r * kernel.act_row_bytes;
          if (count == 2) dot_pair(dot2, kernel, type, inputs, values, weight_row, input, input + kernel.act_row_bytes);
          else kernel.vec_dot(inputs, values, 0, weight_row, 0, input, 0, 1);
          for (int j = 0; j < count; ++j, ++r) {
            const bf16_t rounded = f32_to_bf16(values[j]);
            if (yp) yp[(size_t)r * outputs + o] = rounded;
            else {
              const float v = bf16_to_f32(rounded);
              const int i = b * rows + r;
              if (v > maxima[i] || (std::isnan(v) && !std::isnan(maxima[i]))) {
                maxima[i] = v;
                indices[i] = o;
              }
            }
          }
        }
    }
  });
  if (argmax) {
    auto* out = output.data_ptr<int64_t>();
    for (int r = 0; r < rows; ++r) {
      float best = maxima[r];
      out[r] = indices[r];
      for (int b = 1; b < blocks; ++b) {
        const int i = b * rows + r;
        if (maxima[i] > best || (std::isnan(maxima[i]) && !std::isnan(best))) {
          best = maxima[i];
          out[r] = indices[i];
        }
      }
    }
  }
  return output;
}

void bind_gguf_cpu(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("gguf_linear", &gguf_linear, py::arg("x"), py::arg("weight"), py::arg("ggml_type"),
        py::arg("inputs"), py::arg("argmax") = false, py::call_guard<py::gil_scoped_release>());
  m.def("gguf_dot2_enabled", [] { return paired_dot(ggml_api(), GGML_TYPE_Q4_K) != nullptr; });
}

namespace {

// Packed ggml rows of the layers whose experts have this gate_up type and this down type, dotted by ggml-cpu against the activation converted to each type's vec_dot input (q8_0, q8_1 or q8_K, or the float type itself).
// With scales, each expert's (gate, up, down, 1) NVFP4 weight_scale_2 multiplies its gate and up rows before the glu and its down rows before the routing weight, as llama.cpp's build_moe_ffn does.
class GgufFormat final : public MoeFormat {
 public:
  GgufFormat(const MoeParams& p, const FormatArgs& a, bool scaled)
      : MoeFormat(p), ggml_(ggml_api()), gate_up_(a.banks.count("gate_up") ? a.bank("gate_up") : nullptr),
        gate_(gate_up_ ? nullptr : a.bank("gate")), up_(gate_up_ ? nullptr : a.bank("up")),
        down_(a.bank("down")), scales_(scaled ? a.bank("global") : nullptr) {
    gu_type_ = static_cast<int>(a.option("gate_up_type"));
    dn_type_ = static_cast<int>(a.option("down_type"));
    gu_ = kernel(gu_type_, p.H, "gate_up");
    dn_ = kernel(dn_type_, p.I, "down");
    gu_dot2_ = paired_dot(ggml_, gu_type_);
    dn_dot2_ = paired_dot(ggml_, dn_type_);
    xf_.resize(p.H);
    xq_.resize(static_cast<size_t>(p.max_tokens) * gu_.act_row_bytes);
    gq_.resize(static_cast<size_t>(p.max_tokens) * p.top_k * dn_.act_row_bytes);
  }

  std::string isa_name() const override { return std::string(ggml_.name) + "(ggml-cpu)"; }

  void prep_input(int tok, const bf16_t* x) override {
    for (int i = 0; i < p.H; ++i) xf_[i] = bf16_to_f32(x[i]);
    gu_.from_float(xf_.data(), xq_.data() + static_cast<size_t>(tok) * gu_.act_row_bytes, p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    thread_local std::vector<float> gf;
    gf.resize(p.I);
    for (int i = 0; i < p.I; ++i) gf[i] = bf16_to_f32(g[i]);
    dn_.from_float(gf.data(), gq_.data() + static_cast<size_t>(r) * dn_.act_row_bytes, p.I);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in, bf16_t* g) const override {
    const uint8_t* w = gate_up_ ? static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) + static_cast<size_t>(e) * 2 * p.I * gu_.row_bytes : nullptr;
    const uint8_t* xq = xq_.data() + static_cast<size_t>(tok) * gu_.act_row_bytes;
    const float* sc = expert_scales(layer, e);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      float s;
      const uint8_t* wr = w ? w + static_cast<size_t>(row) * gu_.row_bytes :
          static_cast<const uint8_t*>(tbl_at(row < p.I ? gate_ : up_, layer)) +
          (static_cast<size_t>(e) * p.I + row % p.I) * gu_.row_bytes;
      gu_.vec_dot(p.H, &s, 0, wr, 0, xq, 0, 1);
      return sc == nullptr ? s : s * sc[row < p.I ? 0 : 1];
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(down_, layer));
    for (int h = h0; h < h1; ++h)
      for (int j = 0; j < n;) {
        const auto* row = w + (static_cast<size_t>(routes[j].e) * p.H + h) * dn_.row_bytes;
        const auto* x0 = gq_.data() + static_cast<size_t>(routes[j].r) * dn_.act_row_bytes;
        if (dn_dot2_ && j + 1 < n && routes[j].e == routes[j + 1].e) {
          float values[2];
          dot_pair(dn_dot2_, dn_, dn_type_, p.I, values, row, x0,
                   gq_.data() + static_cast<size_t>(routes[j + 1].r) * dn_.act_row_bytes);
          for (int k = 0; k < 2; ++k) out[(j + k) * kRowBlock + h - h0] = values[k];
          j += 2;
        } else {
          dn_.vec_dot(p.I, &out[j * kRowBlock + h - h0], 0, row, 0, x0, 0, 1);
          ++j;
        }
      }
    if (scales_ == nullptr) return;
    for (int j = 0; j < n; ++j) {
      const float s = expert_scales(layer, routes[j].e)[2];
      for (int h = h0; h < h1; ++h) out[j * kRowBlock + h - h0] *= s;
    }
  }

  void gate_up_group(int layer, int e, int i0, int i1, const GroupRoute* routes,
                     int n, bool /*gpu_rounding*/) const override {
    const uint8_t* combined = gate_up_ ? static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) +
        static_cast<size_t>(e) * 2 * p.I * gu_.row_bytes : nullptr;
    const uint8_t* gate = combined ? combined : static_cast<const uint8_t*>(tbl_at(gate_, layer)) +
        static_cast<size_t>(e) * p.I * gu_.row_bytes;
    const uint8_t* up = combined ? combined + static_cast<size_t>(p.I) * gu_.row_bytes :
        static_cast<const uint8_t*>(tbl_at(up_, layer)) + static_cast<size_t>(e) * p.I * gu_.row_bytes;
    const float* scales = expert_scales(layer, e);
    for (int i = i0; i < i1; ++i) {
      const uint8_t* gate_row = gate + static_cast<size_t>(i) * gu_.row_bytes;
      const uint8_t* up_row = up + static_cast<size_t>(i) * gu_.row_bytes;
      int j = 0;
      for (; gu_dot2_ && j + 1 < n; j += 2) {
        float gates[2], ups[2];
        const auto* x0 = xq_.data() + static_cast<size_t>(routes[j].tok) * gu_.act_row_bytes;
        const auto* x1 = xq_.data() + static_cast<size_t>(routes[j + 1].tok) * gu_.act_row_bytes;
        dot_pair(gu_dot2_, gu_, gu_type_, p.H, gates, gate_row, x0, x1);
        dot_pair(gu_dot2_, gu_, gu_type_, p.H, ups, up_row, x0, x1);
        for (int k = 0; k < 2; ++k) {
          const auto& route = routes[j + k];
          gated_rows(p, i, i + 1, route.w_in, route.g, [&](int row) {
            const float value = row < p.I ? gates[k] : ups[k];
            return scales ? value * scales[row < p.I ? 0 : 1] : value;
          });
        }
      }
      for (; j < n; ++j) {
        const auto& route = routes[j];
        const uint8_t* xq = xq_.data() + static_cast<size_t>(route.tok) * gu_.act_row_bytes;
        gated_rows(p, i, i + 1, route.w_in, route.g, [&](int row) {
          float value;
          gu_.vec_dot(p.H, &value, 0, row < p.I ? gate_row : up_row, 0, xq, 0, 1);
          return scales ? value * scales[row < p.I ? 0 : 1] : value;
        });
      }
    }
  }

 private:
  // the expert's (gate, up, down, 1) row of the layer's [E, 4] fp32 scale bank, or nullptr without scales
  const float* expert_scales(int layer, int e) const {
    return scales_ == nullptr ? nullptr : static_cast<const float*>(tbl_at(scales_, layer)) + static_cast<size_t>(e) * 4;
  }

  ft_ggml_row_kernel kernel(int type, int k, const char* role) const {
    ft_ggml_row_kernel out{};
    const int rc = ggml_.row_kernel(type, k, &out);
    const std::string name = ggml_.type_name(type);
    if (rc == FT_GGML_NO_KERNEL) throw std::invalid_argument("ggml-cpu has no dot product for " + name + " experts");
    if (rc != FT_GGML_OK)
      throw std::invalid_argument("CPU MoE " + name + " " + role + " rows of " + std::to_string(k) + " values are not whole blocks");
    return out;
  }

  const ft_ggml_cpu_api& ggml_;
  const uint64_t* gate_up_;
  const uint64_t* gate_;
  const uint64_t* up_;
  const uint64_t* down_;
  const uint64_t* scales_;
  ft_ggml_row_kernel gu_{}, dn_{};
  int gu_type_, dn_type_;
  Dot2 gu_dot2_, dn_dot2_;
  std::vector<float> xf_;
  // the activation in the vec_dot input type: x per token, g per route
  AlignedVector<uint8_t> xq_, gq_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_gguf_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<GgufFormat>(p, a, false);
}

std::unique_ptr<MoeFormat> make_gguf_scaled_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<GgufFormat>(p, a, true);
}

}  // namespace freetoken::cpu_moe
