#pragma once

#include <ATen/Parallel.h>
#include <limits>

#if AT_PARALLEL_OPENMP && !defined(_OPENMP)
#error "CPU Target requires OpenMP compilation when PyTorch uses OpenMP"
#endif

namespace cpu_target {

inline void check(const torch::Tensor& t, at::ScalarType type) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == type && t.is_contiguous(),
              "CPU Target requires contiguous CPU tensors of the declared dtype");
}

using project_fn = void (*)(const float*, const bf16_t*, float*, int, int);

inline void project_scalar(const float* w, const bf16_t* x, float* out, int k, int rows) {
  for (int r = 0; r < rows; ++r) out[r] = 0;
  for (int j = 0; j < k; ++j)
    for (int r = 0; r < rows; ++r) out[r] += w[j] * bf16_to_f32(x[r * k + j]);
}

#if CPU_MOE_X86
__attribute__((target("avx512f")))
inline void project_512(const float* w, const bf16_t* x, float* out, int k, int rows) {
  __m512 sums[4];
  for (int r = 0; r < rows; ++r) sums[r] = _mm512_setzero_ps();
  int j = 0;
  for (; j + 16 <= k; j += 16) {
    const auto weight = _mm512_loadu_ps(w + j);
    for (int r = 0; r < rows; ++r) {
      const auto input = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + r * k + j))), 16));
      sums[r] = _mm512_fmadd_ps(weight, input, sums[r]);
    }
  }
  for (int r = 0; r < rows; ++r) out[r] = _mm512_reduce_add_ps(sums[r]);
  for (; j < k; ++j)
    for (int r = 0; r < rows; ++r) out[r] = std::fma(w[j], bf16_to_f32(x[r * k + j]), out[r]);
}

__attribute__((target("avx2,fma")))
inline void project_256(const float* w, const bf16_t* x, float* out, int k, int rows) {
  __m256 sums[2];
  for (int r = 0; r < rows; ++r) sums[r] = _mm256_setzero_ps();
  int j = 0;
  for (; j + 8 <= k; j += 8) {
    const auto weight = _mm256_loadu_ps(w + j);
    for (int r = 0; r < rows; ++r) {
      const auto input = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + r * k + j))), 16));
      sums[r] = _mm256_fmadd_ps(weight, input, sums[r]);
    }
  }
  for (int r = 0; r < rows; ++r) {
    auto sum = _mm_add_ps(_mm256_castps256_ps128(sums[r]), _mm256_extractf128_ps(sums[r], 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    out[r] = _mm_cvtss_f32(sum);
  }
  for (; j < k; ++j)
    for (int r = 0; r < rows; ++r) out[r] = std::fma(w[j], bf16_to_f32(x[r * k + j]), out[r]);
}
#endif

struct Projection {
  project_fn fn = project_scalar;
  int rows = 1;
  std::string isa = "scalar";
  Projection() {
#if CPU_MOE_X86
    if (__builtin_cpu_supports("avx512f")) { fn = project_512; rows = 4; isa = "avx512f"; }
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
      fn = project_256; rows = 2; isa = "avx2_fma";
    }
#endif
  }
};

inline torch::Tensor linear(torch::Tensor x, torch::Tensor weight, bool argmax) {
  check(x, at::kBFloat16); check(weight, at::kFloat);
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1)
              && x.size(0) > 0 && weight.size(0) > 0 && x.size(1) > 0, "projection geometry");
  const int rows = x.size(0), n = weight.size(0), k = x.size(1), block = 256;
  const int blocks = (n + block - 1) / block;
  auto output = torch::empty(argmax ? std::vector<int64_t>{rows} : std::vector<int64_t>{rows, n},
                             x.options().dtype(argmax ? at::kLong : at::kBFloat16));
  auto maxima = torch::empty({blocks, rows}, weight.options());
  auto indices = torch::empty({blocks, rows}, x.options().dtype(at::kLong));
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = weight.data_ptr<float>();
  auto* yp = argmax ? nullptr : reinterpret_cast<bf16_t*>(output.data_ptr());
  auto* mp = maxima.data_ptr<float>(); auto* ip = indices.data_ptr<int64_t>();
  const Projection dispatch;
  at::parallel_for(0, blocks, 1, [&](int64_t begin, int64_t end) {
    for (int b = begin; b < end; ++b) {
      for (int r = 0; r < rows; ++r) { mp[b * rows + r] = -INFINITY; ip[b * rows + r] = b * block; }
      for (int r = 0; r < rows; r += dispatch.rows) {
        int nr = std::min(dispatch.rows, rows - r);
        for (int o = b * block; o < std::min(n, (b + 1) * block); ++o) {
          float values[4];
          dispatch.fn(wp + o * k, xp + r * k, values, k, nr);
          for (int t = 0; t < nr; ++t) {
            const auto rounded = f32_to_bf16(values[t]);
            if (yp) yp[(r + t) * n + o] = rounded;
            const float v = bf16_to_f32(rounded);
            const int index = b * rows + r + t;
            if (v > mp[index] || (std::isnan(v) && !std::isnan(mp[index]))) {
              mp[index] = v; ip[index] = o;
            }
          }
        }
      }
    }
  });
  if (argmax) {
    auto* result = output.data_ptr<int64_t>();
    for (int r = 0; r < rows; ++r) {
      float best = mp[r]; result[r] = ip[r];
      for (int b = 1; b < blocks; ++b) {
        float v = mp[b * rows + r];
        if (v > best || (std::isnan(v) && !std::isnan(best))) {
          best = v; result[r] = ip[b * rows + r];
        }
      }
    }
  }
  return output;
}

inline void advance(float* state, const float* key, const float* value,
                    float decay, float beta, int kd, int vd) {
  for (int v = 0; v < vd; ++v) {
    float* s = state + v * kd;
    float dot = 0;
    for (int k = 0; k < kd; ++k) { s[k] *= decay; dot += s[k] * key[k]; }
    const float delta = (value[v] - dot) * beta;
    for (int k = 0; k < kd; ++k) s[k] += delta * key[k];
  }
}

#if CPU_MOE_X86
__attribute__((target("avx512f")))
inline void advance_v2(float* state, const float* key, const float* value,
                       float decay, float beta, int kd, int vd) {
  for (int v = 0; v < vd; ++v) {
    float* s = state + v * kd;
    auto dot = _mm512_setzero_ps(); int k = 0;
    for (; k + 16 <= kd; k += 16) {
      const auto sv = _mm512_mul_ps(_mm512_loadu_ps(s + k), _mm512_set1_ps(decay));
      _mm512_storeu_ps(s + k, sv);
      dot = _mm512_fmadd_ps(sv, _mm512_loadu_ps(key + k), dot);
    }
    float sum = _mm512_reduce_add_ps(dot);
    for (; k < kd; ++k) { s[k] *= decay; sum = std::fma(s[k], key[k], sum); }
    const float delta = (value[v] - sum) * beta;
    k = 0;
    for (; k + 16 <= kd; k += 16)
      _mm512_storeu_ps(s + k, _mm512_fmadd_ps(_mm512_set1_ps(delta),
          _mm512_loadu_ps(key + k), _mm512_loadu_ps(s + k)));
    for (; k < kd; ++k) s[k] = std::fma(delta, key[k], s[k]);
  }
}

__attribute__((target("avx512f")))
inline float readout_v2(const float* state, const float* query, float scale, int kd) {
  auto sum = _mm512_setzero_ps(); int k = 0;
  for (; k + 16 <= kd; k += 16)
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(state + k),
        _mm512_mul_ps(_mm512_loadu_ps(query + k), _mm512_set1_ps(scale)), sum);
  float result = _mm512_reduce_add_ps(sum);
  for (; k < kd; ++k) result = std::fma(state[k], query[k] * scale, result);
  return result;
}
#endif

template <bool v2>
inline void target_advance(float* state, const float* key, const float* value,
                           float decay, float beta, int kd, int vd) {
  if constexpr (v2) {
#if CPU_MOE_X86
    advance_v2(state, key, value, decay, beta, kd, vd);
#else
    TORCH_CHECK(false, "ggml_v2 requires AVX-512F");
#endif
  } else advance(state, key, value, decay, beta, kd, vd);
}

template <bool v2 = false>
inline void replay(torch::Tensor state, torch::Tensor inputs, int count) {
  if constexpr (v2) TORCH_CHECK(Projection().isa == "avx512f", "ggml_v2 requires AVX-512F");
  check(state, at::kFloat); check(inputs, at::kFloat);
  TORCH_CHECK(state.dim() == 3 && inputs.dim() == 3 && count >= 0 && count <= inputs.size(0), "replay geometry");
  int vh = state.size(0), vd = state.size(1), kd = state.size(2), stride = kd + vd + 2;
  TORCH_CHECK(inputs.size(1) == vh && inputs.size(2) == stride, "replay input geometry");
  auto* s = state.data_ptr<float>(); auto* p = inputs.data_ptr<float>();
  at::parallel_for(0, vh, 1, [&](int64_t begin, int64_t end) {
    for (int h = begin; h < end; ++h)
      for (int r = 0; r < count; ++r) {
        auto* item = p + (r * vh + h) * stride;
        target_advance<v2>(s + h * vd * kd, item, item + kd, item[kd + vd], item[kd + vd + 1], kd, vd);
      }
  });
}

template <bool v2 = false>
inline void gdn(torch::Tensor raw, torch::Tensor z, torch::Tensor a, torch::Tensor b,
                torch::Tensor conv_weight, torch::Tensor a_log, torch::Tensor bias,
                torch::Tensor norm_weight, torch::Tensor state, torch::Tensor windows,
                torch::Tensor inputs, torch::Tensor scratch, torch::Tensor output, int kh, double eps) {
  if constexpr (v2) TORCH_CHECK(Projection().isa == "avx512f", "ggml_v2 requires AVX-512F");
  for (auto& t : {raw, z, a, b, windows, output}) check(t, at::kBFloat16);
  for (auto& t : {conv_weight, a_log, bias, norm_weight, state, inputs, scratch}) check(t, at::kFloat);
  TORCH_CHECK(raw.dim() == 2 && state.dim() == 3 && windows.dim() == 2 && conv_weight.dim() == 2,
              "GDN dimensions");
  const int rows = raw.size(0), vh = state.size(0), vd = state.size(1), kd = state.size(2);
  const int channels = 2 * kh * kd + vh * vd, kernel = conv_weight.size(1), ws = windows.size(1);
  const int stride = kd + vd + 2;
  TORCH_CHECK(kh > 0 && vh % kh == 0 && raw.size(1) == channels && kernel > 0
      && windows.size(0) == channels && ws >= rows + kernel - 1
      && conv_weight.size(0) == channels && z.numel() == rows * vh * vd
      && a.numel() == rows * vh && b.numel() == a.numel() && a_log.numel() == vh
      && bias.numel() == vh && norm_weight.numel() == vd && inputs.dim() == 3
      && inputs.size(0) >= rows && inputs.size(1) == vh && inputs.size(2) == stride
      && scratch.numel() >= vh * (2 * kd + 2 * vd) && output.numel() == rows * vh * vd,
      "GDN workspace geometry");
  auto* rp = reinterpret_cast<const bf16_t*>(raw.data_ptr());
  auto* win = reinterpret_cast<bf16_t*>(windows.data_ptr());
  auto* zp = reinterpret_cast<const bf16_t*>(z.data_ptr());
  auto* ap = reinterpret_cast<const bf16_t*>(a.data_ptr());
  auto* bp = reinterpret_cast<const bf16_t*>(b.data_ptr());
  auto* out = reinterpret_cast<bf16_t*>(output.data_ptr());
  auto* cw = conv_weight.data_ptr<float>(); auto* al = a_log.data_ptr<float>();
  auto* dt = bias.data_ptr<float>(); auto* nw = norm_weight.data_ptr<float>();
  auto* st = state.data_ptr<float>(); auto* rec = inputs.data_ptr<float>(); auto* tmp = scratch.data_ptr<float>();
  at::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
    for (int ch = begin; ch < end; ++ch)
      for (int r = 0; r < rows; ++r) win[ch * ws + kernel - 1 + r] = rp[r * channels + ch];
  });
  at::parallel_for(0, vh, 1, [&](int64_t begin, int64_t end) {
    for (int h = begin; h < end; ++h) {
      float* q = tmp + h * (2 * kd + 2 * vd); float* key = q + kd;
      float* value = key + kd; float* core = value + vd;
      const int qh = h / (vh / kh);
      for (int r = 0; r < rows; ++r) {
        auto mix = [&](int ch) {
          float sum = 0;
          for (int t = 0; t < kernel; ++t) sum += bf16_to_f32(win[ch * ws + r + t]) * cw[ch * kernel + t];
          return bf16_to_f32(f32_to_bf16(sum / (1.f + std::exp(-sum))));
        };
        float qs = 0, ks = 0;
        for (int k = 0; k < kd; ++k) {
          q[k] = mix(qh * kd + k); key[k] = mix(kh * kd + qh * kd + k);
          qs += q[k] * q[k]; ks += key[k] * key[k];
        }
        const float qi = 1.f / std::sqrt(qs + 1e-6f), ki = 1.f / std::sqrt(ks + 1e-6f);
        for (int k = 0; k < kd; ++k) { q[k] *= qi; key[k] *= ki; }
        for (int v = 0; v < vd; ++v) value[v] = mix(2 * kh * kd + h * vd + v);
        const float av = bf16_to_f32(ap[r * vh + h]) + dt[h];
        const float softplus = av > 20 ? av : std::log1p(std::exp(av));
        const float decay = std::exp(-std::exp(al[h]) * softplus);
        const float beta = 1.f / (1.f + std::exp(-bf16_to_f32(bp[r * vh + h])));
        float* item = rec + (r * vh + h) * stride;
        std::copy(key, key + kd, item); std::copy(value, value + vd, item + kd);
        item[kd + vd] = decay; item[kd + vd + 1] = beta;
        target_advance<v2>(st + h * vd * kd, key, value, decay, beta, kd, vd);
        float square = 0;
        const float scale = 1.f / std::sqrt(static_cast<float>(kd));
        for (int v = 0; v < vd; ++v) {
          float sum = 0;
          if constexpr (v2) {
#if CPU_MOE_X86
            sum = readout_v2(st + (h * vd + v) * kd, q, scale, kd);
#endif
          } else {
            for (int k = 0; k < kd; ++k) sum += st[(h * vd + v) * kd + k] * (q[k] * scale);
          }
          core[v] = bf16_to_f32(f32_to_bf16(sum)); square += core[v] * core[v];
        }
        const float inv = 1.f / std::sqrt(square / vd + static_cast<float>(eps));
        for (int v = 0; v < vd; ++v) {
          const int index = (r * vh + h) * vd + v;
          float zv = bf16_to_f32(zp[index]);
          out[index] = f32_to_bf16(core[v] * inv * nw[v] * (zv / (1.f + std::exp(-zv))));
        }
      }
    }
  });
}

inline void attention(torch::Tensor query, torch::Tensor keys, torch::Tensor values,
                      torch::Tensor kv_tmp, torch::Tensor accum, torch::Tensor output, int start) {
  for (auto& t : {query, keys, values, output}) check(t, at::kBFloat16);
  check(kv_tmp, at::kFloat); check(accum, at::kFloat);
  TORCH_CHECK(query.dim() == 3 && keys.dim() == 3 && keys.sizes() == values.sizes(), "attention dimensions");
  const int rows = query.size(0), qh = query.size(1), d = query.size(2), kvh = keys.size(1), block = 64;
  TORCH_CHECK(kvh > 0 && qh % kvh == 0 && d > 0 && keys.size(2) == d && start >= 0
      && start + rows <= keys.size(0) && output.sizes() == query.sizes()
      && kv_tmp.numel() >= kvh * block * 2 * d && accum.numel() >= rows * qh * (d + 2), "attention geometry");
  auto* qp = reinterpret_cast<const bf16_t*>(query.data_ptr());
  auto* kp = reinterpret_cast<const bf16_t*>(keys.data_ptr());
  auto* vp = reinterpret_cast<const bf16_t*>(values.data_ptr());
  auto* op = reinterpret_cast<bf16_t*>(output.data_ptr());
  auto* tmp = kv_tmp.data_ptr<float>(); auto* ac = accum.data_ptr<float>();
  at::parallel_for(0, kvh, 1, [&](int64_t begin, int64_t end) {
    for (int h = begin; h < end; ++h) {
      float* tk = tmp + h * block * 2 * d; float* tv = tk + block * d;
      for (int r = 0; r < rows; ++r)
        for (int g = h * qh / kvh; g < (h + 1) * qh / kvh; ++g) {
          float* a = ac + (r * qh + g) * (d + 2);
          std::fill(a, a + d + 2, 0); a[d] = -INFINITY;
        }
      for (int base = 0; base < start + rows; base += block) {
        int size = std::min(block, start + rows - base);
        for (int t = 0; t < size; ++t)
          for (int j = 0; j < d; ++j) {
            tk[t * d + j] = bf16_to_f32(kp[((base + t) * kvh + h) * d + j]);
            tv[t * d + j] = bf16_to_f32(vp[((base + t) * kvh + h) * d + j]);
          }
        for (int r = 0; r < rows; ++r)
          for (int g = h * qh / kvh; g < (h + 1) * qh / kvh; ++g) {
            float* a = ac + (r * qh + g) * (d + 2);
            for (int t = 0; t < std::min(size, start + r + 1 - base); ++t) {
              float score = 0;
              for (int j = 0; j < d; ++j) score += bf16_to_f32(qp[(r * qh + g) * d + j]) * tk[t * d + j];
              score *= 1.f / std::sqrt(static_cast<float>(d));
              const float m = std::max(a[d], score), alpha = std::exp(a[d] - m), beta = std::exp(score - m);
              for (int j = 0; j < d; ++j) a[j] = a[j] * alpha + tv[t * d + j] * beta;
              a[d + 1] = a[d + 1] * alpha + beta; a[d] = m;
            }
          }
      }
      for (int r = 0; r < rows; ++r)
        for (int g = h * qh / kvh; g < (h + 1) * qh / kvh; ++g) {
          float* a = ac + (r * qh + g) * (d + 2);
          for (int j = 0; j < d; ++j) op[(r * qh + g) * d + j] = f32_to_bf16(a[j] / a[d + 1]);
        }
    }
  });
}

}  // namespace cpu_target
