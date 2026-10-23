#pragma once
#include <chrono>

// SIMD outer-product structure adapted from llama.cpp simd-gemm.h.
// See ggml_v2_sources.json and ggml_v2_license.txt for provenance and license.
namespace cpu_target {

#if CPU_MOE_X86
__attribute__((target("avx512f")))
inline void gemm_v2(float* c, const float* a, const float* b, int rows,
                    const int* lengths, int cols, int as, int bs, int cs) {
  for (int j = 0; j < cols; j += 16) {
    const __mmask16 mask = (1u << std::min(16, cols - j)) - 1;
    __m512 acc[4];
    for (int r = 0; r < rows; ++r) acc[r] = _mm512_maskz_loadu_ps(mask, c + r * cs + j);
    int kmax = 0;
    for (int r = 0; r < rows; ++r) kmax = std::max(kmax, lengths[r]);
    for (int k = 0; k < kmax; ++k) {
      const auto bv = _mm512_maskz_loadu_ps(mask, b + k * bs + j);
      for (int r = 0; r < rows; ++r)
        // A masked future V must not enter even a zero-weight FMA.
        if (k < lengths[r]) acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(a[r * as + k]), bv, acc[r]);
    }
    for (int r = 0; r < rows; ++r) _mm512_mask_storeu_ps(c + r * cs + j, mask, acc[r]);
  }
}
#endif

#if CPU_MOE_X86
__attribute__((target("avx512f")))
inline __m512 load_bf16_v2(const bf16_t* x) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x))), 16));
}

__attribute__((target("avx512f")))
inline void norm_row_v2(const bf16_t* x, const float* w, bf16_t* y, int d, float eps, bool gemma) {
  auto sum = _mm512_setzero_ps(); int j = 0;
  for (; j + 16 <= d; j += 16) {
    const auto v = load_bf16_v2(x + j); sum = _mm512_fmadd_ps(v, v, sum);
  }
  float square = _mm512_reduce_add_ps(sum);
  for (; j < d; ++j) { const float v = bf16_to_f32(x[j]); square = std::fma(v, v, square); }
  const float scale = 1.f / std::sqrt(square / d + eps);
  j = 0;
  for (; j + 16 <= d; j += 16) {
    float result[16];
    const auto weight = _mm512_add_ps(_mm512_loadu_ps(w + j), _mm512_set1_ps(gemma ? 1.f : 0.f));
    _mm512_storeu_ps(result, _mm512_mul_ps(_mm512_mul_ps(load_bf16_v2(x + j), _mm512_set1_ps(scale)), weight));
    for (int i = 0; i < 16; ++i) y[j + i] = f32_to_bf16(result[i]);
  }
  for (; j < d; ++j) y[j] = f32_to_bf16(bf16_to_f32(x[j]) * scale * (w[j] + (gemma ? 1.f : 0.f)));
}

__attribute__((target("avx512f")))
inline void rope_row_v2(const bf16_t* x, const float* cosine, const float* sine, bf16_t* y, int d, int half) {
  int j = 0;
  for (; j + 16 <= half; j += 16) {
    const auto a = load_bf16_v2(x + j), b = load_bf16_v2(x + half + j);
    const auto c = _mm512_loadu_ps(cosine + j), s = _mm512_loadu_ps(sine + j);
    float first[16], second[16];
    _mm512_storeu_ps(first, _mm512_sub_ps(_mm512_mul_ps(a, c), _mm512_mul_ps(b, s)));
    _mm512_storeu_ps(second, _mm512_add_ps(_mm512_mul_ps(b, c), _mm512_mul_ps(a, s)));
    for (int i = 0; i < 16; ++i) { y[j + i] = f32_to_bf16(first[i]); y[half + j + i] = f32_to_bf16(second[i]); }
  }
  for (; j < half; ++j) {
    const float a = bf16_to_f32(x[j]), b = bf16_to_f32(x[half + j]);
    y[j] = f32_to_bf16(a * cosine[j] - b * sine[j]);
    y[half + j] = f32_to_bf16(b * cosine[j] + a * sine[j]);
  }
  std::copy(x + 2 * half, x + d, y + 2 * half);
}
#endif

inline torch::Tensor norm_v2(torch::Tensor x, torch::Tensor weight, double eps, bool gemma) {
  check(x, at::kBFloat16); check(weight, at::kFloat);
  TORCH_CHECK(Projection().isa == "avx512f", "ggml_v2 requires AVX-512F");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0 && weight.dim() == 1 && weight.numel() == x.size(-1), "norm geometry");
  auto out = torch::empty_like(x);
#if CPU_MOE_X86
  const int d = x.size(-1);
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr()); auto* yp = reinterpret_cast<bf16_t*>(out.data_ptr());
  at::parallel_for(0, x.numel() / d, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) norm_row_v2(xp + r * d, weight.data_ptr<float>(), yp + r * d, d, eps, gemma);
  });
#endif
  return out;
}

inline torch::Tensor rope_v2(torch::Tensor x, torch::Tensor cosine, torch::Tensor sine) {
  check(x, at::kBFloat16); check(cosine, at::kFloat); check(sine, at::kFloat);
  TORCH_CHECK(Projection().isa == "avx512f", "ggml_v2 requires AVX-512F");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0 && cosine.dim() == 1 && cosine.sizes() == sine.sizes()
      && cosine.numel() * 2 <= x.size(-1), "rope geometry");
  auto out = torch::empty_like(x);
#if CPU_MOE_X86
  const int d = x.size(-1), half = cosine.numel();
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr()); auto* yp = reinterpret_cast<bf16_t*>(out.data_ptr());
  at::parallel_for(0, x.numel() / d, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r)
      rope_row_v2(xp + r * d, cosine.data_ptr<float>(), sine.data_ptr<float>(), yp + r * d, d, half);
  });
#endif
  return out;
}

inline torch::Tensor gate_v2(torch::Tensor x, torch::Tensor gate, bool silu) {
  check(x, at::kBFloat16); check(gate, at::kBFloat16);
  TORCH_CHECK(x.dim() >= 2 && x.size(0) > 0 && (gate.sizes() == x.sizes()
      || (!silu && gate.dim() == 2 && gate.size(0) == x.size(0) && gate.size(1) == 1)), "gate geometry");
  auto out = torch::empty_like(x);
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr()); auto* gp = reinterpret_cast<const bf16_t*>(gate.data_ptr());
  auto* yp = reinterpret_cast<bf16_t*>(out.data_ptr()); const int stride = x.numel() / x.size(0);
  const bool broadcast = gate.numel() != x.numel();
  at::parallel_for(0, x.numel(), 256, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float g = bf16_to_f32(gp[broadcast ? i / stride : i]);
      const float sigmoid = 1.f / (1.f + std::exp(-g));
      const float multiplier = silu ? g * sigmoid : bf16_to_f32(f32_to_bf16(sigmoid));
      yp[i] = f32_to_bf16(bf16_to_f32(xp[i]) * multiplier);
    }
  });
  return out;
}

inline torch::Tensor linear_v2(torch::Tensor x, torch::Tensor weight, int outputs,
                               torch::Tensor input_tmp, torch::Tensor maxima,
                               torch::Tensor indices, bool argmax) {
  check(x, at::kBFloat16); check(weight, at::kFloat); check(input_tmp, at::kFloat);
  check(maxima, at::kFloat); check(indices, at::kLong);
#if CPU_MOE_X86
  TORCH_CHECK(__builtin_cpu_supports("avx512f"), "ggml_v2 requires AVX-512F");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 3 && x.size(0) > 0 && x.size(1) > 0
      && outputs > 0 && weight.size(0) == (outputs + 31) / 32
      && weight.size(1) == x.size(1) && weight.size(2) == 32, "packed projection geometry");
  const int rows = x.size(0), k = x.size(1), blocks = weight.size(0);
  TORCH_CHECK(input_tmp.numel() >= x.numel() && maxima.numel() >= blocks * rows
      && indices.numel() >= blocks * rows, "packed projection workspace");
  auto output = torch::empty(argmax ? std::vector<int64_t>{rows} : std::vector<int64_t>{rows, outputs},
                             x.options().dtype(argmax ? at::kLong : at::kBFloat16));
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* xt = input_tmp.data_ptr<float>(); auto* wp = weight.data_ptr<float>();
  auto* mp = maxima.data_ptr<float>(); auto* ip = indices.data_ptr<int64_t>();
  auto* yp = argmax ? nullptr : reinterpret_cast<bf16_t*>(output.data_ptr());
  at::parallel_for(0, x.numel(), 256, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) xt[i] = bf16_to_f32(xp[i]);
  });
  at::parallel_for(0, blocks, 1, [&](int64_t begin, int64_t end) {
    for (int block = begin; block < end; ++block) {
      const int cols = std::min(32, outputs - block * 32);
      for (int r0 = 0; r0 < rows; r0 += 4) {
        float tile[4 * 32] = {};
        const int lengths[4] = {k, k, k, k}, nr = std::min(4, rows - r0);
        gemm_v2(tile, xt + r0 * k, wp + static_cast<int64_t>(block) * k * 32,
                nr, lengths, cols, k, 32, 32);
        for (int r = 0; r < nr; ++r) {
          float best = -INFINITY; int index = block * 32;
          for (int j = 0; j < cols; ++j) {
            const auto rounded = f32_to_bf16(tile[r * 32 + j]);
            if (yp) yp[(r0 + r) * outputs + block * 32 + j] = rounded;
            const float value = bf16_to_f32(rounded);
            if (value > best || (std::isnan(value) && !std::isnan(best))) {
              best = value; index = block * 32 + j;
            }
          }
          mp[block * rows + r0 + r] = best; ip[block * rows + r0 + r] = index;
        }
      }
    }
  });
  if (argmax) {
    auto* result = output.data_ptr<int64_t>();
    for (int r = 0; r < rows; ++r) {
      float best = mp[r]; result[r] = ip[r];
      for (int block = 1; block < blocks; ++block) {
        const float value = mp[block * rows + r];
        if (value > best || (std::isnan(value) && !std::isnan(best))) {
          best = value; result[r] = ip[block * rows + r];
        }
      }
    }
  }
  return output;
#else
  TORCH_CHECK(false, "ggml_v2 requires AVX-512F");
#endif
}

inline void attention_workspace_v2(torch::Tensor kv_tmp, torch::Tensor segments, int kvh) {
  check(kv_tmp, at::kFloat); check(segments, at::kFloat);
  TORCH_CHECK(kv_tmp.dim() == 2 && kv_tmp.size(0) >= at::get_num_threads()
      && segments.dim() == 4 && kvh > 0 && segments.size(2) % kvh == 0, "workspace initialization geometry");
  const int ns = segments.size(0), rows = segments.size(1), qh = segments.size(2), ds = segments.size(3);
  const int groups = (qh / kvh + 1) / 2;
  auto* tmp = kv_tmp.data_ptr<float>(); auto* sp = segments.data_ptr<float>();
  at::parallel_for(0, kv_tmp.size(0), 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t)
      std::fill(tmp + t * kv_tmp.size(1), tmp + (t + 1) * kv_tmp.size(1), 0.f);
  });
  at::parallel_for(0, kvh * groups * ns, 1, [&](int64_t begin, int64_t end) {
    for (int task = begin; task < end; ++task) {
      const int s = task % ns, group = (task / ns) % groups, h = task / (ns * groups);
      for (int r = 0; r < rows; ++r)
        for (int g = h * (qh / kvh) + group * 2; g < std::min(h * (qh / kvh) + group * 2 + 2, (h + 1) * (qh / kvh)); ++g) {
          float* out = sp + ((static_cast<int64_t>(s) * rows + r) * qh + g) * ds;
          std::fill(out, out + ds, 0.f);
        }
    }
  });
}

template <bool profile>
inline void attention_v2_impl(torch::Tensor query, torch::Tensor keys, torch::Tensor values,
                         torch::Tensor query_tmp, torch::Tensor kv_tmp, torch::Tensor segments,
                         torch::Tensor output, int start, double* timing) {
  for (auto& t : {query, keys, values, output}) check(t, at::kBFloat16);
  for (auto& t : {query_tmp, kv_tmp, segments}) check(t, at::kFloat);
#if CPU_MOE_X86
  TORCH_CHECK(__builtin_cpu_supports("avx512f"), "ggml_v2 requires AVX-512F");
  TORCH_CHECK(query.dim() == 3 && keys.dim() == 3 && keys.sizes() == values.sizes(), "attention dimensions");
  const int rows = query.size(0), qh = query.size(1), d = query.size(2), kvh = keys.size(1);
  TORCH_CHECK(rows > 0 && qh > 0 && kvh > 0 && qh % kvh == 0 && d > 0
      && keys.size(2) == d && start >= 0 && start + rows <= keys.size(0)
      && output.sizes() == query.sizes(), "attention geometry");
  const int ns = (start + rows + 255) / 256, groups = (qh / kvh + 1) / 2;
  TORCH_CHECK(segments.dim() == 4 && segments.size(0) >= ns && segments.size(1) >= rows
      && segments.size(2) == qh && segments.size(3) == d + 2
      && query_tmp.numel() >= query.numel() && kv_tmp.dim() == 2
      && kv_tmp.size(0) >= at::get_num_threads() && kv_tmp.size(1) >= 128 * d,
      "attention v2 workspace capacity exceeded");
  auto* qp = reinterpret_cast<const bf16_t*>(query.data_ptr());
  auto* kp = reinterpret_cast<const bf16_t*>(keys.data_ptr());
  auto* vp = reinterpret_cast<const bf16_t*>(values.data_ptr());
  auto* op = reinterpret_cast<bf16_t*>(output.data_ptr());
  auto* qt = query_tmp.data_ptr<float>(); auto* tmp = kv_tmp.data_ptr<float>();
  auto* sp = segments.data_ptr<float>();
  const int64_t ss = segments.stride(0), ts = kv_tmp.stride(0);
  const float scale = 1.f / std::sqrt(static_cast<float>(d));
  using clock = std::chrono::steady_clock;
  clock::time_point stamp;
  if constexpr (profile) stamp = clock::now();
  at::parallel_for(0, query.numel(), 256, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) qt[i] = bf16_to_f32(qp[i]);
  });
  if constexpr (profile) timing[0] = std::chrono::duration<double>(clock::now() - stamp).count();
  at::parallel_for(0, kvh * groups * ns, 1, [&](int64_t begin, int64_t end) {
    float* tk = tmp + at::get_thread_num() * ts; float* tv = tk + 64 * d;
    for (int task = begin; task < end; ++task) {
      double phases[4] = {};
      clock::time_point phase_start;
      const int s = task % ns, group = (task / ns) % groups, h = task / (ns * groups);
      const int g0 = h * (qh / kvh) + group * 2, g1 = std::min(g0 + 2, (h + 1) * (qh / kvh));
      for (int r = 0; r < rows; ++r)
        for (int g = g0; g < g1; ++g) {
          float* acc = sp + s * ss + (r * qh + g) * (d + 2);
          std::fill(acc, acc + d + 2, 0.f); acc[d] = -INFINITY;
        }
      for (int base = s * 256; base < std::min((s + 1) * 256, start + rows); base += 64) {
        if constexpr (profile) phase_start = clock::now();
        const int size = std::min(64, start + rows - base);
        for (int j = 0; j < d; ++j)
          for (int t = 0; t < 64; ++t)
            tk[j * 64 + t] = t < size ? bf16_to_f32(kp[((base + t) * kvh + h) * d + j]) : 0.f;
        for (int t = 0; t < size; ++t)
          for (int j = 0; j < d; ++j) tv[t * d + j] = bf16_to_f32(vp[((base + t) * kvh + h) * d + j]);
        if constexpr (profile) phases[0] += std::chrono::duration<double>(clock::now() - phase_start).count();
        for (int g = g0; g < g1; ++g)
          for (int r0 = 0; r0 < rows; r0 += 4) {
            const int nr = std::min(4, rows - r0);
            float scores[4 * 64] = {};
            int lengths[4] = {d, d, d, d};
            if constexpr (profile) phase_start = clock::now();
            gemm_v2(scores, qt + (r0 * qh + g) * d, tk, nr, lengths, 64, qh * d, 64, 64);
            if constexpr (profile) {
              phases[1] += std::chrono::duration<double>(clock::now() - phase_start).count();
              phase_start = clock::now();
            }
            for (int r = 0; r < nr; ++r) {
              lengths[r] = std::max(0, std::min(size, start + r0 + r + 1 - base));
              if (!lengths[r]) continue;
              float* acc = sp + s * ss + ((r0 + r) * qh + g) * (d + 2);
              float m = acc[d];
              for (int t = 0; t < lengths[r]; ++t) {
                scores[r * 64 + t] *= scale;
                m = std::max(m, scores[r * 64 + t]);
              }
              const float alpha = std::exp(acc[d] - m);
              for (int j = 0; j < d; ++j) acc[j] *= alpha;
              acc[d + 1] *= alpha;
              for (int t = 0; t < lengths[r]; ++t) {
                scores[r * 64 + t] = std::exp(scores[r * 64 + t] - m);
                acc[d + 1] += scores[r * 64 + t];
              }
              acc[d] = m;
            }
            if constexpr (profile) {
              phases[2] += std::chrono::duration<double>(clock::now() - phase_start).count();
              phase_start = clock::now();
            }
            gemm_v2(sp + s * ss + (r0 * qh + g) * (d + 2), scores, tv, nr, lengths,
                    d, 64, d, qh * (d + 2));
            if constexpr (profile) phases[3] += std::chrono::duration<double>(clock::now() - phase_start).count();
          }
      }
      if constexpr (profile)
        for (int i = 0; i < 4; ++i) timing[2 + task * 4 + i] = phases[i];
    }
  });
  if constexpr (profile) stamp = clock::now();
  at::parallel_for(0, rows * qh, 1, [&](int64_t begin, int64_t end) {
    for (int i = begin; i < end; ++i) {
      float* acc = sp + i * (d + 2);
      // Absolute segment order is independent of workers and query width.
      for (int s = 1; s <= (start + i / qh) / 256; ++s) {
        const float* next = sp + s * ss + i * (d + 2);
        const float m = std::max(acc[d], next[d]);
        const float alpha = std::exp(acc[d] - m), beta = std::exp(next[d] - m);
        for (int j = 0; j < d; ++j) acc[j] = acc[j] * alpha + next[j] * beta;
        acc[d + 1] = acc[d + 1] * alpha + next[d + 1] * beta; acc[d] = m;
      }
      for (int j = 0; j < d; ++j) op[i * d + j] = f32_to_bf16(acc[j] / acc[d + 1]);
    }
  });
  if constexpr (profile) timing[1] = std::chrono::duration<double>(clock::now() - stamp).count();
#else
  TORCH_CHECK(false, "ggml_v2 requires AVX-512F");
#endif
}

inline void attention_v2(torch::Tensor query, torch::Tensor keys, torch::Tensor values,
                         torch::Tensor query_tmp, torch::Tensor kv_tmp, torch::Tensor segments,
                         torch::Tensor output, int start) {
  attention_v2_impl<false>(query, keys, values, query_tmp, kv_tmp, segments, output, start, nullptr);
}

inline void attention_v2_profiled(torch::Tensor query, torch::Tensor keys, torch::Tensor values,
                         torch::Tensor query_tmp, torch::Tensor kv_tmp, torch::Tensor segments,
                         torch::Tensor output, int start, torch::Tensor timing) {
  check(timing, at::kDouble);
  TORCH_CHECK(query.dim() == 3 && keys.dim() == 3 && keys.size(1) > 0 && start >= 0, "profile geometry");
  const int tasks = keys.size(1) * ((query.size(1) / keys.size(1) + 1) / 2) * ((start + query.size(0) + 255) / 256);
  TORCH_CHECK(timing.numel() >= 2 + tasks * 4, "profile workspace capacity");
  attention_v2_impl<true>(query, keys, values, query_tmp, kv_tmp, segments, output, start, timing.data_ptr<double>());
}

}  // namespace cpu_target
