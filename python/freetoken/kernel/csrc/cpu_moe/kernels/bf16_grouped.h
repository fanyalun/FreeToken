#pragma once

// Preserve the ordinary GEMV's four-accumulator reduction while sharing weight loads.
#if CPU_MOE_X86
template<int rows, typename Input = bf16_t>
__attribute__((target("avx512f")))
void tile_dot_compat_avx512f_impl(const bf16_t* w, const Input* const* x, float* out, int n) {
  __m512 acc[rows][4];
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < 4; ++j) acc[r][j] = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 4; ++j) {
      const auto weight = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j * 16))), 16));
      for (int r = 0; r < rows; ++r) {
        if constexpr (std::is_same_v<Input, float>)
          acc[r][j] = _mm512_fmadd_ps(weight, _mm512_loadu_ps(x[r] + i + j * 16), acc[r][j]);
        else
          acc[r][j] = _mm512_fmadd_ps(weight, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r] + i + j * 16))), 16)), acc[r][j]);
      }
    }
  }
  for (; i + 16 <= n; i += 16) {
    const auto weight = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i))), 16));
    for (int r = 0; r < rows; ++r) {
      if constexpr (std::is_same_v<Input, float>)
        acc[r][0] = _mm512_fmadd_ps(weight, _mm512_loadu_ps(x[r] + i), acc[r][0]);
      else
        acc[r][0] = _mm512_fmadd_ps(weight, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r] + i))), 16)), acc[r][0]);
    }
  }
  for (int r = 0; r < rows; ++r) out[r] = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[r][0], acc[r][1]), _mm512_add_ps(acc[r][2], acc[r][3])));
  for (; i < n; ++i) {
    const float weight = bf16_to_f32(w[i]);
    for (int r = 0; r < rows; ++r) {
      if constexpr (std::is_same_v<Input, float>) out[r] += weight * x[r][i];
      else out[r] += weight * bf16_to_f32(x[r][i]);
    }
  }
}

__attribute__((target("avx512f")))
void tile_dot_float_avx512f(const bf16_t* w, const float* const* x, float* out, int n, int rows) {
  if (rows == 1) tile_dot_compat_avx512f_impl<1>(w, x, out, n);
  else if (rows == 2) tile_dot_compat_avx512f_impl<2>(w, x, out, n);
  else tile_dot_compat_avx512f_impl<4>(w, x, out, n);
}
template<int rows>
__attribute__((target("avx2,fma")))
void tile_dot_compat_avx2_impl(const bf16_t* w, const bf16_t* const* x, float* out, int n) {
  __m256 acc[rows][4];
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < 4; ++j) acc[r][j] = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 4; ++j) {
      const auto weight = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j * 8))), 16));
      for (int r = 0; r < rows; ++r)
        acc[r][j] = _mm256_fmadd_ps(weight, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[r] + i + j * 8))), 16)), acc[r][j]);
    }
  }
  for (; i + 8 <= n; i += 8) {
    const auto weight = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i))), 16));
    for (int r = 0; r < rows; ++r)
      acc[r][0] = _mm256_fmadd_ps(weight, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[r] + i))), 16)), acc[r][0]);
  }
  for (int r = 0; r < rows; ++r) out[r] = hsum256(_mm256_add_ps(_mm256_add_ps(acc[r][0], acc[r][1]), _mm256_add_ps(acc[r][2], acc[r][3])));
  for (; i < n; ++i) {
    const float weight = bf16_to_f32(w[i]);
    for (int r = 0; r < rows; ++r) out[r] += weight * bf16_to_f32(x[r][i]);
  }
}
#ifdef CPU_MOE_HAS_AVX512BF16
template<int rows>
__attribute__((target("avx512bf16,avx512f")))
void tile_dot_compat_avx512bf16_impl(const bf16_t* w, const bf16_t* const* x, float* out, int n) {
  __m512 acc[rows][4];
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < 4; ++j) acc[r][j] = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 4; ++j) {
      const auto weight = load_bh(w + i + j * 32);
      for (int r = 0; r < rows; ++r)
        acc[r][j] = _mm512_dpbf16_ps(acc[r][j], weight, load_bh(x[r] + i + j * 32));
    }
  }
  for (; i + 32 <= n; i += 32) {
    const auto weight = load_bh(w + i);
    for (int r = 0; r < rows; ++r)
      acc[r][0] = _mm512_dpbf16_ps(acc[r][0], weight, load_bh(x[r] + i));
  }
  for (int r = 0; r < rows; ++r) out[r] = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[r][0], acc[r][1]), _mm512_add_ps(acc[r][2], acc[r][3])));
  for (; i < n; ++i) {
    const float weight = bf16_to_f32(w[i]);
    for (int r = 0; r < rows; ++r) out[r] += weight * bf16_to_f32(x[r][i]);
  }
}
void tile_dot_compat_avx512bf16(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  if (rows == 2) tile_dot_compat_avx512bf16_impl<2>(w, x, out, n);
  else tile_dot_compat_avx512bf16_impl<4>(w, x, out, n);
}
#endif
void tile_dot_compat_avx512f(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  if (rows == 2) tile_dot_compat_avx512f_impl<2>(w, x, out, n);
  else tile_dot_compat_avx512f_impl<4>(w, x, out, n);
}
void tile_dot_compat_avx2(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  tile_dot_compat_avx2_impl<2>(w, x, out, n);
}
#endif
