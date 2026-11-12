#include "../format.h"
#include <type_traits>

namespace freetoken::cpu_moe {
namespace {

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. The selected impl is a function
// pointer chosen at runtime; the per-row call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

#if CPU_MOE_X86
CPU_MOE_TARGET("avx512f")
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  // 4 independent accumulators -> more in-flight loads (memory-level parallelism),
  // which is what lifts a bandwidth-bound GEMV toward peak.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#ifdef CPU_MOE_HAS_AVX512BF16
CPU_MOE_TARGET("avx512bf16,avx512f")
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

CPU_MOE_TARGET("avx512bf16,avx512f")
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  // 4 accumulators (128 bf16/iter) for memory-level parallelism + a prefetch nudge.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16 available

// --------- AVX2 + FMA (256-bit fallback for CPUs without AVX-512) ----------
// Covers Intel 12-14th gen / Arrow Lake (AVX-512 fused off) and AMD Zen<4. The
// bf16->fp32 widen is a zero-extend + <<16; with 4 independent accumulators the
// GEMV is memory-bandwidth bound, same as the AVX-512 path (just half the width).
CPU_MOE_TARGET("avx2,fma")
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // CPU_MOE_X86

dot_fn select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return dot_avx512bf16;
#endif
  if (t >= ISA_AVX512) return dot_avx512f;
  if (t >= ISA_AVX2) return dot_avx2;
#endif
  (void)t;
  return dot_scalar;
}

#include "bf16_grouped.h"

class Bf16Format final : public MoeFormat {
 public:
  Bf16Format(const MoeParams& p, const FormatArgs& a)
      : MoeFormat(p), dot_(select_dot()), gate_up_(a.bank("gate_up")), down_(a.bank("down")) {
#if CPU_MOE_X86
    if (dot_ == dot_avx512f) { tile_dot_ = tile_dot_compat_avx512f; tile_rows_ = 4; }
    else if (dot_ == dot_avx2) { tile_dot_ = tile_dot_compat_avx2; tile_rows_ = 2; }
#ifdef CPU_MOE_HAS_AVX512BF16
    else if (dot_ == dot_avx512bf16) { tile_dot_ = tile_dot_compat_avx512bf16; tile_rows_ = 4; }
#endif
#endif
  }

  std::string isa_name() const override { return tier_name(pick_isa()); }

  void gate_up(int layer, int e, int /*tok*/, int i0, int i1, const bf16_t* x, float w_in,
               bf16_t* g) const override {
    const bf16_t* w = static_cast<const bf16_t*>(tbl_at(gate_up_, layer)) + (size_t)e * (2 * p.I) * p.H;
    gated_rows(p, i0, i1, w_in, g, [&](int row) { return dot_(w + (size_t)row * p.H, x, p.H); });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const bf16_t* w = static_cast<const bf16_t*>(tbl_at(down_, layer));
    const bf16_t* inputs[64];
    float values[64];
    for (int h = h0; h < h1; ++h)
      for (int j = 0; j < n;) {
        int end = j + 1;
        while (end < n && routes[end].e == routes[j].e) ++end;
        for (int r = j; r < end; ++r) inputs[r - j] = routes[r].g;
        grouped_dot(w + ((size_t)routes[j].e * p.H + h) * p.I, inputs, values, p.I, end - j);
        for (int r = j; r < end; ++r) out[r * kRowBlock + h - h0] = values[r - j];
        j = end;
      }
  }

  void gate_up_group(int layer, int e, int i0, int i1, const GroupRoute* routes,
                     int n, bool gpu_rounding) const override {
    const bf16_t* w = static_cast<const bf16_t*>(tbl_at(gate_up_, layer)) + (size_t)e * 2 * p.I * p.H;
    const bf16_t* inputs[64];
    float gate[64], up[64];
    for (int j = 0; j < n; ++j) inputs[j] = routes[j].x;
    for (int i = i0; i < i1; ++i) {
      grouped_dot(w + (size_t)i * p.H, inputs, gate, p.H, n);
      grouped_dot(w + (size_t)(p.I + i) * p.H, inputs, up, p.H, n);
      for (int j = 0; j < n; ++j) {
        const auto& r = routes[j];
        gated_rows(p, i, i + 1, 1.0f, r.g, [&](int row) {
          const float v = (row < p.I ? gate[j] : up[j]) * r.w_in;
          return gpu_rounding ? bf16_to_f32(f32_to_bf16(v)) : v;
        });
      }
    }
  }

 private:
  dot_fn dot_;
  using tile_dot_fn = void (*)(const bf16_t*, const bf16_t* const*, float*, int, int);
  tile_dot_fn tile_dot_ = nullptr;
  int tile_rows_ = 1;

  void grouped_dot(const bf16_t* w, const bf16_t* const* x, float* out, int k, int rows) const {
    for (int first = 0; first < rows;) {
      const int left = rows - first;
      const int count = tile_rows_ >= 4 && left >= 4 ? 4 : tile_rows_ >= 2 && left >= 2 ? 2 : 1;
      if (count == 1) out[first] = dot_(w, x[first], k);
      else tile_dot_(w, x + first, out + first, k, count);
      first += count;
    }
  }
  // bf16 rows: gate_up [E, 2I, H], down [E, H, I].
  const uint64_t* gate_up_;
  const uint64_t* down_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_bf16_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Bf16Format>(p, a);
}

}  // namespace freetoken::cpu_moe
