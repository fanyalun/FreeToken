// BF16 speculative Hybrid and CPU Target preserve their task and rounding contracts.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <freetoken/hip_compat.h>
#include <torch/extension.h>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#else
inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ACT_SWIGLUOAI is the clamped (up + 1) swiglu (gpt-oss "swigluoai" /
// MiniMax-M3): gate/up are combined jointly with the runtime alpha/limit
// scalars, so it is handled in the do_pass1 epilogue (act_apply never sees it;
// the mxfp4 kernel additionally fuses its own copy of the same math).
// ACT_SWIGLU_CLAMP (GLM-5.3 "swiglu_limit") is the same clamped form WITHOUT
// the (up + 1) bias: clamp(gate, max=lim) * sigmoid(alpha*gate) * clamp(up, +-lim).
enum ActKind {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_GELU_TANH = 2,
  ACT_SWIGLUOAI = 3,
  ACT_SWIGLU_CLAMP = 4,
};

inline float act_apply(int act, float x) {
  if (act == ACT_SILU) return x / (1.0f + std::exp(-x));
  if (act == ACT_GELU)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
  // gelu_tanh
  const float k0 = 0.79788456080286535588f;  // sqrt(2/pi)
  const float inner = k0 * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. The selected impl is a function
// pointer chosen at runtime; the per-row call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

using tile_dot_fn = void (*)(const bf16_t*, const bf16_t* const*, float*, int, int);

void tile_dot_scalar(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  for (int r = 0; r < rows; ++r) out[r] = 0;
  for (int k = 0; k < n; ++k) {
    const float v = bf16_to_f32(w[k]);
    for (int r = 0; r < rows; ++r) out[r] += v * bf16_to_f32(x[r][k]);
  }
}

#if CPU_MOE_X86
template<int Rows>
__attribute__((target("avx512f")))
void tile_dot_avx512_impl(const bf16_t* w, const bf16_t* const* x, float* out, int n) {
  __m512 acc[Rows];
  for (int r = 0; r < Rows; ++r) acc[r] = _mm512_setzero_ps();
  int k = 0;
  for (; k + 16 <= n; k += 16) {
    const __m512 weight = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k))), 16));
    for (int r = 0; r < Rows; ++r) {
      const __m512 input = _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r] + k))), 16));
      acc[r] = _mm512_fmadd_ps(weight, input, acc[r]);
    }
  }
  for (int r = 0; r < Rows; ++r) out[r] = _mm512_reduce_add_ps(acc[r]);
  for (; k < n; ++k) {
    const float weight = bf16_to_f32(w[k]);
    for (int r = 0; r < Rows; ++r) out[r] += weight * bf16_to_f32(x[r][k]);
  }
}

__attribute__((target("avx512f")))
void tile_dot_avx512(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  if (rows == 8) tile_dot_avx512_impl<8>(w, x, out, n);
  else if (rows == 4) tile_dot_avx512_impl<4>(w, x, out, n);
  else tile_dot_avx512_impl<2>(w, x, out, n);
}

template<int Rows>
__attribute__((target("avx2,fma")))
void tile_dot_avx2_impl(const bf16_t* w, const bf16_t* const* x, float* out, int n) {
  __m256 acc[Rows];
  for (int r = 0; r < Rows; ++r) acc[r] = _mm256_setzero_ps();
  int k = 0;
  for (; k + 8 <= n; k += 8) {
    const __m256 weight = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k))), 16));
    for (int r = 0; r < Rows; ++r) {
      const __m256 input = _mm256_castsi256_ps(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[r] + k))), 16));
      acc[r] = _mm256_fmadd_ps(weight, input, acc[r]);
    }
  }
  for (int r = 0; r < Rows; ++r) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc[r]);
    out[r] = 0;
    for (float v : lanes) out[r] += v;
  }
  for (; k < n; ++k) {
    const float weight = bf16_to_f32(w[k]);
    for (int r = 0; r < Rows; ++r) out[r] += weight * bf16_to_f32(x[r][k]);
  }
}

__attribute__((target("avx2,fma")))
void tile_dot_avx2(const bf16_t* w, const bf16_t* const* x, float* out, int n, int rows) {
  if (rows == 8) tile_dot_avx2_impl<8>(w, x, out, n);
  else if (rows == 4) tile_dot_avx2_impl<4>(w, x, out, n);
  else tile_dot_avx2_impl<2>(w, x, out, n);
}
#endif

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

// Software prefetch distance (bytes) ahead of the current weight row stream. The
// weight stream is the bandwidth bottleneck (read once, never reused); nudging the
// HW prefetcher with a few cache lines of lookahead raises sustained throughput.
constexpr int PF_AHEAD = 512;

#if CPU_MOE_X86
__attribute__((target("avx512f")))
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

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512BF16 1
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
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
__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
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


// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done[slot]=0 then ready[slot]=1), sync = WAIT_VALUE(done[slot] >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (Windows WDDM,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
#include <windows.h>
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
#include <dlfcn.h>
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

static bool cumemop_resolve() {
  static bool resolved = [] {
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync.
// Returns true only if the whole memop path works on THIS stream/device/driver.
static bool cumemops_probe(uintptr_t stream, uintptr_t scratch_addr) {
  if (!cumemop_resolve()) return false;
  auto* s = reinterpret_cast<void*>(stream);
  if (g_cu_write64(s, (unsigned long long)scratch_addr, 7ULL, kCuWriteDefault) != 0) return false;
  if (g_cu_wait64(s, (unsigned long long)scratch_addr, 7ULL, kCuWaitValueGeq) != 0) return false;
  return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

static void cumemop_submit(uintptr_t stream, uintptr_t done_addr, uintptr_t ready_addr,
                           int64_t slot) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  cumemop_check(g_cu_write64(s, (unsigned long long)(done_addr + (size_t)slot * 8), 0ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(done)");
  cumemop_check(g_cu_write64(s, (unsigned long long)(ready_addr + (size_t)slot * 8), 1ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(ready)");
}

static void cumemop_sync(uintptr_t stream, uintptr_t done_addr, int64_t slot) {
  cumemop_check(g_cu_wait64(reinterpret_cast<void*>(stream),
                            (unsigned long long)(done_addr + (size_t)slot * 8), 1ULL,
                            kCuWaitValueGeq),
                "cuStreamWaitValue64(done)");
}

struct DotChoice {
  dot_fn fn;
  const char* name;
};

// SIMD tiers, ascending. Each format picks the highest tier <= the one chosen by
// pick_isa() that it implements (fp4 formats have no bf16-specific tier, so the
// avx512bf16 tier maps to their avx512 kernel).
enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

// Best tier the CPU+build supports, optionally capped DOWN by
// FREETOKEN_CPU_MOE_ISA={scalar,avx2,avx512,avx512bf16} (A/B testing on a machine
// that supports more). FREETOKEN_CPU_MOE_SCALAR=1 forces scalar (legacy alias).
inline IsaTier pick_isa() {
#if CPU_MOE_X86
  if (getenv("FREETOKEN_CPU_MOE_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOE_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOE_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;  // cap downward; never force above hw/build support
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

DotChoice select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return {dot_avx512bf16, "avx512bf16"};
#endif
  if (t >= ISA_AVX512) return {dot_avx512f, "avx512f"};
  if (t >= ISA_AVX2) return {dot_avx2, "avx2"};
#endif
  (void)t;
  return {dot_scalar, "scalar"};
}

// --------------------------------- executor ---------------------------------

struct SpeculativeBf16Executor;

#if CPU_MOE_X86
// Outer-product SIMD structure follows the pinned GGML microkernel; BF16 input packing is local.
__attribute__((target("avx512f")))
inline void packed_target_dot(const bf16_t* weight, const bf16_t* const* inputs,
                               float* output, int k, int rows, int cols) {
  for (int j = 0; j < cols; j += 16) {
    __m512 sums[4];
    for (int r = 0; r < rows; ++r) sums[r] = _mm512_setzero_ps();
    for (int i = 0; i < k; ++i) {
      const auto w = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + i * 32 + j))), 16));
      for (int r = 0; r < rows; ++r)
        sums[r] = _mm512_fmadd_ps(_mm512_set1_ps(bf16_to_f32(inputs[r][i])), w, sums[r]);
    }
    for (int r = 0; r < rows; ++r) _mm512_storeu_ps(output + r * 32 + j, sums[r]);
  }
}
#endif

struct MoeTask {
  SpeculativeBf16Executor* exec;
  int layer_id;
  int num_tokens;
  const bf16_t* x;     // [num_tokens, H]
  const int32_t* ids;  // [num_tokens, top_k]  (raw expert ids; <0 = skip)
  const float* w;      // [num_tokens, top_k]
  bf16_t* y;           // [num_tokens, H]
  bool grouped = false;
  bool grouped_compat = false;
  bool grouped_packed = false;
  bool grouped_reuse = false;
  bool grouped_numa = false;
};

// Output-row tiling. Small enough to give every worker independent work even at
// batch size 1; large enough to amortize the atomic work-grab.
//
// Bandwidth notes (Sapphire Rapids 8480+, 13 cores): the two passes already read
// every expert weight byte exactly once per token (each output row block is owned
// by one worker), and x stays hot in L1 across a (token,expert)'s rows -- so the
// kernel is single-read bandwidth-optimal at bs=1 (~205 GB/s vs ~55 GB/s PCIe).
// One worker per *physical* core, pinned, is the sweet spot; SMT oversubscription
// thrashes the spin-barrier. Deferred (not worth it here / for this workload):
//   - AMX-bf16: a GEMM tile engine; decode is M=1 GEMV so tiles sit idle. It would
//     only pay off in a grouped/batched (dedup) path.
//   - expert dedup for bs>1: read each distinct expert once and GEMM its tokens.
//     Helps locality+bytes when bs is large; decode batches here are tiny (<=4).
//   - NUMA: a single node is assumed. Multi-socket machines would split each
//     expert's K dimension per node (banks are already per-row contiguous).
constexpr int IBLK = 32;
constexpr int HBLK = 32;

inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return tbl ? reinterpret_cast<const void*>(tbl[layer_id]) : nullptr;
}

struct SpeculativeBf16Executor {
  int num_threads;
  int num_layers, num_experts, top_k;
  int H, I;
  int act, apply_on_input;
  const uint64_t* gate_up_tbl;
  const uint64_t* down_tbl;
  float swiglu_alpha, swiglu_limit;
  dot_fn dot;
  const char* isa;
  bool profile_workers = false;

  std::vector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate
  struct GroupTile { int expert, begin, rows; };
  int grouped_capacity = 0, group_tiles = 0;
  bool expert_grouped = false;
  std::vector<std::vector<int>> legacy_routes;
  std::vector<int> legacy_experts;
  std::vector<float> legacy_down;
  std::vector<int> group_counts, group_offsets, group_routes, group_cursor;
  std::vector<GroupTile> group_tasks;
  struct alignas(64) GroupQueue {
    std::atomic<int64_t> gate_next{0}, down_next{0};
    int64_t gate_end = 0, down_end = 0;
  } group_nodes[2];
  std::vector<float> route_output;
  std::vector<float> group_input_float, group_intermediate_float;
  tile_dot_fn tile_dot = tile_dot_scalar;
  tile_dot_fn tile_dot_compat = tile_dot_scalar;
  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  bool stop = false;
  uint64_t cur_gen = 0;
  MoeTask* cur_task = nullptr;
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};

  std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> prt_next{0};  // ds_fp4 intermediate fp8 round-trip phase
  int64_t p1_total = 0, p2_total = 0, prt_total = 0;
  int n_iblk = 0, n_hblk = 0;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  std::vector<MoeTask*> owned_tasks;  // persistent task descriptors (graph-stable)
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // A tiny GPU kernel bumps ready_flags[slot] at submit; this coordinator thread busy-polls
  // it, runs the slot's task on the worker pool, and sets done_flags[slot], which a GPU
  // spin-wait kernel polls at sync. This removes the ~2x30-50us host-func dispatch round
  // trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a
  // 75-layer model). One slot per (layer, decode batch size) pair -- the Python side
  // allocates slots as tasks are created. Flags live in mapped-pinned host memory (UVA:
  // the same pointers are used by the GPU kernels and by this thread).
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  volatile int64_t* ready_flags = nullptr;  // GPU increments, this thread polls
  volatile int64_t* done_flags = nullptr;   // this thread sets, GPU spin-waits
  int coord_num_slots = 0;
  std::vector<MoeTask*> flag_task;           // slot -> task (registered lazily)
  std::vector<int64_t> flag_served;          // slot -> completed dispatch count (tests/debug)
  std::atomic<int> active_flag_slot{-1};
  std::mutex flag_task_mtx;

  // Portable ordering for the flag handshake: "ready observed => the DMA'd inputs that
  // preceded the bump are visible" and "y stores are visible before done". Plain
  // volatile loads lean on x86 TSO; acquire/release makes it hold on aarch64 too
  // (GH200/Jetson) at zero x86 cost. (MSVC branch is x86-only today: compiler barrier
  // + TSO.)
  static int64_t flag_load_acquire(const volatile int64_t* p) {
#if defined(_MSC_VER)
    const int64_t v = *p;
    _ReadWriteBarrier();
    return v;
#else
    return __atomic_load_n(const_cast<const int64_t*>(p), __ATOMIC_ACQUIRE);
#endif
  }

  static void flag_store_release(volatile int64_t* p, int64_t v) {
#if defined(_MSC_VER)
    _ReadWriteBarrier();
    *p = v;
#else
    __atomic_store_n(const_cast<int64_t*>(p), v, __ATOMIC_RELEASE);
#endif
  }

  SpeculativeBf16Executor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens, int activation_id,
                 int apply_router_weight_on_input, uintptr_t gate_up_ptr,
                 uintptr_t down_ptr, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_)
      : num_threads(num_threads_ > 0 ? num_threads_ : 1), num_layers(num_layers_),
        num_experts(num_experts_), top_k(top_k_), H(hidden_size), I(inter_size),
        act(activation_id), apply_on_input(apply_router_weight_on_input),
        gate_up_tbl(reinterpret_cast<const uint64_t*>(gate_up_ptr)),
        down_tbl(reinterpret_cast<const uint64_t*>(down_ptr)),
        swiglu_alpha(static_cast<float>(swiglu_alpha_)),
        swiglu_limit(static_cast<float>(swiglu_limit_)), core_ids(std::move(core_ids_)) {
    const auto choice = select_dot();
    dot = choice.fn;
    isa = choice.name;
    g_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  void pin_self(int tid) {
#if CPU_MOE_HAS_AFFINITY
    if (core_ids.empty()) return;
    const int cpu = core_ids[tid % static_cast<int>(core_ids.size())];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid;
#endif
  }

  ~SpeculativeBf16Executor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
    for (MoeTask* t : owned_tasks) delete t;
  }

  uintptr_t create_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                        uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    MoeTask* t = new MoeTask{this,
                             layer_id,
                             num_tokens,
                             reinterpret_cast<const bf16_t*>(x_ptr),
                             reinterpret_cast<const int32_t*>(ids_ptr),
                             reinterpret_cast<const float*>(w_ptr),
                             reinterpret_cast<bf16_t*>(y_ptr)};
    owned_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  void prepare_grouped(int max_tokens) {
    if (max_tokens < 1 || !owned_tasks.empty())
      throw std::runtime_error("prepare_grouped requires BF16 and must precede task registration");
    grouped_capacity = max_tokens;
    const size_t routes = static_cast<size_t>(max_tokens) * top_k;
    g_scratch.resize(routes * I);
    route_output.resize(routes * H);
    group_counts.resize(num_experts);
    group_offsets.resize(num_experts + 1);
    group_cursor.resize(num_experts);
    group_routes.resize(routes);
    group_tasks.resize(routes);
#if CPU_MOE_X86
    const auto tier = pick_isa();
    if (tier >= ISA_AVX512) tile_dot = tile_dot_avx512;
    else if (tier >= ISA_AVX2) tile_dot = tile_dot_avx2;
#endif
    tile_dot_compat = tile_dot_scalar;
#if CPU_MOE_X86
    if (dot == dot_avx512f) tile_dot_compat = tile_dot_compat_avx512f;
    else if (dot == dot_avx2) tile_dot_compat = tile_dot_compat_avx2;
#ifdef CPU_MOE_HAS_AVX512BF16
    else if (dot == dot_avx512bf16) tile_dot_compat = tile_dot_compat_avx512bf16;
#endif
#endif
  }

  uintptr_t create_grouped_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                                uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    if (num_tokens < 1 || num_tokens > grouped_capacity || layer_id < 0 || layer_id >= num_layers)
      throw std::runtime_error("grouped CPU MoE task exceeds prepared geometry");
    auto task = create_task(layer_id, num_tokens, x_ptr, ids_ptr, w_ptr, y_ptr);
    reinterpret_cast<MoeTask*>(task)->grouped = true;
    return task;
  }

  void build_groups(const MoeTask* t) {
    std::fill(group_counts.begin(), group_counts.end(), 0);
    const int routes = t->num_tokens * top_k;
    for (int r = 0; r < routes; ++r) {
      const int e = t->ids[r];
      if (e >= 0 && e < num_experts && (t->grouped_compat || t->w[r] != 0)) ++group_counts[e];
    }
    group_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
      group_offsets[e + 1] = group_offsets[e] + group_counts[e];
      group_cursor[e] = group_offsets[e];
    }
    for (int r = 0; r < routes; ++r) {
      const int e = t->ids[r];
      if (e >= 0 && e < num_experts && (t->grouped_compat || t->w[r] != 0)) group_routes[group_cursor[e]++] = r;
    }
    group_tiles = 0;
    int max_rows = 8;
#if CPU_MOE_X86
    // Four accumulators per route must fit alongside temporaries in SIMD registers.
    if (t->grouped_compat) max_rows = dot == dot_avx2 ? 2 : 4;
#endif
    for (int e = 0; e < num_experts; ++e) {
      int begin = group_offsets[e], left = group_counts[e];
      if (t->grouped_reuse && left) {
        group_tasks[group_tiles++] = {e, begin, left};
        continue;
      }
      while (left) {
        const int chunk = std::min(left, max_rows);
        const int rows = chunk >= 8 ? 8 : chunk >= 4 ? 4 : chunk >= 2 ? 2 : 1;
        group_tasks[group_tiles++] = {e, begin, rows};
        begin += rows;
        left -= rows;
      }
    }
  }

  void packed_grouped_pass(const MoeTask* t, int64_t p, bool down) {
#if CPU_MOE_X86
    const int blocks = down ? n_hblk : n_iblk;
    const auto tile = group_tasks[p / blocks];
    const int block = p % blocks, first = block * 32;
    const int cols = std::min(32, (down ? H : I) - first), k = down ? I : H;
    const auto bank = reinterpret_cast<const bf16_t*>(tbl_at(down ? down_tbl : gate_up_tbl, t->layer_id));
    const int64_t matrix = static_cast<int64_t>(blocks) * k * 32;
    const auto weight = bank + tile.expert * matrix * (down ? 1 : 2) + block * k * 32;
    const bf16_t* inputs[4]; int routes[4];
    for (int r = 0; r < tile.rows; ++r) {
      routes[r] = group_routes[tile.begin + r];
      inputs[r] = down ? g_scratch.data() + static_cast<size_t>(routes[r]) * I
                       : t->x + static_cast<size_t>(routes[r] / top_k) * H;
    }
    float a[4 * 32], b[4 * 32];
    packed_target_dot(weight, inputs, a, k, tile.rows, cols);
    if (!down) packed_target_dot(weight + matrix, inputs, b, k, tile.rows, cols);
    for (int r = 0; r < tile.rows; ++r)
      for (int j = 0; j < cols; ++j) {
        const size_t route = routes[r];
        if (down) route_output[route * H + first + j] = a[r * 32 + j];
        else {
          const float scale = apply_on_input ? t->w[route] : 1.f;
          float gate = a[r * 32 + j] * scale, up = b[r * 32 + j] * scale;
          if (act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP) {
            gate = std::min(gate, swiglu_limit);
            up = std::max(-swiglu_limit, std::min(up, swiglu_limit));
            g_scratch[route * I + first + j] = f32_to_bf16(
                gate / (1.f + std::exp(-gate * swiglu_alpha)) * (up + (act == ACT_SWIGLUOAI ? 1 : 0)));
          } else g_scratch[route * I + first + j] = f32_to_bf16(act_apply(act, gate) * up);
        }
      }
#else
    TORCH_CHECK(false, "packed CPU Target requires AVX-512F");
#endif
  }

  void grouped_pass(const MoeTask* t, int64_t p, bool down) {
    if (t->grouped_packed) { packed_grouped_pass(t, p, down); return; }
    const int blocks = down ? n_hblk : n_iblk;
    auto tile = group_tasks[p / blocks];
    const int block = static_cast<int>(p % blocks);
    if (t->grouped_reuse) {
      // Keep an expert's output tile local across all of its token microtiles.
      int limit = 4;
#if CPU_MOE_X86
      if (dot == dot_avx2) limit = 2;
#endif
      int left = tile.rows;
      while (left) {
        const int chunk = std::min(left, limit);
        tile.rows = chunk >= 4 ? 4 : chunk >= 2 ? 2 : 1;
        grouped_tile_pass(t, tile, block, down);
        tile.begin += tile.rows;
        left -= tile.rows;
      }
      return;
    }
    grouped_tile_pass(t, tile, block, down);
  }

  bool expanded_group_inputs(const MoeTask* t) const {
#if CPU_MOE_X86
    return t->grouped_reuse && dot == dot_avx512f;
#else
    return false;
#endif
  }

  void grouped_tile_pass(const MoeTask* t, const GroupTile& tile, int block, bool down) {
    const int first = block * (down ? HBLK : IBLK);
    const int last = std::min(down ? H : I, first + (down ? HBLK : IBLK));
    const auto bank = reinterpret_cast<const bf16_t*>(tbl_at(down ? down_tbl : gate_up_tbl, t->layer_id));
    const bf16_t* inputs[8];
    const float* expanded[8];
    const bool use_expanded = expanded_group_inputs(t);
    int routes[8];
    for (int r = 0; r < tile.rows; ++r) {
      routes[r] = group_routes[tile.begin + r];
      inputs[r] = down ? g_scratch.data() + static_cast<size_t>(routes[r]) * I
                       : t->x + static_cast<size_t>(routes[r] / top_k) * H;
      if (use_expanded)
        expanded[r] = down ? group_intermediate_float.data() + static_cast<size_t>(routes[r]) * I
                           : group_input_float.data() + static_cast<size_t>(routes[r] / top_k) * H;
    }
    for (int out = first; out < last; ++out) {
      float a[8], b[8];
      const auto w = bank + (static_cast<size_t>(tile.expert) * (down ? H : 2 * I) + out) * (down ? I : H);
#if CPU_MOE_X86
      if (use_expanded) tile_dot_float_avx512f(w, expanded, a, down ? I : H, tile.rows);
      else
#endif
      if (tile.rows == 1) a[0] = dot(w, inputs[0], down ? I : H);
      else (t->grouped_compat ? tile_dot_compat : tile_dot)(w, inputs, a, down ? I : H, tile.rows);
      if (!down) {
#if CPU_MOE_X86
        if (use_expanded) tile_dot_float_avx512f(w + static_cast<size_t>(I) * H, expanded, b, H, tile.rows);
        else
#endif
        if (tile.rows == 1) b[0] = dot(w + static_cast<size_t>(I) * H, inputs[0], H);
        else (t->grouped_compat ? tile_dot_compat : tile_dot)(w + static_cast<size_t>(I) * H, inputs, b, H, tile.rows);
      }
      for (int r = 0; r < tile.rows; ++r) {
        const size_t route = routes[r];
        if (down) {
          route_output[route * H + out] = a[r];
        } else {
          const float weight = apply_on_input ? t->w[route] : 1.0f;
          // Match the GPU's BF16 GateUp output before applying the activation.
          float gate = a[r] * weight;
          float up = b[r] * weight;
          if (!t->grouped_compat) {
            gate = bf16_to_f32(f32_to_bf16(gate));
            up = bf16_to_f32(f32_to_bf16(up));
          }
          if (act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP) {
            gate = std::min(gate, swiglu_limit);
            up = std::max(-swiglu_limit, std::min(up, swiglu_limit));
            g_scratch[route * I + out] = f32_to_bf16(
                gate / (1.0f + std::exp(-gate * swiglu_alpha)) * (up + (act == ACT_SWIGLUOAI ? 1 : 0)));
          } else {
            g_scratch[route * I + out] = f32_to_bf16(act_apply(act, gate) * up);
          }
          if (use_expanded)
            group_intermediate_float[route * I + out] = bf16_to_f32(g_scratch[route * I + out]);
        }
      }
    }
  }

  void grouped_reduce(const MoeTask* t, int64_t p) {
    const int tok = static_cast<int>(p / n_hblk), first = (p % n_hblk) * HBLK;
    for (int h = first; h < std::min(H, first + HBLK); ++h) {
      float sum = 0;
      for (int k = 0; k < top_k; ++k) {
        const int r = tok * top_k + k, e = t->ids[r];
        if (e >= 0 && e < num_experts && (t->grouped_compat || t->w[r] != 0)) {
          if (t->grouped_compat) {
            sum += route_output[static_cast<size_t>(r) * H + h] * (apply_on_input ? 1 : t->w[r]);
          } else {
            const float weighted = route_output[static_cast<size_t>(r) * H + h] * (apply_on_input ? 1 : t->w[r]);
            sum += bf16_to_f32(f32_to_bf16(weighted));
          }
        }
      }
      t->y[static_cast<size_t>(tok) * H + h] = f32_to_bf16(sum);
    }
  }

  const char* isa_name() const { return isa; }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      while (bar_sense.load() != local_sense) {
#if CPU_MOE_X86
        _mm_pause();
#endif
      }
    }
  }

  void do_pass1(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? t->w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* gate_up_l = reinterpret_cast<const bf16_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const bf16_t* x_row = t->x + (size_t)tok * H;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const bool clamped = act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP;
    const float up_bias = act == ACT_SWIGLUOAI ? 1.0f : 0.0f;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int i = i0; i < i1; ++i) {
      // gate = row i, up = row I+i
      float gate =
          dot(gate_up_l + ((size_t)e * (2 * I) + i) * H, x_row, H) * w_in;
      float up = dot(gate_up_l + ((size_t)e * (2 * I) + I + i) * H, x_row, H) * w_in;
      if (clamped) {
        // clamp(gate, max=lim) * sigmoid(alpha * gate) * (clamp(up, +-lim) + up_bias)
        // -- swigluoai carries the +1 up bias (gpt-oss/MiniMax); swiglu_clamp
        // (GLM-5.3) does not. lim == +inf: no clamp.
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
        const float glu = gate / (1.0f + std::exp(-gate * alpha));
        g_row[i] = f32_to_bf16(glu * (up + up_bias));
      } else {
        g_row[i] = f32_to_bf16(act_apply(act, gate) * up);
      }
    }
  }

  void do_pass2(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* down_l = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float w_out = apply_on_input ? 1.0f : t->w[static_cast<size_t>(tok) * top_k + k];
        const size_t gr = (size_t)tok * top_k + k;
        const bf16_t* g_row = g_scratch.data() + gr * I;
        acc += dot(down_l + ((size_t)e * H + h) * I, g_row, I) * w_out;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void set_expert_grouped(bool enabled) {
    if (enabled && (apply_on_input))
      throw std::invalid_argument("expert grouping requires BF16 output-weighted experts");
    expert_grouped = enabled;
  }

  void legacy_pass(const MoeTask* t, int64_t p, bool down_pass) {
    const int blocks = down_pass ? n_hblk : n_iblk;
    const int block = p % blocks, e = legacy_experts[p / blocks];
    if (!down_pass) {
      for (int route : legacy_routes[e]) do_pass1(t, static_cast<int64_t>(route) * n_iblk + block);
      return;
    }
    const auto* down = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    for (int h = block * HBLK; h < std::min(H, (block + 1) * HBLK); ++h)
      for (int route : legacy_routes[e]) {
        const bf16_t* g = g_scratch.data() + static_cast<size_t>(route) * I;
        legacy_down[static_cast<size_t>(route) * H + h] =
            dot(down + ((size_t)e * H + h) * I, g, I);
      }
  }

  void legacy_reduce(const MoeTask* t, int64_t p) {
    const int tok = p / n_hblk, hb = p % n_hblk;
    for (int h = hb * HBLK; h < std::min(H, (hb + 1) * HBLK); ++h) {
      float acc = 0;
      for (int k = 0; k < top_k; ++k) {
        const size_t route = static_cast<size_t>(tok) * top_k + k;
        if (t->ids[route] >= 0 && t->ids[route] < num_experts)
          acc += legacy_down[route * H + h] * t->w[route];
      }
      t->y[static_cast<size_t>(tok) * H + h] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MoeTask* t, int tid) {
    int local_sense = 0;
    const int node = tid < num_threads / 2 ? 0 : 1;
    auto& gate_next = t->grouped_numa ? group_nodes[node].gate_next : p1_next;
    const auto gate_end = t->grouped_numa ? group_nodes[node].gate_end : p1_total;
    auto& down_next = t->grouped_numa ? group_nodes[node].down_next : p2_next;
    const auto down_end = t->grouped_numa ? group_nodes[node].down_end : p2_total;
    if (profile_workers) nvtxRangePushA("CPU/GateUp + Activation");
    for (;;) {
      int64_t p = gate_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= gate_end) break;
      if (t->grouped) grouped_pass(t, p, false);
      else if (expert_grouped) legacy_pass(t, p, false);
      else do_pass1(t, p);
    }
    if (profile_workers) {
      nvtxRangePop();
      nvtxRangePushA("CPU/Barrier/GateUp");
    }
    barrier(local_sense);
    if (profile_workers) nvtxRangePop();
    if (profile_workers) nvtxRangePushA("CPU/Down");
    for (;;) {
      int64_t p = down_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= down_end) break;
      if (t->grouped) grouped_pass(t, p, true);
      else if (expert_grouped) legacy_pass(t, p, true);
      else do_pass2(t, p);
    }
    if (profile_workers) nvtxRangePop();
    if (t->grouped || expert_grouped) {
      if (profile_workers) nvtxRangePushA("CPU/Barrier/Down");
      barrier(local_sense);
      if (profile_workers) {
        nvtxRangePop();
        nvtxRangePushA("CPU/Ordered Reduce");
      }
      for (;;) {
        const int64_t p = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (p >= prt_total) break;
        if (t->grouped) grouped_reduce(t, p);
        else legacy_reduce(t, p);
      }
      if (profile_workers) nvtxRangePop();
    }
  }

  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MoeTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      run_task_body(t, tid);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MoeTask* t) {
    if (t->grouped) build_groups(t);
    if (expanded_group_inputs(t)) {
      // Expand each BF16 input once; output channels retain the original dot reduction.
      const size_t inputs = static_cast<size_t>(t->num_tokens) * H;
      group_input_float.resize(inputs);
      group_intermediate_float.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
      for (size_t i = 0; i < inputs; ++i) group_input_float[i] = bf16_to_f32(t->x[i]);
    }
    bool active = false;
    for (int i = 0; i < t->num_tokens * top_k; ++i) {
      if (t->ids[i] >= 0 && t->ids[i] < num_experts && (!t->grouped || t->grouped_compat || t->w[i] != 0)) {
        active = true;
        break;
      }
    }
    if (!active) {
      // The previous task has drained before this submit; sync can keep its generation.
      std::fill(t->y, t->y + static_cast<size_t>(t->num_tokens) * H, bf16_t{0});
      return;
    }
    n_iblk = (I + IBLK - 1) / IBLK;
    n_hblk = (H + HBLK - 1) / HBLK;
    // Grow the per-token intermediate scratch if a larger batch shows up than the
    // construction-time hint (CUDA-graph capture warms the largest bs first, so
    // this happens at most once, before any capture, while the pool is idle).
    const size_t need = static_cast<size_t>(t->num_tokens) * top_k * I;
    if (need > g_scratch.size()) g_scratch.resize(need);
    p1_total = static_cast<int64_t>(t->num_tokens) * top_k * n_iblk;
    p2_total = static_cast<int64_t>(t->num_tokens) * n_hblk;
    prt_total = 0;
    if (t->grouped) {
      p1_total = static_cast<int64_t>(group_tiles) * n_iblk;
      p2_total = static_cast<int64_t>(group_tiles) * n_hblk;
      prt_total = static_cast<int64_t>(t->num_tokens) * n_hblk;
      if (t->grouped_numa) {
        // Match expert halves to worker groups; the caller arranges first touch.
        int split = 0;
        while (split < group_tiles && group_tasks[split].expert < num_experts / 2) ++split;
        group_nodes[0].gate_next.store(0, std::memory_order_relaxed);
        group_nodes[0].down_next.store(0, std::memory_order_relaxed);
        group_nodes[0].gate_end = static_cast<int64_t>(split) * n_iblk;
        group_nodes[0].down_end = static_cast<int64_t>(split) * n_hblk;
        group_nodes[1].gate_next.store(group_nodes[0].gate_end, std::memory_order_relaxed);
        group_nodes[1].down_next.store(group_nodes[0].down_end, std::memory_order_relaxed);
        group_nodes[1].gate_end = p1_total;
        group_nodes[1].down_end = p2_total;
      }
    }
    if (expert_grouped && !t->grouped) {
      legacy_routes.assign(num_experts, {});
      legacy_experts.clear();
      for (int route = 0; route < t->num_tokens * top_k; ++route) {
        const int e = t->ids[route];
        if (e >= 0 && e < num_experts) legacy_routes[e].push_back(route);
      }
      for (int e = 0; e < num_experts; ++e)
        if (!legacy_routes[e].empty()) legacy_experts.push_back(e);
      legacy_down.resize(static_cast<size_t>(t->num_tokens) * top_k * H);
      p1_total = static_cast<int64_t>(legacy_experts.size()) * n_iblk;
      p2_total = static_cast<int64_t>(legacy_experts.size()) * n_hblk;
      prt_total = static_cast<int64_t>(t->num_tokens) * n_hblk;
    }
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  void sync() {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &SpeculativeBf16Executor::submit_cb,
                       reinterpret_cast<void*>(task));
  }

  void sync_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &SpeculativeBf16Executor::sync_cb,
                       reinterpret_cast<void*>(task));
  }

  // Register a (layer, batch-size) slot's task so the coordinator can dispatch it on a
  // flag bump.
  void register_flag_task(int slot, uintptr_t task) {
    std::lock_guard<std::mutex> lk(flag_task_mtx);
    if (static_cast<int>(flag_task.size()) <= slot) flag_task.resize(slot + 1, nullptr);
    flag_task[slot] = reinterpret_cast<MoeTask*>(task);
  }

  int64_t flag_served_count(int slot) const {
    return (slot >= 0 && slot < static_cast<int>(flag_served.size())) ? flag_load_acquire(&flag_served[slot]) : 0;
  }

  // Start the busy-poll coordinator over the mapped-pinned flag arrays. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void start_flag_coordinator(uintptr_t ready_ptr, uintptr_t done_ptr, int num_slots,
                              int pin_core) {
    ready_flags = reinterpret_cast<volatile int64_t*>(ready_ptr);
    done_flags = reinterpret_cast<volatile int64_t*>(done_ptr);
    coord_num_slots = num_slots;
    {
      std::lock_guard<std::mutex> lk(flag_task_mtx);
      if (static_cast<int>(flag_task.size()) < num_slots) flag_task.resize(num_slots, nullptr);
    }
    flag_served.assign(num_slots, 0);
    coord_stop.store(false);
    coord_thread = std::thread([this, pin_core] {
#if CPU_MOE_HAS_AFFINITY
      if (pin_core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      coordinator_loop();
    });
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with slot
    // count and pause cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including host-func-only
    // stretches (prefill bursts, --moe-cpu-layers subsets) -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int L = 0; L < coord_num_slots; ++L) {
        // Binary handshake (memop-compatible: the GPU-side WAIT compares against an
        // immediate baked at graph capture, so the protocol resets per step instead of
        // counting). Acquire: everything the GPU made visible before setting ready --
        // the D2H input copies -- is visible to the worker pool after this read.
        if (flag_load_acquire(&ready_flags[L]) != 0) {
          active_flag_slot.store(L, std::memory_order_release);
          flag_store_release(&ready_flags[L], 0);  // consume this step's doorbell
          MoeTask* t;
          {
            std::lock_guard<std::mutex> lk(flag_task_mtx);
            t = (L < static_cast<int>(flag_task.size())) ? flag_task[L] : nullptr;
          }
          if (t != nullptr) {
            char label[96];
            if (profile_workers) {
              std::snprintf(label, sizeof(label), "CPU_MOE/layer=%d/rows=%d", t->layer_id, t->num_tokens);
              nvtxRangePushA(label);
            }
            submit(t);
            sync();
            if (profile_workers) nvtxRangePop();
          }
          // Release: the workers' y stores are visible before the GPU sees done.
          flag_store_release(&done_flags[L], 1);
          if (L < static_cast<int>(flag_served.size()))
            flag_store_release(&flag_served[L], flag_load_acquire(&flag_served[L]) + 1);
          active_flag_slot.store(-1, std::memory_order_release);
          any = true;
        }
      }
      if (any) {
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-scan all slots, ~20% of a core at 992 slots).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
#if CPU_MOE_X86
          _mm_pause();
#endif
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: release any in-flight (or future) spin-wait immediately so a replay
    // caught mid-shutdown exits its sync kernel now instead of owning the watchdog
    // stall. Runs before the destructor's join() returns, while the flag arrays are
    // still alive on the Python side.
    for (int L = 0; L < coord_num_slots; ++L) {
      flag_store_release(&done_flags[L], INT64_MAX);
    }
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run_task(uintptr_t task) {
    MoeTask* t = reinterpret_cast<MoeTask*>(task);
    submit(t);
    sync();
  }

  static void CUDART_CB submit_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->submit(t);
  }
  static void CUDART_CB sync_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->sync();
  }
};

}  // namespace

#include "cpu_target.h"
#include "cpu_target_v2.h"

void bind_speculative_bf16(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("target_linear", &cpu_target::linear, py::call_guard<py::gil_scoped_release>());
  m.def("target_linear_v2", &cpu_target::linear_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_norm_v2", &cpu_target::norm_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_rope_v2", &cpu_target::rope_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_gate_v2", &cpu_target::gate_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_gdn", &cpu_target::gdn<false>, py::call_guard<py::gil_scoped_release>());
  m.def("target_gdn_v2", &cpu_target::gdn<true>, py::call_guard<py::gil_scoped_release>());
  m.def("target_replay", &cpu_target::replay<false>, py::call_guard<py::gil_scoped_release>());
  m.def("target_replay_v2", &cpu_target::replay<true>, py::call_guard<py::gil_scoped_release>());
  m.def("target_attention", &cpu_target::attention, py::call_guard<py::gil_scoped_release>());
  m.def("target_attention_v2", &cpu_target::attention_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_attention_v2_profiled", &cpu_target::attention_v2_profiled, py::call_guard<py::gil_scoped_release>());
  m.def("target_attention_workspace_v2", &cpu_target::attention_workspace_v2, py::call_guard<py::gil_scoped_release>());
  m.def("target_isa", []() { return cpu_target::Projection().isa; });
  m.def("target_parallel_backend", []() {
#if AT_PARALLEL_OPENMP
    return "openmp";
#else
    return "native";
#endif
  });
  m.def("target_parallel_threads", []() {
    std::atomic<uint64_t> mask{0};
    at::parallel_for(0, 4096, 1, [&](int64_t begin, int64_t end) {
      mask.fetch_or(uint64_t{1} << std::min(63, at::get_thread_num()));
    });
    return __builtin_popcountll(mask.load());
  }, py::call_guard<py::gil_scoped_release>());
  py::class_<SpeculativeBf16Executor>(m, "SpeculativeBf16Executor")
      .def(py::init<int, int, int, int, int, int, int, int, int, uintptr_t, uintptr_t,
                    double, double, std::vector<int>>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("gate_up_ptr"), py::arg("down_ptr"),
           py::arg("swiglu_alpha"), py::arg("swiglu_limit"), py::arg("core_ids"))
      .def("create_task", &SpeculativeBf16Executor::create_task, py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("prepare_grouped", &SpeculativeBf16Executor::prepare_grouped)
      .def("set_expert_grouped", &SpeculativeBf16Executor::set_expert_grouped)
      .def("create_grouped_task", &SpeculativeBf16Executor::create_grouped_task)
      .def("create_grouped_compat_task", [](SpeculativeBf16Executor& e, int layer, int rows,
            uintptr_t x, uintptr_t ids, uintptr_t w, uintptr_t y) {
        auto task = e.create_grouped_task(layer, rows, x, ids, w, y);
        reinterpret_cast<MoeTask*>(task)->grouped_compat = true;
        return task;
      })
      .def("create_grouped_reuse_task", [](SpeculativeBf16Executor& e, int layer, int rows,
            uintptr_t x, uintptr_t ids, uintptr_t w, uintptr_t y) {
        auto task = e.create_grouped_task(layer, rows, x, ids, w, y);
        reinterpret_cast<MoeTask*>(task)->grouped_compat = true;
        reinterpret_cast<MoeTask*>(task)->grouped_reuse = true;
        return task;
      })
      .def("create_grouped_numa_task", [](SpeculativeBf16Executor& e, int layer, int rows,
            uintptr_t x, uintptr_t ids, uintptr_t w, uintptr_t y) {
        TORCH_CHECK(e.num_threads >= 2 && e.num_threads % 2 == 0 && e.num_experts % 2 == 0,
                    "NUMA expert schedule requires equal worker and expert halves");
        auto task = e.create_grouped_task(layer, rows, x, ids, w, y);
        reinterpret_cast<MoeTask*>(task)->grouped_compat = true;
        reinterpret_cast<MoeTask*>(task)->grouped_reuse = true;
        reinterpret_cast<MoeTask*>(task)->grouped_numa = true;
        return task;
      })
      .def("create_grouped_packed_task", [](SpeculativeBf16Executor& e, int layer, int rows,
            uintptr_t x, uintptr_t ids, uintptr_t w, uintptr_t y) {
#if CPU_MOE_X86
        TORCH_CHECK(__builtin_cpu_supports("avx512f"), "packed CPU Target requires AVX-512F");
#else
        TORCH_CHECK(false, "packed CPU Target requires AVX-512F");
#endif
        auto task = e.create_grouped_task(layer, rows, x, ids, w, y);
        reinterpret_cast<MoeTask*>(task)->grouped_compat = true;
        reinterpret_cast<MoeTask*>(task)->grouped_packed = true;
        return task;
      })
      .def("submit_with_cuda_stream", &SpeculativeBf16Executor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &SpeculativeBf16Executor::sync_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("run_task", &SpeculativeBf16Executor::run_task, py::arg("task"),
           py::call_guard<py::gil_scoped_release>())
      .def("register_flag_task", &SpeculativeBf16Executor::register_flag_task,
           py::arg("slot"), py::arg("task"))
      .def("flag_served_count", &SpeculativeBf16Executor::flag_served_count, py::arg("slot"))
      .def("flag_active_slot", [](SpeculativeBf16Executor& e) {
        return e.active_flag_slot.load(std::memory_order_acquire);
      })
      .def("worker_dispatch_count", [](SpeculativeBf16Executor& e) {
        return e.submitted.load(std::memory_order_acquire);
      })
      .def("start_flag_coordinator", &SpeculativeBf16Executor::start_flag_coordinator,
           py::arg("ready_ptr"), py::arg("done_ptr"), py::arg("num_slots"),
           py::arg("pin_core"))
      .def("set_profile_workers", [](SpeculativeBf16Executor& e, bool value) { e.profile_workers = value; })
      .def("isa_name", &SpeculativeBf16Executor::isa_name);
  m.def("speculative_memops_probe", &cumemops_probe, py::arg("stream"), py::arg("scratch_addr"));
  m.def("speculative_memop_submit", &cumemop_submit, py::arg("stream"), py::arg("done_addr"),
        py::arg("ready_addr"), py::arg("slot"));
  m.def("speculative_memop_sync", &cumemop_sync, py::arg("stream"), py::arg("done_addr"),
        py::arg("slot"));
}
