"""Last-query value influence for TableMoE Prefill in Gated DeltaNet layers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _last_query_coefficients(Q, K, G, Beta, Cu, Coeff,
                             HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
                             QS: tl.constexpr, KS: tl.constexpr,
                             GS: tl.constexpr, BS: tl.constexpr, BLOCK: tl.constexpr):
    seq, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(Cu + seq), tl.load(Cu + seq + 1)
    if start == end:
        return
    kh = head // (HV // HK)
    ds = tl.arange(0, BLOCK)
    q = tl.load(Q + (end - 1) * QS + kh * D + ds, ds < D, 0).to(tl.float32)
    q = (q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6)).to(Q.dtype.element_ty).to(tl.float32)
    adjoint = q * (D ** -0.5)
    for step in range(end - start):
        row = end - 1 - step
        k = tl.load(K + row * KS + kh * D + ds, ds < D, 0).to(tl.float32)
        k = (k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)).to(K.dtype.element_ty).to(tl.float32)
        beta = tl.load(Beta + row * BS + head).to(tl.float32)
        coefficient = beta * tl.sum(adjoint * k, 0)
        tl.store(Coeff + row * HV + head, coefficient)
        decay = tl.exp(tl.load(G + row * GS + head).to(tl.float32))
        # Reverse the decay and delta overwrite without touching the live recurrent state.
        adjoint = decay * (adjoint - coefficient * k)


def last_token_gdn_scores(q, k, g, beta, cu_seqlens):
    """Mean absolute last-query value coefficients, within each current request chunk.

    For S_t = exp(g_t) (I - beta_t k_t k_t^T) S_(t-1) + beta_t k_t v_t^T,
    reverse the final query through the state transition to recover each value's
    signed coefficient. Magnitudes supply nonnegative expert importance weights.
    """
    if (q.ndim != 3 or k.shape != q.shape or g.ndim != 2 or beta.shape != g.shape
            or q.shape[0] != g.shape[0] or g.shape[1] % q.shape[1]):
        raise ValueError("incompatible GDN TableMoE score dimensions")
    coefficients = torch.empty(g.shape, dtype=torch.float32, device=q.device)
    _last_query_coefficients[(cu_seqlens.numel() - 1, g.shape[1])](
        q, k, g, beta, cu_seqlens, coefficients,
        q.shape[1], g.shape[1], q.shape[2], q.stride(0), k.stride(0),
        g.stride(0), beta.stride(0), triton.next_power_of_2(q.shape[2]),
        num_warps=4, enable_fp_fusion=False)
    return coefficients.abs().mean(-1)
