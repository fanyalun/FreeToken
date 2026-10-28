import torch
import triton
import triton.language as tl


@triton.jit
def _project_max(
    x, weight, maxima, indices, valid_rows,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    offset = tl.arange(0, BK)
    count = tl.minimum(tl.load(valid_rows), M) if HAS_VALID_ROWS else M
    accumulator = tl.zeros((BM, BN), tl.float32)
    if tl.program_id(0) * BM < count:
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + offset
            activation = tl.load(x + rows[:, None] * K + k[None, :],
                                 (rows[:, None] < count) & (k[None, :] < K), 0)
            tile = tl.load(weight + columns[None, :] * K + k[:, None],
                           (columns[None, :] < N) & (k[:, None] < K), 0)
            accumulator = tl.dot(activation, tile, accumulator)
    # Argmax observes the same output dtype as the ordinary projection.
    logits = accumulator.to(x.dtype.element_ty).to(tl.float32)
    logits = tl.where(rows[:, None] < count, logits, 0.0)
    logits = tl.where(columns[None, :] < N, logits, -float("inf"))
    first_nan = tl.min(tl.where(logits != logits, columns[None, :], 2147483647), 1)
    maximum = tl.max(tl.where(logits == logits, logits, -float("inf")), 1)
    winner = tl.min(tl.where(logits == maximum[:, None], columns[None, :], 2147483647), 1)
    maximum = tl.where(first_nan < N, float("nan"), maximum)
    winner = tl.where(first_nan < N, first_nan, winner)
    blocks = tl.cdiv(N, BN)
    address = rows * blocks + tl.program_id(1)
    tl.store(maxima + address, maximum, rows < M)
    tl.store(indices + address, winner, rows < M)


@triton.jit
def _merge_max(maxima, indices, output, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(maxima + row * BLOCKS + offsets, offsets < BLOCKS, -float("inf"))
    tokens = tl.load(indices + row * BLOCKS + offsets, offsets < BLOCKS, 2147483647)
    first_nan = tl.min(tl.where(values != values, tokens, 2147483647), 0)
    maximum = tl.max(tl.where(values == values, values, -float("inf")), 0)
    winner = tl.min(tl.where(values == maximum, tokens, 2147483647), 0)
    winner = tl.where(first_nan < 2147483647, first_nan, winner)
    tl.store(output + row, winner)


def greedy_head(x: torch.Tensor, weight: torch.Tensor, valid_rows=None, *, tile_rows=None) -> torch.Tensor:
    """BF16 short-window projection and greedy selection without full logits storage."""
    rows, hidden = x.shape
    vocab = weight.shape[0]
    blocks = triton.cdiv(vocab, 128)
    maxima = torch.empty((rows, blocks), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, blocks), dtype=torch.int32, device=x.device)
    output = torch.empty(rows, dtype=torch.int64, device=x.device)
    tile_rows = (16 if rows <= 16 else 32) if tile_rows is None else tile_rows
    _project_max[(triton.cdiv(rows, tile_rows), blocks)](
        x, weight, maxima, indices, valid_rows, rows, vocab, hidden,
        BM=tile_rows, BN=128, BK=64, HAS_VALID_ROWS=valid_rows is not None,
        num_warps=4, num_stages=3,
    )
    _merge_max[(rows,)](maxima, indices, output, blocks, triton.next_power_of_2(blocks))
    return output
