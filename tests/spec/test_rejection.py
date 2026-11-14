"""Rejection sampler and verify-row sampling: greedy prefix match, per-row sampling params and per-row penalties."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
VOCAB = 8


def test_tree_sampling_penalties_exclude_siblings_and_count_repeated_ancestors():
    from freetoken.engine.sample import BatchSamplingArgs, Sampler
    from freetoken.engine.speculative_tree import TreeProposal, tree_paths
    from freetoken.spec.mtp.tree import sample_tree

    torch.manual_seed(71)
    parents = torch.tensor([-1, 0, 0, 1, 2, 3, 4])
    paths, depths = tree_paths(parents, 3)
    trees = [TreeProposal(torch.tensor(0), torch.tensor(tokens), parents, paths, depths)
             for tokens in ([2, 3, 2, 4, 5, 3], [5, 2, 4, 2, 4, 5])]
    counts = torch.tensor([0, 0, 2, 0, 0, 1, 0, 0])
    logits = torch.randn(14, 8)
    args = BatchSamplingArgs(None, penalties=[(i, counts, 0.7, 0.4) for i in range(2)])
    actual = sample_tree(logits, args, trees, Sampler(torch.device("cpu"), 8))
    for i, tree in enumerate(trees):
        ids = tree.verify_inputs()
        for node in range(7):
            seen = counts.clone()
            parent = node
            while parent > 0:
                seen[ids[parent]] += 1
                parent = int(parents[parent])
            expected = logits[i * 7 + node] - 0.4 * seen - 0.7 * (seen > 0)
            assert actual[i, node] == expected.argmax()
    assert counts.tolist() == [0, 0, 2, 0, 0, 1, 0, 0]


@pytest.mark.parametrize("budget", [2, 7])
@pytest.mark.parametrize("bs", [1, 3])
def test_mtp_tree_keeps_unexpanded_children_and_uses_cumulative_probability(budget, bs):
    from freetoken.engine.speculative_tree import build_mtp_tree, build_mtp_trees

    probability = torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.7, 0.2, 0.1]])
    visited = []

    def forward(ids, hidden, paths, depths, slots):
        visited.append(ids.tolist())
        return ids[:, None].long()

    hidden = torch.arange(bs)[:, None]
    if bs == 1:
        actual = [build_mtp_tree(torch.tensor(2), hidden, 3, 2, budget,
                   lambda h: probability[h[:, 0]].log(), lambda t: t, forward)]
    else:
        actual = build_mtp_trees(torch.full((bs,), 2), hidden, 3, 2, budget,
                   lambda h: probability[h[:, 0]].log(), lambda t: t, forward)
    for request, tree in enumerate(actual):
        # Independent scalar beam search with all children retained for pruning.
        nodes = [(2, -1, 1.0)]
        beam = [0]
        for step in range(3):
            candidates = []
            for parent in beam:
                distribution = probability[request if parent == 0 else nodes[parent][0]].tolist()
                for token in sorted(range(3), key=lambda t: -distribution[t])[:2]:
                    candidates.append(len(nodes))
                    nodes.append((token, parent, nodes[parent][2] * distribution[token]))
            beam = sorted(candidates, key=lambda i: -nodes[i][2])[:2]
        selected = [0] + sorted(sorted(range(1, len(nodes)), key=lambda i: -nodes[i][2])[:budget - 1])
        assert tree.verify_inputs().tolist() == [nodes[i][0] for i in selected]
        assert tree.parents.tolist() == [-1] + [selected.index(nodes[i][1]) for i in selected[1:]]
    assert len(visited) == 2 and all(len(v) == bs * 2 for v in visited)


@pytest.mark.parametrize("remaining,eos", [(8, ()), (2, ()), (8, (6,))])
def test_tree_commit_follows_target_children_not_flattened_neighbors(remaining, eos):
    from freetoken.engine.speculative import VerifyResult
    from freetoken.engine.speculative_tree import TreeProposal, tree_paths, greedy_tree_commit

    # Two siblings predict the same token; only the selected parent's child is eligible.
    parents = torch.tensor([-1, 0, 0, 1, 2, 4, 4])
    ids = torch.tensor([0, 3, 5, 6, 6, 2, 7], dtype=torch.int32)
    predictions = torch.tensor([5, 6, 6, 2, 7, 1, 4], dtype=torch.int32)
    paths, depths = tree_paths(parents, 3)
    proposal = TreeProposal(ids[:1], ids[1:], parents, paths, depths)
    commit, path = greedy_tree_commit(proposal, VerifyResult(predictions, torch.empty(7, 0)),
                                      remaining=remaining, eos_ids=eos)
    want, visited, node = [], [], 0
    while len(want) < remaining:
        visited.append(node)
        token = int(predictions[node])
        want.append(token)
        children = [i for i, p in enumerate(parents.tolist()) if p == node and ids[i] == token]
        if token in eos or not children:
            break
        node = children[0]
    assert commit.tokens[:commit.input_count].tolist() == want
    assert path[:commit.input_count].tolist() == visited
    assert bool(commit.terminal) == (len(want) == remaining or want[-1] in eos)
    for i, depth in enumerate(depths.tolist()):
        chain, node = [], i
        while node >= 0:
            chain.insert(0, node)
            node = int(parents[node])
        assert paths[i, :depth + 1].tolist() == chain


def test_tree_budget_keeps_ancestors_when_path_scores_tie():
    from freetoken.engine.speculative_tree import select_tree

    parents = torch.tensor([0, 0, 1, 2, 3, 4])
    scores = torch.tensor([0.5, 0.5, 0.5, 0.25, 0.5, 0.125])
    result = select_tree(torch.tensor([99]), torch.arange(6), parents, scores, 5, 3)
    assert result.tokens.tolist() == [0, 1, 2, 4]
    assert result.parents.tolist() == [-1, 0, 0, 1, 3]
    assert result.paths[-1].tolist() == [0, 1, 3, 4]


def _sampler(device, vocab=VOCAB):
    from freetoken.engine.sample import Sampler

    return Sampler(device=device, vocab_size=vocab)


@cuda
def test_greedy_prefix_match_exact():
    from freetoken.spec.rejection import reject
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    k = 3
    # Target argmax sequence: [3, 1, 2, 0] over 4 positions.
    logits = torch.full((k + 1, VOCAB), -10.0, device=device)
    for pos, t in enumerate([3, 1, 2, 0]):
        logits[pos, t] = 10.0
    args = BatchSamplingArgs(temperatures=None)
    sampler = _sampler(device)

    cases = [
        ([3, 1, 2], 3, [3, 1, 2, 0]),   # all accepted + bonus
        ([3, 1, 7], 2, [3, 1, 2]),      # mismatch at pos 2 -> recovery = target argmax
        ([7, 1, 2], 0, [3]),            # first draft rejected
    ]
    for draft, want_emitted, want_tokens in cases:
        d = torch.tensor([draft], device=device)
        out, emitted = reject(logits, d, args, sampler)
        assert int(emitted[0]) == want_emitted
        assert out[0, : want_emitted + 1].tolist() == want_tokens


@cuda
def test_each_verify_row_counts_the_drafts_before_it_as_committed():
    """Row j penalizes the committed counts plus drafts d1..dj, frequency per occurrence and presence once per token; the counts stay as they were."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(1)
    vocab, presence, frequency = 32, 0.7, 0.4
    drafts = torch.tensor([[5, 9, 5]], device=device)  # d3 repeats d1
    counts = torch.zeros(vocab, dtype=torch.int32, device=device)
    counts[9] = 2  # already committed, so presence must not apply to it again
    logits = torch.randn(drafts.shape[1] + 1, vocab, device=device)
    args = BatchSamplingArgs(temperatures=None, penalties=[(0, counts, presence, frequency)])

    sampler = _sampler(device, vocab)
    tokens = sampler.sample_verify(logits, args, drafts)
    sampled = BatchSamplingArgs(temperatures=torch.ones(1, device=device), penalties=args.penalties)
    probs = sampler.probs(logits, sampled, drafts)

    for j in range(drafts.shape[1] + 1):
        seen = counts.clone()
        for d in drafts[0, :j].tolist():
            seen[d] += 1
        expected = logits[j] - frequency * seen - presence * (seen > 0)
        assert int(tokens[j]) == int(expected.argmax()), f"row {j}"
        # the verify with draft probs reads the same penalized rows
        assert torch.allclose(probs[j], torch.softmax(expected, dim=-1), atol=1e-6), f"row {j}"
    assert counts[9] == 2 and counts.sum() == 2


@cuda
@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_each_verify_row_samples_with_its_requests_params(tree):
    """Request i's sampling params cover its k+1 verify rows: a top_k=1 request and a greedy one give the argmax on every row, a sampled one does not; the draft-probs verify reads each row's softmax at its request's temperature, renormalized over its top-k and then its top-p."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, k, trials, bs = 32, 2, 16, 4
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.3, 1.0, 0.7], device=device),
        top_k=torch.tensor([1, vocab, vocab, 6], dtype=torch.int32, device=device),
        top_p=torch.tensor([1.0, 0.9, 1.0, 0.8], device=device),
        greedy_mask=torch.tensor([False, False, True, False], device=device),
    )
    drafts = torch.zeros(bs, k, dtype=torch.int64, device=device)
    off_argmax = []
    for _ in range(trials):
        logits = 0.1 * torch.randn(bs * (k + 1), vocab, device=device)
        if tree:
            from freetoken.engine.speculative_tree import TreeProposal, tree_paths
            from freetoken.spec.mtp.tree import sample_tree

            parents = torch.tensor([-1, 0, 0], device=device)
            paths, depths = tree_paths(parents, 1)
            trees = [TreeProposal(torch.tensor(0, device=device), row, parents, paths, depths) for row in drafts]
            tokens = sample_tree(logits, args, trees, _sampler(device, vocab)).flatten()
        else:
            tokens = _sampler(device, vocab).sample_verify(logits, args, drafts)
        off_argmax.append((tokens != logits.argmax(dim=-1)).view(bs, k + 1))
    off_argmax = torch.stack(off_argmax)
    assert not off_argmax[:, 0].any() and not off_argmax[:, 2].any()
    assert off_argmax[:, 1].any(dim=0).all()

    logits = torch.randn(bs * (k + 1), vocab, device=device)
    probs = _sampler(device, vocab).probs(logits, args, drafts)
    for row, i in enumerate(torch.arange(bs).repeat_interleave(k + 1).tolist()):
        want = torch.softmax(logits[row] / args.temperatures[i], dim=-1)
        want = torch.where(want >= want.topk(int(args.top_k[i])).values[-1], want, 0.0)
        # top-p keeps the smallest best-first set whose mass reaches top_p
        p_sorted, order = (want / want.sum()).sort(descending=True)
        p_sorted = torch.where(p_sorted.cumsum(0) - p_sorted < args.top_p[i], p_sorted, 0.0)
        want = torch.zeros_like(want).scatter(0, order, p_sorted / p_sorted.sum())
        if args.greedy_mask[i]:
            want = torch.nn.functional.one_hot(logits[row].argmax(), vocab).float()
        torch.testing.assert_close(probs[row], want, atol=1e-6, rtol=1e-5, msg=f"row {row}")


@cuda
def test_reject_with_draft_probs_commits_the_target_distribution():
    """Drafts drawn from a q over K candidates per slot, in a batch that mixes rows trusting q, rows whose q is unusable and greedy rows: the token committed at each position still follows the target p on both sampled kinds, and a greedy row commits the argmax prefix (Monte Carlo, k=2, V=6, K=3)."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.spec.rejection import reject_with_draft_probs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, K, k, n = 6, 3, 2, 6_000_000
    p = torch.softmax(torch.randn(k + 1, vocab, device=device) * 1.5, dim=-1)
    cand = torch.tensor([[0, 2, 5], [1, 2, 3]], device=device)
    q = torch.softmax(torch.randn(k, K, device=device), dim=-1)
    picks = torch.stack([torch.multinomial(q[j], n, replacement=True) for j in range(k)], dim=1)
    drafts = cand.gather(1, picks.T).T
    # rows cycle through: q trusted, q unusable (a NaN or a total of 2), greedy
    kind = torch.arange(n, device=device) % 3
    q_rows = q.expand(n, k, K).clone()
    unusable = (kind == 1).nonzero().squeeze(1)
    q_rows[unusable[::2], 0, 0] = float("nan")
    q_rows[unusable[1::2], 0] *= 2
    # softmax(log p) at T=1 hands the verify exactly p on every row
    logits = p.log().repeat(n, 1)
    args = BatchSamplingArgs(temperatures=torch.ones(n, device=device), greedy_mask=kind == 2)
    out, emitted = reject_with_draft_probs(logits, drafts, cand.expand(n, k, K).contiguous(), q_rows, args, _sampler(device, vocab))

    def marginal(tokens):
        return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()

    for rows in (kind == 0, kind == 1):
        rows_out, rows_emitted = out[rows], emitted[rows]
        accepted_first = rows_emitted >= 1
        assert int(accepted_first.sum()) > 100_000
        assert (marginal(rows_out[:, 0]) - p[0]).abs().max() < 5e-3
        assert (marginal(rows_out[accepted_first, 1]) - p[1]).abs().max() < 5e-3
    # both kinds draw the bonus token from p itself; a row that does not trust q rarely accepts every draft
    accepted_all = (kind < 2) & (emitted == k)
    assert int(accepted_all.sum()) > 100_000
    assert (marginal(out[accepted_all, k]) - p[k]).abs().max() < 5e-3
    greedy = kind == 2
    argmax = logits[: k + 1].argmax(dim=-1)
    assert torch.equal(emitted[greedy], (drafts[greedy] == argmax[:k]).cumprod(dim=1).sum(dim=1))
    committed = torch.arange(k + 1, device=device) <= emitted[greedy].unsqueeze(1)
    assert torch.equal(out[greedy][committed], argmax.expand(int(greedy.sum()), -1)[committed])
