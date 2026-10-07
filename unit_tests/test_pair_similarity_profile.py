from fractions import Fraction

import torch

from compression_topics.spatial.scripts import pair_similarity_profile as psp


def test_identical_pairs_are_fully_similar():
    x = torch.randn(10, 16)
    cos, rel = psp.pair_metrics(x, x.clone(), 0)
    assert torch.allclose(cos, torch.ones(10), atol=1e-5)
    assert torch.allclose(rel, torch.zeros(10), atol=1e-6)


def test_setting_aside_the_biggest_difference_removes_a_single_outlier():
    base = torch.randn(4, 16)
    other = base.clone()
    other[:, 3] += 10.0
    _, rel0 = psp.pair_metrics(base, other, 0)
    cos1, rel1 = psp.pair_metrics(base, other, 1)
    assert (rel0 > 0.1).all()
    assert torch.allclose(rel1, torch.zeros(4), atol=1e-6)
    assert torch.allclose(cos1, torch.ones(4), atol=1e-5)


def test_rel_is_difference_over_mean():
    a = torch.tensor([[3.0, 0.0]])
    b = torch.tensor([[1.0, 0.0]])
    _, rel = psp.pair_metrics(a, b, 0)
    assert abs(rel.item() - 0.5) < 1e-6  # |d| = 1, |m| = 2


def test_aligned_pairs_follow_absolute_positions():
    states = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
    first, second = psp.aligned_pairs(states, 0)
    assert first.flatten().tolist() == [0.0, 2.0, 4.0]
    assert second.flatten().tolist() == [1.0, 3.0, 5.0]
    # Chunk starting at odd position 3: first aligned pair is (4, 5).
    first, second = psp.aligned_pairs(states[:, :, 3:], 3)
    assert first.flatten().tolist() == [4.0]
    assert second.flatten().tolist() == [5.0]
    assert psp.aligned_pairs(states[:, :, :1], 0) is None


def test_histogram_fractions():
    h = psp.Histograms()
    h.add(torch.tensor([0.99, 0.6, 0.1]), torch.tensor([0.05, 0.25, 0.9]))
    summary = h.summary()
    assert summary["n"] == 3
    assert abs(summary["frac_rel_at_most"]["0.3"] - 2 / 3) < 1e-9
    assert abs(summary["frac_cos_at_least"]["0.5"] - 2 / 3) < 1e-9


def test_profile_runs_on_a_tiny_llama_with_prefill_and_decode():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    prompts = [torch.randint(0, 64, (9,)), torch.randint(0, 64, (12,))]
    profiler = psp.run_profile(
        model, prompts, max_new_tokens=10, rope=psp.rope_tables(model), fractions=(Fraction(0), Fraction(1, 8)),
        generate_kwargs={"min_new_tokens": 10},
    )
    summary = profiler.summary()
    assert set(summary) == {"k_rope", "k_plain", "v"}
    for kind in summary.values():
        assert set(kind) == {"prefill", "decode"}
    # Prompt of 9 gives 4 pairs and of 12 gives 6; 2 KV heads each, per layer.
    assert summary["v"]["prefill"]["0"]["0"]["n"] == (4 + 6) * 2
    # Generated tokens sit at positions 9..18 and 12..21; the 9-start gives 5 pairs of (10,11)..(17,18) -> 4.
    assert summary["v"]["decode"]["0"]["0"]["n"] > 0
    assert psp.markdown_tables(summary, fractions=(Fraction(0), Fraction(1, 8)))
    hist = profiler.histogram_dump()["hist"]["v"]["prefill"]["0"]["0"]
    assert hist["n"] == (4 + 6) * 2 and sum(hist["rel"]) <= hist["n"] and len(hist["cos"]) == psp.COS_BINS
    norms = profiler.norms_dump()["norms"]["v"]["prefill"]["0"]
    assert norms["count"] == (4 + 6) * 2 * 2 and norms["max"] >= norms["rms"] >= norms["mean"] > 0
    moves = profiler.moves_dump()["moves"]["v"]["prefill"]["0"]
    assert len(moves) == 2 and len(moves[0]) == len(psp.BUCKET_EDGES) + 1
    assert sum(cell[0] for cell in moves[0]) == (4 + 6) * 2
    arrays = profiler.sample_arrays()
    assert arrays["v/prefill/0/curve"].shape[1] == 9 and arrays["v/prefill/0/meta"].shape[1] == len(psp.Reservoir.META)
    assert arrays["v/prefill/0/curve"].shape[0] <= 2000
    dump = profiler.kneeded_dump()
    assert dump["head_dim"] == 8 and len(dump["taus"]) == 50
    row = dump["counts"]["v"]["prefill"]["0"]
    assert len(row) == 50 and len(row[0]) == 9
    assert sum(row[0]) == (4 + 6) * 2


def test_tier_counts_pick_the_smallest_residual_within_the_threshold():
    # ks = (0, 8, 16); three pairs: fine at k=0, fine only at k=16, never fine.
    rels = torch.tensor([[0.05, 0.9, 0.9], [0.04, 0.5, 0.9], [0.03, 0.1, 0.9]])
    counts = psp.tier_counts(rels, 0.2)
    assert counts.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_tier_summary_bytes_against_dense():
    # Half the pairs merge with no residual, half stay exact: (16 + 32) / 2 / 32 = 0.75.
    summary = psp.tier_summary(torch.tensor([1.0, 0.0, 1.0]), (0, 8), 128)
    assert abs(summary["bytes_fraction_of_dense"] - 0.75) < 1e-9
    # Every pair merged with k=8: (16 * 128 + 128 + 16 * 8) / (32 * 128).
    full = psp.tier_summary(torch.tensor([0.0, 4.0, 0.0]), (0, 8), 128)
    assert abs(full["bytes_fraction_of_dense"] - (16 * 128 + 128 + 128) / (32 * 128)) < 1e-9


def test_k_needed_counts_match_the_per_k_metrics():
    torch.manual_seed(1)
    a = torch.randn(200, 16)
    b = a + 0.3 * torch.randn(200, 16)
    b[:50, 5] += 4.0  # a few pairs with one big outlier difference
    taus = (0.1, 0.3, 0.6)
    counts = psp.k_needed_counts(a, b, taus)
    assert counts.shape == (3, 17)
    assert (counts.sum(dim=1) == 200).all()
    for row, tau in enumerate(taus):
        expected = torch.zeros(17, dtype=torch.float64)
        for k in range(17):
            _, rel = psp.pair_metrics(a, b, k)
            if k == 0:
                done = rel <= tau
                need = torch.where(done, torch.zeros(200), torch.full((200,), 99.0))
            else:
                need = torch.where((need == 99.0) & (rel <= tau), torch.full((200,), float(k)), need)
        for k in range(17):
            expected[k] = float((need == k).sum())
        assert torch.allclose(counts[row], expected), (tau, counts[row], expected)


def test_bytes_from_kneeded_agree_with_the_tier_tables():
    torch.manual_seed(2)
    a = torch.randn(300, 16)
    b = a + 0.4 * torch.randn(300, 16)
    ks = (0, 4, 8)
    rels = torch.stack([psp.pair_metrics(a, b, k)[1] for k in ks])
    old = psp.tier_summary(psp.tier_counts(rels, 0.3), ks, 16)["bytes_fraction_of_dense"]
    row = psp.k_needed_counts(a, b, (0.3,))[0]
    new, _ = psp.bytes_from_kneeded(row, ks, 16)
    assert abs(old - new) < 1e-9


def test_movement_sums_bucket_by_rel_before():
    torch.manual_seed(3)
    a = torch.randn(400, 16)
    b = a + 0.5 * torch.randn(400, 16)
    ks = (0, 4)
    rels = torch.stack([psp.pair_metrics(a, b, k)[1] for k in ks])
    sums = psp.movement_sums(rels)
    assert sums.shape == (2, len(psp.BUCKET_EDGES) + 1, 4)
    assert (sums[:, :, 0].sum(dim=1) == 400).all()
    # k = 0 moves nothing: sum before equals sum after.
    assert torch.allclose(sums[0, :, 1], sums[0, :, 2])
    # Residuals only lower rel, so the average after never exceeds the average before.
    assert (sums[1, :, 2] <= sums[1, :, 1] + 1e-9).all()
    # Check one bucket against a direct computation.
    edges = torch.tensor((0.0,) + psp.BUCKET_EDGES + (1e9,))
    low, high = edges[3].item(), edges[4].item()
    mask = (rels[0] >= low) & (rels[0] < high)
    assert abs(sums[1, 3, 0].item() - mask.sum().item()) < 1e-9
    assert abs(sums[1, 3, 2].item() - rels[1][mask].sum().item()) < 1e-3


def test_rel_curves_match_pair_metrics_and_end_at_zero():
    torch.manual_seed(4)
    a = torch.randn(20, 12)
    b = a + 0.5 * torch.randn(20, 12)
    curves = psp.rel_curves(a, b)
    assert curves.shape == (20, 13)
    for k in (0, 3, 12):
        assert torch.allclose(curves[:, k], psp.pair_metrics(a, b, k)[1], atol=1e-4)
    assert (curves[:, :-1] >= curves[:, 1:] - 1e-6).all()


def test_reservoir_keeps_a_bounded_sample_with_metadata():
    torch.manual_seed(5)
    reservoir = psp.Reservoir(10)
    for question in range(3):
        a = torch.randn(40, 8)
        b = a + 0.2 * torch.randn(40, 8)
        head = torch.arange(40) // 20
        position = 2 * (torch.arange(40) % 20)
        reservoir.add(a, b, head, position, question)
    assert reservoir.curves.shape == (10, 9) and reservoir.meta.shape == (10, len(psp.Reservoir.META))
    assert set(reservoir.meta[:, 0].tolist()) <= {0.0, 1.0, 2.0}
    assert (reservoir.meta[:, 1] <= 1).all() and (reservoir.meta[:, 2] % 2 == 0).all()


def test_dump_kv_writes_the_exact_stored_states(tmp_path):
    from safetensors.torch import safe_open
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 64, (9,))
    psp.run_profile(
        model, [ids], max_new_tokens=6, rope=psp.rope_tables(model), fractions=(Fraction(0), Fraction(1, 8)),
        generate_kwargs={"min_new_tokens": 6}, sample_pairs=0, dump_dir=tmp_path,
    )
    with safe_open(str(tmp_path / "question_0000.safetensors"), framework="pt") as f:
        keys, values, queries = f.get_tensor("keys"), f.get_tensor("values"), f.get_tensor("queries")
        assert f.metadata()["prompt_len"] == "9"
        assert f.get_tensor("input_ids").tolist() == ids.tolist()
    # 9 prompt tokens plus 6 generated; the last generated token is never fed back.
    assert keys.shape == (2, 2, 14, 8) and values.shape == keys.shape
    assert queries.shape == (2, 4, 14, 8)
    # The stored keys are the post-RoPE keys the model attended with: rerun and compare layer 0.
    captured = {}
    from transformers.cache_utils import Cache

    original = Cache.update

    def spy(self, key_states, value_states, layer_idx, *a, **kw):
        if layer_idx == 0:
            captured.setdefault("k", []).append(key_states.detach())
        return original(self, key_states, value_states, layer_idx, *a, **kw)

    Cache.update = spy
    try:
        model.generate(input_ids=ids.unsqueeze(0), attention_mask=torch.ones(1, 9, dtype=torch.long),
                       max_new_tokens=6, min_new_tokens=6, do_sample=False)
    finally:
        Cache.update = original
    assert torch.allclose(torch.cat(captured["k"], dim=-2)[0], keys[0], atol=1e-6)


def test_reservoir_and_norm_dump_record_token_norms():
    torch.manual_seed(6)
    a = torch.randn(30, 8)
    b = a + 0.3 * torch.randn(30, 8)
    reservoir = psp.Reservoir(30)
    reservoir.add(a, b, torch.zeros(30), torch.arange(30) * 2, 0)
    meta = dict(zip(psp.Reservoir.META, reservoir.meta.T))
    order = meta["position"].long() // 2
    assert torch.allclose(meta["norm_first"], a.norm(dim=-1)[order], atol=1e-5)
    assert torch.allclose(meta["norm_second"], b.norm(dim=-1)[order], atol=1e-5)
    assert torch.allclose(meta["norm_d"], ((a - b) / 2).norm(dim=-1)[order], atol=1e-5)


def test_fractions_resolve_to_whole_feature_counts():
    assert psp.resolve_ks((Fraction(0), Fraction(1, 8), Fraction(1, 2)), 128) == (0, 16, 64)
    assert psp.fraction_label(16, 128) == "1/8" and psp.fraction_label(0, 128) == "0"
    try:
        psp.resolve_ks((Fraction(1, 3),), 128)
    except ValueError:
        pass
    else:
        raise AssertionError("1/3 of 128 is not a whole number")
