import pytest
import torch

from compression_topics.spatial.algorithms import pair_rank


def _pairs(batch=2, heads=3, pairs=8, features=16, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(batch, heads, pairs, features) + 2.0
    # Each pair gets its own noise level, so the ranking is well defined.
    scale = torch.rand(batch, heads, pairs, 1) * 2
    b = a + scale * torch.randn_like(a)
    x = torch.empty(batch, heads, 2 * pairs, features)
    x[..., 0::2, :] = a
    x[..., 1::2, :] = b
    return x


def _distance(x, keep_pct=0):
    first, second = x[..., 0::2, :], x[..., 1::2, :]
    mean, delta = (first + second) / 2, (first - second) / 2
    kept = pair_rank._largest(delta, pair_rank.kept_features(x.shape[-1], keep_pct))
    return (delta - kept).norm(dim=-1) / mean.norm(dim=-1)


def _merged(x, out):
    return ~torch.isclose(out[..., 0::2, :], x[..., 0::2, :]).all(dim=-1)


@pytest.mark.parametrize("merge_pct", [0, 25, 50, 75, 100])
def test_merges_exactly_the_closest_share_of_each_sequence(merge_pct):
    x = _pairs()
    pair_rank.reset_stats()
    out = pair_rank.close_pairs(x, merge_pct=merge_pct, keep_pct=0, target="k", layer_idx=3)
    merged = _merged(x, out)
    per_sequence = 3 * 8
    assert (merged.reshape(2, -1).sum(dim=1) == round(per_sequence * merge_pct / 100)).all()
    distance = _distance(x)
    for row in range(2):
        chosen, rest = distance[row][merged[row]], distance[row][~merged[row]]
        if chosen.numel() and rest.numel():
            assert chosen.max() <= rest.min()  # most similar first, across heads
    stats = pair_rank.pop_stats()["k"]
    assert stats["pairs"] == 2 * per_sequence and stats["merged"] == int(merged.sum())


def test_merged_pairs_read_back_as_the_mean():
    x = _pairs()
    out = pair_rank.close_pairs(x, merge_pct=100, keep_pct=0)
    mean = (x[..., 0::2, :] + x[..., 1::2, :]) / 2
    assert torch.allclose(out[..., 0::2, :], mean) and torch.allclose(out[..., 1::2, :], mean)


def test_residual_keeps_the_largest_differences_and_ranks_after_it():
    torch.manual_seed(1)
    a = torch.randn(1, 1, 2, 16) + 3.0
    b = a + 0.3 * torch.randn_like(a)
    b[0, 0, 0, 5] += 8.0  # pair 0: one big outlier; far apart before the residual, close after
    x = torch.empty(1, 1, 4, 16)
    x[..., 0::2, :], x[..., 1::2, :] = a, b
    plain = pair_rank.close_pairs(x, merge_pct=50, keep_pct=0)
    with_residual = pair_rank.close_pairs(x, merge_pct=50, keep_pct=25)
    assert torch.equal(plain[..., 0:2, :], x[..., 0:2, :])  # without a residual pair 0 is the worse one
    assert not torch.equal(with_residual[..., 0:2, :], x[..., 0:2, :])  # with it, pair 0 ranks first
    # The kept entries come back exactly: the outlier feature is restored.
    assert torch.allclose(with_residual[0, 0, 1, 5], x[0, 0, 1, 5])


def test_later_pairs_use_the_prefill_threshold():
    x = _pairs(batch=1)
    pair_rank.reset_stats()
    pair_rank.close_pairs(x, merge_pct=50, keep_pct=0, target="k", layer_idx=0)
    threshold = pair_rank.THRESHOLDS[(0, "k")]
    distance = _distance(x)
    out = pair_rank.close_pairs(x, merge_pct=50, keep_pct=0, target="k", layer_idx=0, ranked=False)
    assert torch.equal(_merged(x, out), distance <= threshold)
    # A layer with no prefill threshold merges nothing later.
    untouched = pair_rank.close_pairs(x, merge_pct=50, keep_pct=0, target="k", layer_idx=7, ranked=False)
    assert torch.equal(untouched, x)


def test_odd_start_and_tail_token_are_left_alone():
    x = _pairs(batch=1)
    assert pair_rank.apply(x, layer_idx=0, target="k", merge_pct=100, seq_start=1) is x
    odd = torch.cat([x, x[..., :1, :]], dim=-2)
    out = pair_rank.close_pairs(odd, merge_pct=100, keep_pct=0)
    assert torch.equal(out[..., -1:, :], odd[..., -1:, :])


def test_bytes_and_arguments():
    assert pair_rank.merged_pair_cost(128, 0) == 0.5
    assert abs(pair_rank.merged_pair_cost(128, 25) - (17 * 128 + 16 * 32) / (32 * 128)) < 1e-12
    stats = {"pairs": 10, "merged": 5, "features": 128, "kept": 32}
    assert abs(pair_rank.bytes_vs_dense(stats) - (5 * 0.65625 + 5) / 10) < 1e-12
    x = _pairs(batch=1)
    with pytest.raises(ValueError):
        pair_rank.close_pairs(x, merge_pct=101, keep_pct=0)
    with pytest.raises(ValueError):
        pair_rank.apply(x, layer_idx=0, target="q")


def test_rungs_and_levels():
    from catalog.compressions import pair_rank as spec, pair_rank_residual as spec_residual
    from engine.layer_select.apply import kv_for_slot, parse_singleton
    from engine.layer_select.rungs import fraction_of, rungs_for
    from engine.layer_select.slots import Slot

    assert [r.level for r in rungs_for("pair_rank")] == [25, 50, 75, 100]
    assert fraction_of(rungs_for("pair_rank"), 100) == 0.5
    assert abs(fraction_of(rungs_for("pair_rank_residual"), 100) - 0.34375) < 1e-12
    kv = kv_for_slot(spec_residual(), Slot(4, "k"), level=75)
    assert kv["pipeline"][0]["merge_pct"] == 75 and kv["pipeline"][0]["keep_pct"] == 25
    assert parse_singleton(kv) == (Slot(4, "k"), 75)
    assert parse_singleton(kv_for_slot(spec(), Slot(2, "k"), level=25)) == (Slot(2, "k"), 25)


def test_keys_only_greedy_measures_the_key_cache():
    from engine.layer_select.budgets import selections_for_budgets
    from engine.layer_select.scores import ScoreRow
    from engine.layer_select.slots import Slot
    from catalog.compressions import pair_rank as spec

    rows = [
        ScoreRow(Slot(layer, "k"), level, 10 + layer * 0.01 + level / 1000, 0.0, "")
        for layer in range(4)
        for level in (25, 50, 75, 100)
    ]
    payloads = selections_for_budgets(rows, 4, spec(), 10.0, budgets=(0.2, 0.5), targets=("k",))
    assert all(item["target"] == "k" for p in payloads for item in p["assignment"])
    assert payloads[0]["compression"] >= 0.2 and payloads[1]["compression"] == 0.5  # 0.5 is every key layer at 100%


def test_pair_rank_runs_inside_the_cache_hook():
    from catalog.compressions import pair_rank as spec
    from engine.kv_compress import parse_kv_spec
    from engine.kv_compress.cache import patch_cache_update
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 64, (1, 16))
    pair_rank.reset_stats()
    uninstall = patch_cache_update(parse_kv_spec(spec(k_layers="all", v_layers=[], merge_pct=50)), None)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    finally:
        uninstall()
    stats = pair_rank.pop_stats()
    assert set(stats) == {"k"}
    # 2 layers x 2 KV heads x 8 pairs, half of them merged.
    assert stats["k"]["pairs"] == 2 * 2 * 8 and stats["k"]["merged"] == 2 * 2 * 8 // 2
