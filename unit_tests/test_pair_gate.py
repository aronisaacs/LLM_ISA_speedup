import pytest
import torch

from compression_topics.spatial.algorithms import pair_gate


def _pairs(similar: bool, features: int = 16):
    torch.manual_seed(0)
    a = torch.randn(2, 3, 4, features) + 3.0
    b = a + (0.02 if similar else 3.0) * torch.randn_like(a)
    x = torch.empty(2, 3, 8, features)
    x[..., 0::2, :] = a
    x[..., 1::2, :] = b
    return x


def test_similar_pairs_merge_to_the_mean_without_a_residual():
    pair_gate.reset_stats()
    x = _pairs(True)
    out = pair_gate.close_pairs(x, tau=0.5, keep_pct=0, target="k")
    mean = (x[..., 0::2, :] + x[..., 1::2, :]) / 2
    assert torch.allclose(out[..., 0::2, :], mean) and torch.allclose(out[..., 1::2, :], mean)
    stats = pair_gate.pop_stats()["k"]
    assert stats["pairs"] == stats["merged"] == 2 * 3 * 4
    assert pair_gate.bytes_vs_dense(stats) == 0.5


def test_dissimilar_pairs_stay_exact():
    x = _pairs(False)
    out = pair_gate.close_pairs(x, tau=0.1, keep_pct=0)
    assert torch.equal(out, x)


def test_gate_decides_per_pair():
    x = _pairs(True)
    x[0, 0, 1, :] += torch.randn(16) * 5  # one pair becomes dissimilar
    pair_gate.reset_stats()
    out = pair_gate.close_pairs(x, tau=0.3, keep_pct=0, target="v")
    assert torch.equal(out[0, 0, 0:2], x[0, 0, 0:2])  # positions 0 and 1 form the changed pair
    assert not torch.equal(out[0, 0, 2:4], x[0, 0, 2:4])
    stats = pair_gate.pop_stats()["v"]
    assert stats["merged"] == stats["pairs"] - 1


def test_a_residual_rescues_an_outlier_pair():
    torch.manual_seed(1)
    a = torch.randn(1, 1, 1, 16) + 3.0
    b = a.clone()
    b[..., 5] += 6.0  # one big difference, all else identical
    x = torch.cat([a, b], dim=-2)
    assert torch.equal(pair_gate.close_pairs(x, tau=0.1, keep_pct=0), x)  # no residual: not similar enough
    out = pair_gate.close_pairs(x, tau=0.01, keep_pct=7)  # 7% of 16 features = 1 feature kept
    assert torch.allclose(out, x, atol=1e-5)  # merged, and the kept difference restores it


def test_tail_token_and_odd_start_are_left_alone():
    x = _pairs(True)
    odd = torch.cat([x, x[..., :1, :]], dim=-2)
    out = pair_gate.close_pairs(odd, tau=1.0, keep_pct=0)
    assert torch.equal(out[..., -1:, :], odd[..., -1:, :])
    assert pair_gate.apply(x, layer_idx=0, target="k", tau=1.0, seq_start=1) is x


def test_bytes_with_a_residual():
    stats = {"pairs": 10, "merged": 10, "features": 128, "kept": 16}
    assert abs(pair_gate.bytes_vs_dense(stats) - (17 * 128 + 16 * 16) / (32 * 128)) < 1e-12
    stats["merged"] = 5
    assert abs(pair_gate.bytes_vs_dense(stats) - (5 * (17 * 128 + 256) / (32 * 128) + 5) / 10) < 1e-12


def test_arguments_are_checked():
    x = _pairs(True)
    with pytest.raises(ValueError):
        pair_gate.close_pairs(x, tau=-1, keep_pct=0)
    with pytest.raises(ValueError):
        pair_gate.close_pairs(x, tau=0.3, keep_pct=101)
    with pytest.raises(ValueError):
        pair_gate.apply(x, layer_idx=0, target="q", tau=0.3)


def test_pair_gate_runs_inside_the_cache_hook_and_counts_pairs():
    from catalog.compressions import pair_gate as pair_gate_spec
    from engine.kv_compress import parse_kv_spec
    from engine.kv_compress.cache import patch_cache_update
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 64, (1, 9))
    spec = parse_kv_spec(pair_gate_spec(tau=100.0, keep_pct=0))  # a huge tau merges every pair
    pair_gate.reset_stats()
    uninstall = patch_cache_update(spec, None)
    try:
        model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, min_new_tokens=6, do_sample=False)
    finally:
        uninstall()
    stats = pair_gate.pop_stats()
    for target in ("k", "v"):
        assert stats[target]["pairs"] > 0 and stats[target]["merged"] == stats[target]["pairs"]
    # 2 layers, 2 KV heads. The cache ends with 14 tokens (9 prompt plus 5 fed back), so 7 pairs, each counted once.
    assert stats["k"]["pairs"] == 2 * 2 * 7
