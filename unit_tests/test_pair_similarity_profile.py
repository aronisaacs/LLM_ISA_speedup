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
        model, prompts, max_new_tokens=10, rope=psp.rope_tables(model), ks=(0, 2),
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
    assert psp.markdown_tables(summary, ks=(0, 2))
