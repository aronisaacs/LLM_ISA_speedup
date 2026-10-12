#!/usr/bin/env python3
"""Does query-aware ranking help vector compression of keys? A laptop check.

Plain ranking zeros the smallest k_d^2 (per-scalar) or a^2 + b^2 (RoPE pair
i, i + 64). Query-aware ranking weights each score by the calibrated query
energy w = E[q_d^2] of that layer and KV head: w_d k_d^2, or
(w_i + w_{i+64})(a^2 + b^2) for a pair. The weights are the ones the spatial
study calibrated on WikiText train (post-RoPE queries, summed over the four
query heads of a KV head).

Every layer's keys are sparsified at one uniform level; values stay dense, so
only the key ranking differs between arms. Llama 3.1 8B is streamed one
decoder layer at a time from the cached safetensors (as in
llamacpp_zero_groups_local.py), and all arms advance through each layer
together, so the weights are read once. Compression is applied in the cache
update, so attention already sees the compressed keys, as in the eval runs.

Reported per arm:
  * WikiText test perplexity over the scored tokens;
  * per-layer attention-logit error on the dense arm's inputs,
    E[(q . (k_hat - k))^2] / E[(q . k)^2], over the last --probe queries.

  python compression_topics/vector/scripts/query_aware_ppl_local.py --chunks 1 --levels 50   # pilot
  python compression_topics/vector/scripts/query_aware_ppl_local.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.vector.algorithms.vector_compress import _apply as scalar_sparsify  # noqa: E402
from compression_topics.vector.algorithms.vector_compress_pair import _prune_rope_pairs as pair_sparsify  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
WEIGHTS = ROOT / "compression_topics/spatial/figures/clean_online_ladder_query_hierarchy/query_weights.pt"
OUT = ROOT / "compression_topics/vector/figures/query_aware_local.json"
METHODS = ("scalar", "scalar_query", "pair", "pair_query")


def scalar_query(keys: torch.Tensor, weights: torch.Tensor, prune_pct: int) -> torch.Tensor:
    """Zero the prune_pct of values with the smallest w_d k_d^2. ``weights`` broadcasts over keys."""
    n_drop = keys.shape[-1] * prune_pct // 100
    if n_drop == 0:
        return keys
    score = keys.float().square() * weights
    drop = score.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    return keys.scatter(-1, drop, 0)


def pair_query(keys: torch.Tensor, weights: torch.Tensor, prune_pct: int) -> torch.Tensor:
    """Zero the prune_pct of pairs (i, i + d/2) with the smallest (w_i + w_{i+d/2})(a^2 + b^2)."""
    half = keys.shape[-1] // 2
    n_drop = half * prune_pct // 100
    if n_drop == 0:
        return keys
    first, second = keys[..., :half], keys[..., half:]
    plane = weights[..., :half] + weights[..., half:]
    score = (first.float().square() + second.float().square()) * plane
    drop = score.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(score, dtype=torch.bool).scatter(-1, drop, False)
    return torch.cat((first * keep, second * keep), dim=-1)


def compress(method: str, keys: torch.Tensor, weights: torch.Tensor, prune_pct: int) -> torch.Tensor:
    if method == "scalar":
        return scalar_sparsify(keys, threshold=0.0, prune_pct=prune_pct)
    if method == "pair":
        return pair_sparsify(keys, prune_pct)
    if method == "scalar_query":
        return scalar_query(keys, weights, prune_pct)
    return pair_query(keys, weights, prune_pct)


class KeyCompressor:
    """Stands in for the KV cache of one arm: compresses keys, keeps nothing."""

    def __init__(self, method: str | None, level: int, weights: torch.Tensor) -> None:
        self.method, self.level, self.weights = method, level, weights

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.method is None:
            return key_states, value_states
        w = self.weights[layer_idx].to(key_states.device).reshape(1, key_states.shape[1], 1, -1)
        return compress(self.method, key_states, w, self.level), value_states


def logit_error(queries: torch.Tensor, keys: torch.Tensor, restored: torch.Tensor, probe: int) -> float:
    """E[(q . (k_hat - k))^2] / E[(q . k)^2] over causal pairs of the last ``probe`` queries."""
    batch, q_heads, length, dim = queries.shape
    group = q_heads // keys.shape[1]
    q = queries[:, :, -probe:].float().reshape(batch, keys.shape[1], group, probe, dim)
    k = keys.float().unsqueeze(2)
    delta = (restored.float() - keys.float()).unsqueeze(2)
    positions = torch.arange(length - probe, length, device=queries.device)
    causal = (torch.arange(length, device=queries.device)[None, :] <= positions[:, None]).float()
    signal = ((q @ k.transpose(-1, -2)).square() * causal).sum()
    error = ((q @ delta.transpose(-1, -2)).square() * causal).sum()
    return float(error / signal)


def run(model_dir: Path, input_ids: torch.Tensor, arms: list[tuple[str, str | None, int]], weights: torch.Tensor,
        device: str, probe: int) -> dict:
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

    from engine.kv_compress.importance import capture_queries

    config = AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = "sdpa"
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(model_dir / index[name], framework="pt") as f:
            return f.get_tensor(name)

    dtype = torch.float32 if device == "cpu" else torch.float16
    embed = tensor("model.embed_tokens.weight")
    start_hidden = embed[input_ids].to(device, dtype)
    del embed
    hidden = {name: start_hidden.clone() for name, _, _ in arms}
    del start_hidden
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
    cos_sin = LlamaRotaryEmbedding(config).to(device)(hidden[arms[0][0]], positions)
    caches = {name: KeyCompressor(method, level, weights) for name, method, level in arms}
    probes = sorted({(method, level) for _, method, level in arms if method is not None})
    errors = {f"{method}_{level}": [] for method, level in probes}
    take, uninstall = capture_queries("llama")
    try:
        for i in range(config.num_hidden_layers):
            started = time.monotonic()
            layer = LlamaDecoderLayer(config, i)
            prefix = f"model.layers.{i}."
            layer.load_state_dict({k[len(prefix):]: tensor(k) for k in index if k.startswith(prefix)})
            layer = layer.to(device, dtype).eval()
            with torch.no_grad():
                for name, method, _ in arms:
                    if method is None:
                        seen = {}

                        class Recorder(KeyCompressor):
                            def update(self, key_states, value_states, layer_idx, *args, **kwargs):
                                seen["keys"], seen["queries"] = key_states, take()
                                return key_states, value_states

                        hidden[name] = layer(hidden[name], position_ids=positions, position_embeddings=cos_sin,
                                             past_key_values=Recorder(None, 0, weights))
                        w = weights[i].to(device).reshape(1, seen["keys"].shape[1], 1, -1)
                        for p_method, p_level in probes:
                            restored = compress(p_method, seen["keys"], w, p_level)
                            errors[f"{p_method}_{p_level}"].append(
                                logit_error(seen["queries"], seen["keys"], restored, probe))
                        del seen
                    else:
                        hidden[name] = layer(hidden[name], position_ids=positions, position_embeddings=cos_sin,
                                             past_key_values=caches[name])
                        take()
            del layer
            if device == "mps":
                torch.mps.empty_cache()
            line = "  ".join(f"{key} {values[-1]:.4f}" for key, values in errors.items())
            print(f"layer {i:2d}  {time.monotonic() - started:5.1f}s  logit err  {line}", flush=True)
    finally:
        uninstall()
    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight.data = tensor("model.norm.weight")
    norm = norm.to(device, dtype)
    head = tensor("lm_head.weight").to(device, dtype)
    targets = input_ids[:, 1:].to(device)
    ppl = {}
    with torch.no_grad():
        for name, _, _ in arms:
            states = norm(hidden[name])[:, :-1]
            nll, count = 0.0, 0
            for start in range(0, states.shape[1], 256):
                logits = (states[:, start:start + 256] @ head.T).float()
                nll += float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets[:, start:start + 256].reshape(-1), reduction="sum"))
                count += targets[:, start:start + 256].numel()
            ppl[name] = math.exp(nll / count)
    return {"ppl": ppl, "logit_error": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=4, help="WikiText test chunks")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--levels", type=int, nargs="+", default=[50, 75])
    parser.add_argument("--probe", type=int, default=256, help="queries per chunk for the logit error")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    from engine.eval_runner.text_chunks import wikitext_chunks

    cached = try_to_load_from_cache(MODEL, "model.safetensors.index.json")
    if not isinstance(cached, str):
        raise SystemExit(f"{MODEL} is not in the local Hugging Face cache")
    model_dir = Path(cached).parent
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    chunks, chosen = wikitext_chunks(tokenizer, args.chunks, args.seq_len, args.seed, split="test")
    input_ids = torch.stack(chunks)
    saved = torch.load(WEIGHTS, weights_only=True)
    weights = saved["weights"].float()
    arms = [("dense", None, 0)] + [(f"{method}_{level}", method, level) for level in args.levels for method in METHODS]
    print(f"{input_ids.numel()} tokens as {input_ids.shape[0]} x {input_ids.shape[1]}  device {args.device}  "
          f"arms {', '.join(name for name, _, _ in arms)}", flush=True)
    result = run(model_dir, input_ids, arms, weights, args.device, args.probe)
    print(f"\n{'arm':<18} {'ppl':>8} {'dppl':>8} {'mean logit err':>15}")
    dense = result["ppl"]["dense"]
    for name, _, _ in arms:
        errors = result["logit_error"].get(name)
        mean = f"{sum(errors) / len(errors):15.4f}" if errors else f"{'':>15}"
        print(f"{name:<18} {result['ppl'][name]:8.3f} {result['ppl'][name] - dense:+8.3f} {mean}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model": MODEL, "split": "wikitext-2 test", "chunks": chosen, "seq_len": args.seq_len,
        "levels": args.levels, "probe_queries": args.probe, "query_weights": str(WEIGHTS.relative_to(ROOT)),
        "query_weight_chunks": saved.get("chunks"), "keys_only": True, **result}, indent=2) + "\n")
    print(f"wrote  {OUT}")


if __name__ == "__main__":
    main()
