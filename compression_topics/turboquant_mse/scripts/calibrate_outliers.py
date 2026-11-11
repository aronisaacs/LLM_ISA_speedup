#!/usr/bin/env python3
"""Rank each layer's, KV head's and target's channels by mean |value| for the TurboQuant outlier split.

Runs WikiText-2 train chunks through the dense model and reads K and V where the
cache stores them (``Cache.update``, post-RoPE, the same place compression runs).
Writes the full ranking per head, so any number of outlier channels can be taken
from the front. Uses one GPU; nothing is compressed and nothing goes to results.json.

  python compression_topics/turboquant_mse/scripts/calibrate_outliers.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.models import LLAMA31_8B  # noqa: E402

TABLE = "compression_topics/turboquant_mse/calibration/llama31_8b_outliers.json"


class ChannelStats:
    """Sum of |value| per layer, target, KV head and channel, and the token count per layer and target."""

    def __init__(self):
        self.sums: dict[tuple[int, str], torch.Tensor] = {}
        self.tokens: dict[tuple[int, str], int] = {}

    def add(self, layer: int, target: str, tensor: torch.Tensor) -> None:
        """``tensor`` is ``[batch, kv_heads, seq, head_dim]``."""
        key = (int(layer), target)
        total = tensor.detach().abs().double().sum(dim=(0, 2)).cpu()
        self.sums[key] = self.sums.get(key, torch.zeros_like(total)) + total
        self.tokens[key] = self.tokens.get(key, 0) + tensor.shape[0] * tensor.shape[2]

    def table(self, metadata: dict) -> dict:
        layers = sorted({layer for layer, _ in self.sums})
        if layers != list(range(len(layers))):
            raise ValueError(f"missing layers in calibration: {layers}")
        channels, mean_abs = {}, {}
        for target in ("k", "v"):
            means = [self.sums[(layer, target)] / self.tokens[(layer, target)] for layer in layers]
            channels[target] = [torch.argsort(mean, dim=-1, descending=True).tolist() for mean in means]
            mean_abs[target] = [[[round(float(v), 5) for v in head] for head in mean] for mean in means]
        head_dim = len(channels["k"][0][0])
        return {**metadata, "ranking_length": head_dim, "channels": channels, "mean_abs": mean_abs}


def calibrate(chunks: int, seq_len: int, seed: int, split: str) -> dict:
    from transformers.cache_utils import Cache

    from engine.eval_runner.execute import load_model_if_needed
    from engine.eval_runner.text_chunks import wikitext_chunks

    lm, _, device, _ = load_model_if_needed(None, None, {"model": "hf", "model_args": LLAMA31_8B}, {})
    model = lm.model.eval()
    prompts, chosen = wikitext_chunks(lm.tokenizer, chunks, seq_len, seed, split=split)
    stats = ChannelStats()
    original = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        stats.add(layer_idx, "k", key_states)
        stats.add(layer_idx, "v", value_states)
        return original(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update = update
    try:
        with torch.inference_mode():
            for number, ids in enumerate(prompts, 1):
                batch = ids.unsqueeze(0).to(model.device)
                model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=True)
                print(f"chunk {number}/{len(prompts)}", flush=True)
    finally:
        Cache.update = original
    pretrained = LLAMA31_8B.split(",")[0].split("=", 1)[1]
    return stats.table({
        "model": pretrained,
        "statistic": "mean |value| per channel over tokens, post-RoPE, ranked high to low per layer, KV head and target",
        "data": {"dataset": "EleutherAI/wikitext_document_level wikitext-2-raw-v1", "split": split,
                 "chunks": chunks, "seq_len": seq_len, "seed": seed, "chunk_indices": chosen},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=TABLE)
    args = parser.parse_args()
    table = calibrate(args.chunks, args.seq_len, args.seed, args.split)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
