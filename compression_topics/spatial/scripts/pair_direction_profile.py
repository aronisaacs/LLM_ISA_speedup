#!/usr/bin/env python3
"""Keys-only cosine profile: one-step RoPE alignment and separate token norms.

Fresh WikiText forward passes, using the same chunks as the old 50-sample run.
Stores cosine distributions only. No accuracy claims follow from this profile.
Residuals are selected in the first token's RoPE frame, not the unrotated frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms.pair_rank import _shift_rope
from compression_topics.spatial.algorithms.pair_gate import _largest
from compression_topics.spatial.scripts.pair_similarity_profile import (
    PRETRAINED, rope_tables, run_profile, wikitext_chunks,
)


class DirectionProfiler:
    def __init__(self, rope):
        if rope is None or rope.attention_scaling != 1.0:
            raise ValueError("profile requires pure-rotation RoPE tables")
        self.rope = rope
        self.question = 0
        self.rows = {}

    def observe(self, layer_idx, keys, values, start):
        if start != 0:
            raise ValueError("this experiment profiles prefill only")
        full = keys.shape[-2] // 2 * 2
        if full == 0:
            return
        first = keys[..., :full:2, :].float()
        second = _shift_rope(keys[..., 1:full:2, :].float(), self.rope, inverse=True)
        valid = (first.norm(dim=-1) > 1e-12) & (second.norm(dim=-1) > 1e-12)
        a = torch.nn.functional.normalize(first, dim=-1)
        b = torch.nn.functional.normalize(second, dim=-1)
        mean, delta = (a + b) / 2, (a - b) / 2
        for kept in (0, 4, 8, 16, 32):
            if kept > keys.shape[-1]:
                continue
            residual = _largest(delta, kept)
            rest = delta - residual
            adjusted = torch.nn.functional.cosine_similarity(mean + rest, mean - rest, dim=-1)
            restored_first = torch.nn.functional.normalize(mean + residual, dim=-1) * first.norm(dim=-1, keepdim=True)
            restored_second = torch.nn.functional.normalize(mean - residual, dim=-1) * second.norm(dim=-1, keepdim=True)
            restored_second = _shift_rope(restored_second, self.rope, inverse=False)
            reconstructed = keys[..., :full, :].clone()
            can_merge = valid & ((mean + residual).norm(dim=-1) > 1e-6) & ((mean - residual).norm(dim=-1) > 1e-6)
            reconstructed[..., 0::2, :] = torch.where(can_merge.unsqueeze(-1), restored_first.to(keys.dtype), reconstructed[..., 0::2, :])
            reconstructed[..., 1::2, :] = torch.where(can_merge.unsqueeze(-1), restored_second.to(keys.dtype), reconstructed[..., 1::2, :])
            fidelity = torch.nn.functional.cosine_similarity(keys[..., :full, :].float(), reconstructed.float(), dim=-1)
            for head in range(keys.shape[1]):
                mask = valid[:, head].reshape(-1)
                row = self.rows.setdefault((layer_idx, head, kept), {"pairs": 0, "zero_norm_pairs": 0,
                    "adjusted_cos_hist": torch.zeros(400, dtype=torch.int64),
                    "reconstruction_cos_hist": torch.zeros(400, dtype=torch.int64),
                    "adjusted_cos_sum": 0., "reconstruction_cos_sum": 0., "threshold_counts": {}})
                cos = adjusted[:, head].reshape(-1)[mask].clamp(-1, 1)
                fid = fidelity[:, head, :full].reshape(-1)[mask.repeat_interleave(2)].clamp(-1, 1)
                row["pairs"] += cos.numel()
                row["zero_norm_pairs"] += int((~mask).sum())
                row["adjusted_cos_hist"] += torch.histc(cos.cpu(), bins=400, min=-1, max=1).long()
                row["reconstruction_cos_hist"] += torch.histc(fid.cpu(), bins=400, min=-1, max=1).long()
                row["adjusted_cos_sum"] += float(cos.sum())
                row["reconstruction_cos_sum"] += float(fid.sum())
                for threshold in (.7, .8, .9, .95, .99):
                    name = str(threshold)
                    row["threshold_counts"][name] = row["threshold_counts"].get(name, 0) + int((cos >= threshold).sum())

    def finish_sequence(self):
        print(f"profiled chunk {self.question + 1}", flush=True)

    def payload(self):
        return [{"layer": layer, "head": head, "residual_entries": kept,
                 **{key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in row.items()}}
                for (layer, head, kept), row in sorted(self.rows.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=PRETRAINED)
    parser.add_argument("--out", type=Path, default=ROOT / "compression_topics/spatial/figures/pair_direction")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    prompts, chosen = wikitext_chunks(tokenizer, args.samples, args.seq_len, args.seed)
    profiler = DirectionProfiler(rope_tables(model))
    run_profile(model, prompts, max_new_tokens=0, rope=profiler.rope, profiler=profiler)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "samples": args.samples, "seq_len": args.seq_len,
            "seed": args.seed, "chosen": chosen, "norm_overhead_counted": False,
            "alignment": "second key rotated backward one position into first key frame",
            "representation": "unit-direction mean and sparse half-difference; normalize reconstruction then restore each original norm",
            "ranking": "cosine between mean +/- the unstored directional half-difference",
            "reconstruction_cos": "cosine of each original key against its actual reconstruction",
            "histogram": {"min": -1, "max": 1, "bins": 400},
            "rope": {"inv_freq": profiler.rope.inv_freq, "head_dim": profiler.rope.head_dim,
                     "rope_theta": profiler.rope.rope_theta, "attention_scaling": profiler.rope.attention_scaling}}
    (args.out / "profile.json").write_text(json.dumps({"meta": meta, "rows": profiler.payload()}, indent=1) + "\n")
    print(f"wrote {args.out / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
