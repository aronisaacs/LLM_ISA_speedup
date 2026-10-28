#!/usr/bin/env python3
"""Cosine similarity between each key and value and its rebuilt version.

Read-only profiling, same text as pair_similarity_profile.py: Llama 3.1
8B-Instruct, WikiText-2 test, --samples random chunks of --seq-len tokens (seed 0),
one forward pass per chunk. A hook on Cache.update takes every layer's keys and
values as stored in the cache and, per KV head, merges every aligned pair
(positions 2i, 2i+1) with the official method's reconstruction (representation.py):
unit directions, mean plus 0, 16 or 32 residual entries, rebuilt with the per-token
scale. Keys rank residual entries with the official calibrated query weights;
values by |d|. No budget is applied: every pair is measured in every format.

Reported, per K/V and format: the cosine between the original vector and its
rebuilt version (median over layers of per-layer medians, and the pooled share at
or above each threshold), plus the cosine between the two tokens of each pair as a
check against slide 2's numbers.

  python compression_topics/spatial/scripts/reconstruction_cosine.py --samples 50

On a Mac whose memory cannot hold the model, --offload puts as many layers as fit
on the GPU (MPS), the next ones in CPU memory and the rest on disk:

  python compression_topics/spatial/scripts/reconstruction_cosine.py --samples 10 --offload
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.spatial.algorithms.formats import RESIDUALS  # noqa: E402
from compression_topics.spatial.algorithms.representation import (  # noqa: E402
    prepare,
    query_energy,
    reconstruct,
)
from engine.eval_runner.files import write_json  # noqa: E402
from engine.eval_runner.text_chunks import wikitext_chunks  # noqa: E402

PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
WEIGHTS = ROOT / "compression_topics/spatial/figures/official_llama/query_weights.pt"
OUT = ROOT / "compression_topics/spatial/figures/pair_similarity/reconstruction_cosine.json"
THRESHOLDS = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
BINS = 4000  # cosine in [-1, 1]
FORMATS = ("pair",) + tuple(f"PR{r}" for r in RESIDUALS)  # "pair": cos(a, b)


class Counts:
    """Cosine histograms per (kind, format, layer)."""

    def __init__(self):
        self.hist = {}

    def add(self, kind, name, layer, cos):
        key = (kind, name, layer)
        if key not in self.hist:
            self.hist[key] = torch.zeros(BINS, dtype=torch.float64)
        self.hist[key] += torch.histc(cos.float().clamp(-1, 1).cpu(), bins=BINS, min=-1.0, max=1.0).double()

    def summary(self):
        edges = torch.linspace(-1, 1, BINS + 1, dtype=torch.float64)

        def median(h):
            c = h.cumsum(0)
            return float(edges[int(torch.searchsorted(c, c[-1] / 2))] + 1 / BINS)

        def share(h, t):
            return float(h[int(round((t + 1) / 2 * BINS)):].sum() / h.sum())

        out = {}
        for kind in ("k", "v"):
            for name in FORMATS:
                layers = sorted(l for (k, n, l) in self.hist if k == kind and n == name)
                if not layers:
                    continue
                per_layer = [median(self.hist[(kind, name, l)]) for l in layers]
                pooled = sum(self.hist[(kind, name, l)] for l in layers)
                out.setdefault(kind, {})[name] = {
                    "n": int(pooled.sum()),
                    "median_of_layer_medians": float(torch.tensor(per_layer).median()),
                    "pooled_median": median(pooled),
                    "share_at_least": {str(t): share(pooled, t) for t in THRESHOLDS},
                    "layer_medians": dict(zip(map(str, layers), per_layer)),
                }
        return out


def measure(counts, layer, kind, states, weights, digest):
    """Every aligned pair of a [1, heads, tokens, dim] slice, in every format."""
    states = states[..., : states.shape[-2] // 2 * 2, :].detach()
    if not states.shape[-2]:
        return
    pairs = prepare(states)
    unit = torch.nn.functional.normalize(pairs.source, dim=-1)
    counts.add(kind, "pair", layer, (unit[..., 0, :] * unit[..., 1, :]).sum(-1))
    energy = query_energy(weights, digest, layer, states) if kind == "k" and weights else None
    for residual in RESIDUALS:
        restored, _ = reconstruct(pairs, residual, dtype=torch.bfloat16, query_energy=energy)
        cos = (unit * torch.nn.functional.normalize(restored, dim=-1)).sum(-1)
        counts.add(kind, f"PR{residual}", layer, cos)


def print_table(summary, done, total):
    print(f"--- {done}/{total} chunks: median over layers, then share with cosine >= 0.9 ---", flush=True)
    for kind, rows in summary.items():
        for name, row in rows.items():
            print(f"{kind} {name:>5}: median {row['median_of_layer_medians']:.3f}  "
                  + "  ".join(f">={t}: {100 * v:5.1f}%" for t, v in row["share_at_least"].items()),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=PRETRAINED)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-weights", type=Path, default=WEIGHTS,
                        help="calibrated E[q^2] for keys; 'none' ranks keys by |d|")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--offload", action="store_true",
                        help="split the model over the Mac GPU, CPU memory and disk")
    parser.add_argument("--gpu-memory", default="8GiB", help="with --offload: weights on the GPU")
    parser.add_argument("--cpu-memory", default="3GiB", help="with --offload: weights in CPU memory")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache

    weights = None if str(args.query_weights) == "none" else str(args.query_weights)
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights else None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.offload:
        import tempfile
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            max_memory={device: args.gpu_memory, "cpu": args.cpu_memory} if device == "mps"
            else {"cpu": args.cpu_memory},
            offload_folder=tempfile.mkdtemp(prefix="offload_")).eval()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    prompts, chosen = wikitext_chunks(tokenizer, args.samples, args.seq_len, args.seed)
    counts = Counts()
    original = Cache.update

    def update(cache, keys, values, layer, *rest, **kwargs):
        measure(counts, layer, "k", keys, weights, digest)
        measure(counts, layer, "v", values, None, None)
        return original(cache, keys, values, layer, *rest, **kwargs)

    Cache.update = update
    try:
        with torch.no_grad():
            for n, prompt in enumerate(prompts, 1):
                model(input_ids=prompt.unsqueeze(0).to(device), use_cache=True)
                if n % 10 == 0 or n == len(prompts) or args.offload:
                    print_table(counts.summary(), n, len(prompts))
    finally:
        Cache.update = original
    write_json(args.out, {
        "meta": {"model": args.model, "dataset": "wikitext", "split": "test", "samples": args.samples,
                 "seq_len": args.seq_len, "seed": args.seed, "chosen": chosen,
                 "query_weights": weights, "query_weights_sha256": digest,
                 "formats": "pair = cos(a, b); PRn = cos(a, rebuilt a) with n residual entries per 128"},
        "summary": counts.summary(),
    })
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
