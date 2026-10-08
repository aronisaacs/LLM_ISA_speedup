#!/usr/bin/env python3
"""How similar are adjacent token pairs, per layer, in prefill and in decode?

Read-only profiling. No compression is applied. A hook on Cache.update sees the
exact K/V a normal run stores, and every aligned pair (positions 2i, 2i+1) is
compared per KV head. Pairs from the prompt are the prefill phase. Pairs made
only of generated tokens are the decode phase.

The text is WikiText-2 by default (--dataset wikitext): the test split is cut into
--seq-len token chunks and --samples of them are drawn at random. That is one forward
pass per chunk, so prefill only and fast. --dataset gsm8k runs 5-shot GSM8K questions
with greedy generation instead, which gives prefill and decode.

For each pair, with m = (a + b) / 2 and d = (a - b) / 2:
  cos    cosine similarity of a and b
  rel    ||d||_2 / ||m||_2, the L2 size of the difference against the shared part
         (merging the pair to m moves each token by exactly ||d||_2)
and the same two numbers again after the largest-|d| share of the features is set aside
(a and b both take the value m there, as if those entries were kept in an exact residual).
Residuals are given as a share of the head dimension: "1/8" keeps the top 1/8 of the
features, 16 of 128 on the 8B. "0" is the plain pair.

Keys are measured twice: as stored (RoPE applied) and with RoPE undone.

The full-curve dump (profile_kneeded.json) keeps, for every threshold on a fine grid, how many
pairs need each residual size k = 0..head_dim features (a share k / head_dim) to get within it. Any threshold and any set of
allowed residual sizes can be evaluated from it afterwards (see ``bytes_from_kneeded``).

The movement dump (profile_moves.json) groups pairs into buckets of their rel before any residual
(k = 0) and, per bucket and per k, keeps the pair count and the sums of rel before, rel after and
rel after squared. The average movement of a bucket is mean(before) - mean(after).

The sample dump (profile_samples.npz) keeps individual pairs: a uniform random sample per
(kind, phase, layer) with the pair's whole rel curve rel_k for k = 0..head_dim (half precision),
and its question index, KV head, token position, ||a||, ||b||, ||m||, ||d|| and cosine.

The norm dump (profile_norms.npz, float16) keeps the L2 norm of every stored key and value
vector: per question, keys and values [layers, kv_heads, positions] and the prompt length. It is
written next to the raw dump when --dump-kv is given (it can pass GitHub's 100 MB file limit),
else into --out. RoPE is a rotation, so a key's norm is the same with RoPE applied or undone.
profile_norm_stats.json, always in --out, keeps per (kind, phase, layer) and per KV head the
count, mean, std and quantiles of those norms.

With --dump-kv DIR the exact K, V and (for Llama) query states are saved as one safetensors
file per question or chunk, so other statistics can be computed later with no model: keys and
values [layers, kv_heads, positions, head_dim], queries [layers, heads, positions, head_dim],
the token ids, and "positions", the token positions kept. To keep the dump small only
--dump-windows random windows of --dump-window consecutive tokens are kept per sequence
(default 8 x 16, so 128 of 2048 WikiText tokens); the windows start on multiples of the window
size, so aligned pairs, triples and quads stay whole, and the same positions are kept in every
layer and head. --dump-windows 0 keeps whole sequences, --no-dump-queries drops the queries.
The statistics above always use every token; only the raw dump is sampled. Prompt length is in
the file metadata, and meta.json holds the model, seeds, chosen samples and RoPE settings.
Keys and queries are as stored (RoPE applied).

Per pair, the smallest k whose rel is within a threshold is the residual that pair would
need. The tier tables show how many pairs need each k, how many cannot be merged, and the
bytes left against a dense cache.

  python compression_topics/spatial/scripts/pair_similarity_profile.py --samples 30
  python compression_topics/spatial/scripts/pair_similarity_profile.py --dataset gsm8k --samples 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.eval_runner.datasets import wikitext_chunks
from engine.kv_compress.rope import RopeTables, apply_rope  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_similarity"
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
FRACTIONS = ("0", "1/32", "1/16", "1/8", "1/4")  # residual sizes as a share of head_dim
COS_THRESHOLDS = (0.5, 0.75, 0.9, 0.95, 0.99)
REL_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
REL_BINS = 400  # rel in [0, 4]
COS_BINS = 400  # cos in [-1, 1]
REL_MAX = 4.0
TIER_THRESHOLDS = (0.1, 0.2, 0.3)  # rel a merged pair must reach, with the smallest residual that does
TAU_GRID = tuple(round(0.02 * i, 2) for i in range(1, 51))  # 0.02 .. 1.0, for the full-curve dump
BUCKET_EDGES = tuple(round(0.05 * i, 2) for i in range(1, 21))  # buckets of rel before residuals; the last is >= 1.0
KINDS = ("k_rope", "k_plain", "v")


def ks_for(fractions: tuple[str, ...], head_dim: int) -> tuple[int, ...]:
    """Features kept for each residual share, e.g. "1/8" of 128 is 16."""
    return tuple(int(round(Fraction(f) * head_dim)) for f in fractions)


def label(fraction: str) -> str:
    return "none" if Fraction(fraction) == 0 else f"top {fraction}"
PHASES = ("prefill", "decode")


class Histograms:
    """Counts of cos and rel over many pairs, for one (layer, kind, phase, k)."""

    def __init__(self) -> None:
        self.n = 0
        self.cos = torch.zeros(COS_BINS, dtype=torch.float64)
        self.rel = torch.zeros(REL_BINS, dtype=torch.float64)

    def add(self, cos: torch.Tensor, rel: torch.Tensor) -> None:
        self.n += int(cos.numel())
        self.cos += torch.histc(cos.float().cpu(), bins=COS_BINS, min=-1.0, max=1.0).double()
        self.rel += torch.histc(rel.float().clamp(max=REL_MAX).cpu(), bins=REL_BINS, min=0.0, max=REL_MAX).double()

    def raw(self) -> dict:
        """Full counts: rel over [0, REL_MAX] and cos over [-1, 1], in equal-width bins."""
        return {"n": self.n, "rel": [int(v) for v in self.rel.tolist()], "cos": [int(v) for v in self.cos.tolist()]}

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        cos_edges = torch.linspace(-1.0, 1.0, COS_BINS + 1, dtype=torch.float64)
        rel_edges = torch.linspace(0.0, REL_MAX, REL_BINS + 1, dtype=torch.float64)
        return {
            "n": self.n,
            "median_cos": _median(self.cos, cos_edges),
            "median_rel": _median(self.rel, rel_edges),
            "frac_cos_at_least": {str(t): _fraction_above(self.cos, cos_edges, t) for t in COS_THRESHOLDS},
            "frac_rel_at_most": {str(t): _fraction_below(self.rel, rel_edges, t) for t in REL_THRESHOLDS},
        }


def _median(counts: torch.Tensor, edges: torch.Tensor) -> float:
    cumulative = counts.cumsum(0)
    index = int(torch.searchsorted(cumulative, cumulative[-1] / 2).item())
    return float((edges[index] + edges[index + 1]) / 2)


def _fraction_below(counts: torch.Tensor, edges: torch.Tensor, threshold: float) -> float:
    """Share of values at or under ``threshold``, counting whole bins whose upper edge fits."""
    keep = edges[1:] <= threshold + 1e-9
    return float(counts[keep].sum() / counts.sum())


def _fraction_above(counts: torch.Tensor, edges: torch.Tensor, threshold: float) -> float:
    keep = edges[:-1] >= threshold - 1e-9
    return float(counts[keep].sum() / counts.sum())


def pair_metrics(first: torch.Tensor, second: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and rel for each row pair, after the k largest |difference| entries are set aside.

    ``first`` and ``second`` are [pairs, head_dim]. Set-aside entries of both vectors take
    the shared value m there, so only the remaining difference counts.
    """
    first = first.float()
    second = second.float()
    mean = (first + second) / 2
    delta = (first - second) / 2
    if k > 0:
        index = delta.abs().topk(k, dim=-1).indices
        delta = delta.scatter(-1, index, 0.0)
    a = mean + delta
    b = mean - delta
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
    rel = delta.norm(dim=-1) / mean.norm(dim=-1).clamp_min(1e-8)
    return cos, rel


def tier_counts(rels: torch.Tensor, threshold: float) -> torch.Tensor:
    """For each pair, the smallest residual size whose rel is within ``threshold``.

    ``rels`` is [len(ks), pairs], ordered by increasing k. Returns pair counts per k, plus a last
    entry for pairs no listed k brings within the threshold (those stay exact, unmerged).
    """
    ok = rels <= threshold
    any_ok = ok.any(dim=0)
    first = ok.float().argmax(dim=0)
    counts = torch.zeros(rels.shape[0] + 1, dtype=torch.float64)
    counts[:-1] = torch.bincount(first[any_ok].cpu(), minlength=rels.shape[0]).double()
    counts[-1] = float((~any_ok).sum())
    return counts


def tier_summary(counts: torch.Tensor, ks: tuple[int, ...], head_dim: int, names: tuple[str, ...] | None = None) -> dict:
    """Share of pairs per tier and the bytes left against a dense cache.

    A merged pair stores the mean (16 bits per feature), and when k > 0 also a one-bit-per-feature
    mask and k kept differences at 16 bits. An unmerged pair stays at 2 x 16 bits per feature.
    Per-pair flags are not counted.
    """
    total = float(counts.sum())
    shares = (counts / total).tolist()
    dense = 32 * head_dim
    bits = 0.0
    for k, share in zip(ks, shares[:-1]):
        bits += share * (16 * head_dim + (head_dim + 16 * k if k > 0 else 0))
    bits += shares[-1] * dense
    return {
        "pairs": int(total),
        "share_by_residual": {name: share for name, share in zip(names or tuple(str(k) for k in ks), shares[:-1])},
        "share_unmerged": shares[-1],
        "bytes_fraction_of_dense": bits / dense,
    }


def rel_curves(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """rel_k for every k = 0..head_dim, per pair: [pairs, head_dim + 1]."""
    first = first.float()
    second = second.float()
    mean = (first + second) / 2
    energy = ((first - second) / 2).pow(2).sort(dim=-1, descending=True).values
    return rest_energy(energy).sqrt() / mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class Reservoir:
    """Uniform random sample of pairs, kept by the largest random keys seen so far."""

    META = ("question", "head", "position", "norm_a", "norm_b", "norm_m", "norm_d", "cos")

    def __init__(self, size: int) -> None:
        self.size = size
        self.keys = torch.empty(0)
        self.curves = torch.empty(0, 0, dtype=torch.float16)
        self.meta = torch.empty(0, len(self.META))

    def add(self, first: torch.Tensor, second: torch.Tensor, head: torch.Tensor, position: torch.Tensor, question: int) -> None:
        count = first.shape[0]
        keys = torch.rand(count)
        top = keys.topk(min(self.size, count)).indices
        first, second = first[top.to(first.device)], second[top.to(second.device)]
        mean = (first.float() + second.float()) / 2
        delta = (first.float() - second.float()) / 2
        meta = torch.stack(
            [
                torch.full((len(top),), float(question)),
                head[top].float().cpu(),
                position[top].float().cpu(),
                first.float().norm(dim=-1).cpu(),
                second.float().norm(dim=-1).cpu(),
                mean.norm(dim=-1).cpu(),
                delta.norm(dim=-1).cpu(),
                torch.nn.functional.cosine_similarity(first.float(), second.float(), dim=-1, eps=1e-8).cpu(),
            ],
            dim=1,
        )
        curves = rel_curves(first, second).half().cpu()
        keys = keys[top]
        if self.keys.numel():
            keys = torch.cat([self.keys, keys])
            curves = torch.cat([self.curves, curves])
            meta = torch.cat([self.meta, meta])
        keep = keys.topk(min(self.size, keys.numel())).indices
        self.keys, self.curves, self.meta = keys[keep], curves[keep], meta[keep]


def movement_sums(rels: torch.Tensor, edges: tuple[float, ...] = BUCKET_EDGES) -> torch.Tensor:
    """Bucket pairs by rel before residuals and sum rel before and after for each k.

    ``rels`` is [len(ks), pairs] with row 0 the k = 0 values. Returns float64
    [len(ks), len(edges) + 1, 4]: pairs, sum before, sum after, sum after squared.
    """
    rels = rels.float().cpu()  # float64 sums run on the CPU: Apple's GPU backend has no float64
    before = rels[0]
    bucket = torch.bucketize(before, torch.tensor(edges), right=True)
    out = torch.zeros(rels.shape[0], len(edges) + 1, 4, dtype=torch.float64)
    ones = torch.ones_like(before, dtype=torch.float64)
    for row in range(rels.shape[0]):
        after = rels[row].double()
        out[row, :, 0].index_add_(0, bucket, ones)
        out[row, :, 1].index_add_(0, bucket, before.double())
        out[row, :, 2].index_add_(0, bucket, after)
        out[row, :, 3].index_add_(0, bucket, after * after)
    return out


def rest_energy(sorted_energy: torch.Tensor) -> torch.Tensor:
    """Difference energy left after setting aside the k biggest entries, for k = 0..dim.

    ``sorted_energy`` is d^2 in descending order along the last axis. Summing the remaining terms
    directly (not total minus the removed ones) keeps small values free of cancellation error.
    """
    tail = sorted_energy.flip(-1).cumsum(dim=-1).flip(-1)
    return torch.cat([tail, torch.zeros_like(tail[..., :1])], dim=-1)


def k_needed_counts(first: torch.Tensor, second: torch.Tensor, taus: tuple[float, ...] = TAU_GRID) -> torch.Tensor:
    """For each threshold, how many pairs need each residual size k = 0..head_dim.

    Setting aside the k largest |d| entries leaves the difference energy total - (sum of the k
    biggest d^2), so rel_k = sqrt(that) / |m| falls as k grows and the smallest k within a
    threshold is found by one search per pair. Returns float64 [len(taus), head_dim + 1].
    """
    first = first.float()
    second = second.float()
    mean = (first + second) / 2
    energy = ((first - second) / 2).pow(2).sort(dim=-1, descending=True).values
    rest = rest_energy(energy)  # [pairs, head_dim + 1]
    limit = (torch.tensor(taus, device=mean.device).unsqueeze(0) * mean.norm(dim=-1, keepdim=True)).pow(2)  # [pairs, taus]
    # rest is non-increasing along k, so the smallest k within the limit is the count of k whose rest exceeds it.
    needed = torch.searchsorted((-rest).contiguous(), (-limit).contiguous(), right=False)  # [pairs, taus]
    needed = needed.clamp(max=rest.shape[-1] - 1).cpu()  # float64 counts on the CPU: Apple's GPU backend has no float64
    counts = torch.zeros(len(taus), rest.shape[-1], dtype=torch.float64)
    counts.scatter_add_(1, needed.t().contiguous(), torch.ones(needed.t().shape, dtype=torch.float64))
    return counts


def bytes_from_kneeded(counts_row: torch.Tensor, tiers: tuple[int, ...], head_dim: int) -> tuple[float, float]:
    """Bytes against dense and share left unmerged, for one threshold row of the dump.

    A pair uses the smallest allowed tier that is at least its needed k; none allowed means it
    stays exact. Costs match ``tier_summary``.
    """
    total = float(counts_row.sum())
    bits = 0.0
    unmerged = 0.0
    previous = -1
    for k in sorted(tiers):
        group = float(counts_row[previous + 1 : k + 1].sum())
        bits += group * (16 * head_dim + (head_dim + 16 * k if k > 0 else 0))
        previous = k
    unmerged = float(counts_row[previous + 1 :].sum())
    bits += unmerged * 32 * head_dim
    return bits / (total * 32 * head_dim), unmerged / total


def aligned_pairs(states: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Split [1, heads, seq, dim] states at absolute positions start.. into pairs (2i, 2i+1).

    Returns two [heads * pairs, dim] tensors, or None when no complete pair is present.
    """
    seq = states.shape[-2]
    first_even = start + (start % 2)
    offset = first_even - start
    count = (seq - offset) // 2
    if count < 1:
        return None
    window = states[0, :, offset : offset + 2 * count]
    even = window[:, 0::2].reshape(-1, states.shape[-1])
    odd = window[:, 1::2].reshape(-1, states.shape[-1])
    return even, odd


class Profiler:
    """Collects per-layer histograms from Cache.update calls."""

    def __init__(self, rope: RopeTables | None, fractions: tuple[str, ...] = FRACTIONS, sample_pairs: int = 2000) -> None:
        self.rope = rope
        self.fractions = fractions
        self.ks: tuple[int, ...] = ()  # features kept per share, set once head_dim is seen
        self.sample_pairs = sample_pairs
        self.question = 0
        self.reservoirs: dict[tuple[int, str, str], Reservoir] = {}
        self.hist: dict[tuple[int, str, str, int], Histograms] = {}
        # Pairs per tier at each rel threshold: index i is fractions[i], the last entry is "left exact".
        self.tiers: dict[tuple[int, str, str, float], torch.Tensor] = {}
        self.head_dim: int | None = None
        # Pairs by the smallest k that reaches each threshold of TAU_GRID: [len(TAU_GRID), head_dim + 1].
        self.kneeded: dict[tuple[int, str, str], torch.Tensor] = {}
        # Per bucket of rel before: [len(ks), buckets, 4] = pairs, sum before, sum after, sum after squared.
        self.moves: dict[tuple[int, str, str], torch.Tensor] = {}
        self._generated: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        # L2 norm of every stored key and value: per question, {"k", "v": [layers, kv_heads, positions], "prompt_len"}.
        self.token_norms: list[dict] = []
        self._norms: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}

    def observe(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, start: int) -> None:
        """Called with the new K/V of one update. The first update of a sequence is scored now, later ones buffered."""
        self._norms.setdefault(layer_idx, []).append(
            (start, keys[0].float().norm(dim=-1).cpu(), values[0].float().norm(dim=-1).cpu())
        )
        if start == 0:
            self._score(layer_idx, keys, values, start, "prefill")
        else:
            self._generated.setdefault(layer_idx, []).append((start, keys.detach(), values.detach()))

    def finish_sequence(self) -> None:
        """Score the pairs made of generated tokens, then clear the buffers."""
        for layer_idx, chunks in self._generated.items():
            chunks.sort(key=lambda item: item[0])
            start = chunks[0][0]
            keys = torch.cat([item[1] for item in chunks], dim=-2)
            values = torch.cat([item[2] for item in chunks], dim=-2)
            self._score(layer_idx, keys, values, start, "decode")
        self._generated = {}
        if self._norms:
            layers = sorted(self._norms)
            for chunks in self._norms.values():
                chunks.sort(key=lambda item: item[0])
            prompt_len = next((item[1].shape[-1] for item in self._norms[layers[0]] if item[0] == 0), 0)
            self.token_norms.append(
                {
                    "k": torch.stack([torch.cat([item[1] for item in self._norms[i]], dim=-1) for i in layers]),
                    "v": torch.stack([torch.cat([item[2] for item in self._norms[i]], dim=-1) for i in layers]),
                    "prompt_len": prompt_len,
                }
            )
        self._norms = {}

    def _score(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, start: int, phase: str) -> None:
        views = {"k_rope": keys, "v": values}
        if self.rope is not None:
            positions = torch.arange(start, start + keys.shape[-2], device=keys.device)
            cos, sin = self.rope.cos_sin(positions, keys.dtype)
            views["k_plain"] = apply_rope(keys, cos, sin, inverse=True)
        for kind, states in views.items():
            pairs = aligned_pairs(states, start)
            if pairs is None:
                continue
            self.head_dim = states.shape[-1]
            self.ks = ks_for(self.fractions, self.head_dim)
            if self.sample_pairs > 0:
                per_head = pairs[0].shape[0] // states.shape[1]
                index = torch.arange(pairs[0].shape[0])
                head = index // per_head
                position = start + (start % 2) + 2 * (index % per_head)
                reservoir = self.reservoirs.setdefault((layer_idx, kind, phase), Reservoir(self.sample_pairs))
                reservoir.add(pairs[0], pairs[1], head, position, self.question)
            rels = []
            for fraction, k in zip(self.fractions, self.ks):
                cos, rel = pair_metrics(pairs[0], pairs[1], k)
                self.hist.setdefault((layer_idx, kind, phase, fraction), Histograms()).add(cos, rel)
                rels.append(rel)
            moved = movement_sums(torch.stack(rels))
            self.moves[(layer_idx, kind, phase)] = self.moves.get((layer_idx, kind, phase), torch.zeros_like(moved)) + moved
            counts = k_needed_counts(pairs[0], pairs[1])
            key = (layer_idx, kind, phase)
            self.kneeded[key] = self.kneeded.get(key, torch.zeros_like(counts)) + counts
            for threshold in TIER_THRESHOLDS:
                counts = tier_counts(torch.stack(rels), threshold)
                key = (layer_idx, kind, phase, threshold)
                self.tiers[key] = self.tiers.get(key, torch.zeros_like(counts)) + counts

    def sample_arrays(self) -> dict:
        """Arrays for numpy.savez: '<kind>/<phase>/<layer>/curve' and '.../meta' per cell."""
        out = {}
        for (layer_idx, kind, phase), reservoir in sorted(self.reservoirs.items()):
            out[f"{kind}/{phase}/{layer_idx}/curve"] = reservoir.curves.numpy()
            out[f"{kind}/{phase}/{layer_idx}/meta"] = reservoir.meta.numpy()
        return out

    def norm_arrays(self) -> dict:
        """Arrays for numpy.savez: 'question_NNNN/k', '.../v' as [layers, kv_heads, positions] and '.../prompt_len'."""
        out = {}
        for number, norms in enumerate(self.token_norms):
            out[f"question_{number:04d}/k"] = norms["k"].half().numpy()
            out[f"question_{number:04d}/v"] = norms["v"].half().numpy()
            out[f"question_{number:04d}/prompt_len"] = torch.tensor(norms["prompt_len"]).numpy()
        return out

    def norm_stats(self) -> dict:
        """Per kind (k, v), phase and layer: norm count, mean, std and quantiles, over all heads and per KV head."""
        quantiles = (0.01, 0.1, 0.5, 0.9, 0.99)

        def describe(values: torch.Tensor) -> dict:
            values = values.double()
            q = torch.quantile(values, torch.tensor(quantiles, dtype=torch.float64)) if values.numel() < 2**24 else None
            if q is None:  # torch.quantile has an input size limit; sort instead
                ordered = values.sort().values
                q = ordered[(torch.tensor(quantiles) * (values.numel() - 1)).long()]
            return {
                "n": int(values.numel()), "mean": float(values.mean()), "std": float(values.std()) if values.numel() > 1 else 0.0,
                "max": float(values.max()), **{f"p{int(round(100 * x))}": float(v) for x, v in zip(quantiles, q.tolist())},
            }

        out: dict = {"quantiles": list(quantiles), "stats": {}}
        if not self.token_norms:
            return out
        for kind in ("k", "v"):
            for phase in PHASES:
                parts = []
                for norms in self.token_norms:
                    cut = norms["prompt_len"]
                    parts.append(norms[kind][:, :, :cut] if phase == "prefill" else norms[kind][:, :, cut:])
                values = torch.cat(parts, dim=-1)  # [layers, kv_heads, tokens]
                if values.shape[-1] == 0:
                    continue
                for layer in range(values.shape[0]):
                    out["stats"].setdefault(kind, {}).setdefault(phase, {})[str(layer)] = {
                        "all": describe(values[layer].flatten()),
                        "heads": [describe(values[layer, head]) for head in range(values.shape[1])],
                    }
        return out

    def histogram_dump(self) -> dict:
        """Full rel and cos histograms per (kind, phase, residual share, layer): the pair distribution before ("0") and after residuals."""
        out: dict = {
            "rel_max": REL_MAX, "rel_bins": REL_BINS, "cos_bins": COS_BINS,
            "fractions": list(self.fractions), "ks": list(self.ks), "hist": {},
        }
        for (layer_idx, kind, phase, fraction), histograms in sorted(self.hist.items()):
            out["hist"].setdefault(kind, {}).setdefault(phase, {}).setdefault(fraction, {})[str(layer_idx)] = histograms.raw()
        return out

    def moves_dump(self) -> dict:
        """Per (kind, phase, layer): [residual share][bucket] = [pairs, sum rel before, sum rel after, sum rel after squared]."""
        out: dict = {"bucket_edges": list(BUCKET_EDGES), "fractions": list(self.fractions), "ks": list(self.ks), "moves": {}}
        for (layer_idx, kind, phase), sums in sorted(self.moves.items()):
            out["moves"].setdefault(kind, {}).setdefault(phase, {})[str(layer_idx)] = [
                [[round(v, 6) for v in cell] for cell in row] for row in sums.tolist()
            ]
        return out

    def kneeded_dump(self) -> dict:
        """Pair counts by needed k per (kind, phase, layer), for every threshold of TAU_GRID.

        Column k is a residual of k features, a share k / head_dim of the vector.
        """
        out: dict = {"taus": list(TAU_GRID), "head_dim": self.head_dim, "counts": {}}
        for (layer_idx, kind, phase), counts in sorted(self.kneeded.items()):
            out["counts"].setdefault(kind, {}).setdefault(phase, {})[str(layer_idx)] = [
                [int(v) for v in row] for row in counts.tolist()
            ]
        return out

    def summary(self) -> dict:
        out: dict = {}
        for (layer_idx, kind, phase, fraction), histograms in sorted(self.hist.items()):
            out.setdefault(kind, {}).setdefault(phase, {}).setdefault(fraction, {})[str(layer_idx)] = histograms.summary()
        for (layer_idx, kind, phase, threshold), counts in sorted(self.tiers.items()):
            tiers = out.setdefault(kind, {}).setdefault(phase, {}).setdefault("tiers", {}).setdefault(str(threshold), {})
            tiers[str(layer_idx)] = tier_summary(counts, self.ks, self.head_dim or 0, self.fractions)
        return out


class KvDumper:
    """Collects the K/V (and queries) of one sequence per layer, then writes out random windows of it.

    ``windows`` windows of ``window`` consecutive tokens are drawn per sequence, without
    replacement, from the blocks [0, window), [window, 2 window), ...; 0 keeps every position.
    The draw is seeded by ``seed`` and the sequence number, so a rerun keeps the same tokens.
    """

    def __init__(self, directory: Path, windows: int = 8, window: int = 16, seed: int = 0, queries: bool = True) -> None:
        if window < 2 or window % 2:
            raise ValueError("window must be even and at least 2, so aligned pairs stay whole")
        self.directory = directory
        self.windows = windows
        self.window = window
        self.seed = seed
        self.queries = queries
        directory.mkdir(parents=True, exist_ok=True)
        self.prompt_lengths: list[int] = []
        self.pending_query: torch.Tensor | None = None
        self._reset()

    def _reset(self) -> None:
        self.layers: dict[int, dict[str, list[torch.Tensor]]] = {}

    def record(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        slot = self.layers.setdefault(layer_idx, {"k": [], "v": [], "q": []})
        slot["k"].append(keys.detach().cpu())
        slot["v"].append(values.detach().cpu())
        if self.pending_query is not None and self.queries:
            slot["q"].append(self.pending_query.detach().cpu())
            self.pending_query = None

    def save(self, question: int, ids: torch.Tensor) -> Path:
        from safetensors.torch import save_file

        order = sorted(self.layers)
        states = {
            "keys": torch.stack([torch.cat(self.layers[i]["k"], dim=-2)[0] for i in order]),
            "values": torch.stack([torch.cat(self.layers[i]["v"], dim=-2)[0] for i in order]),
        }
        if self.queries and all(self.layers[i]["q"] for i in order):
            states["queries"] = torch.stack([torch.cat(self.layers[i]["q"], dim=-2)[0] for i in order])
        positions = self.kept_positions(states["keys"].shape[-2], question)
        tensors = {name: tensor[:, :, positions].contiguous() for name, tensor in states.items()}
        tensors["positions"] = positions.contiguous()
        tensors["input_ids"] = ids.cpu().long().contiguous()
        path = self.directory / f"question_{question:04d}.safetensors"
        metadata = {"prompt_len": str(int(ids.shape[-1])), "question": str(question), "stored_len": str(states["keys"].shape[-2])}
        save_file(tensors, str(path), metadata=metadata)
        self.prompt_lengths.append(int(ids.shape[-1]))
        self._reset()
        return path

    def kept_positions(self, length: int, question: int) -> torch.Tensor:
        """Sorted token positions to keep: every position, or ``windows`` random whole blocks."""
        blocks = length // self.window
        if self.windows <= 0 or self.windows >= blocks:
            return torch.arange(length if self.windows <= 0 else blocks * self.window)
        chosen = sorted(random.Random(self.seed * 1_000_003 + question).sample(range(blocks), self.windows))
        return torch.cat([torch.arange(b * self.window, (b + 1) * self.window) for b in chosen])


def run_profile(
    model, prompts: list[torch.Tensor], *, max_new_tokens: int, rope: RopeTables | None, fractions=FRACTIONS, generate_kwargs=None, sample_pairs: int = 2000,
    dump_dir: Path | None = None, dump_windows: int = 8, dump_window: int = 16, dump_seed: int = 0, dump_queries: bool = True, profiler: Profiler | None = None,
) -> Profiler:
    """Generate greedily for each prompt (1-D token ids) while a hook scores every cache update."""
    from transformers.cache_utils import Cache

    if not fractions or Fraction(fractions[0]) != 0:
        raise ValueError('fractions must start with "0": the movement dump buckets pairs by their rel before any residual')
    profiler = profiler if profiler is not None else Profiler(rope, fractions, sample_pairs)
    original = Cache.update
    dumper = KvDumper(dump_dir, dump_windows, dump_window, dump_seed, dump_queries) if dump_dir is not None else None
    llama = None
    original_rotary = None
    if dumper is not None and dump_queries:
        try:
            import transformers.models.llama.modeling_llama as llama

            original_rotary = llama.apply_rotary_pos_emb
        except (ImportError, AttributeError):
            llama = None  # no query capture for this model family

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        length = _stored_length(self, layer_idx)
        profiler.observe(layer_idx, key_states, value_states, length)
        if dumper is not None:
            dumper.record(layer_idx, key_states, value_states)
        return original(self, key_states, value_states, layer_idx, *args, **kwargs)

    def rotary(query, key, *args, **kwargs):
        # Runs right before the layer's cache update, which attributes the query to its layer.
        query_embed, key_embed = original_rotary(query, key, *args, **kwargs)
        dumper.pending_query = query_embed
        return query_embed, key_embed

    Cache.update = update
    if llama is not None:
        llama.apply_rotary_pos_emb = rotary
    try:
        for number, ids in enumerate(prompts):
            profiler.question = number
            batch = ids.unsqueeze(0).to(model.device)
            with torch.no_grad():
                if max_new_tokens == 0:
                    # Prefill only: one forward pass fills the cache through Cache.update.
                    model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=True)
                else:
                    model.generate(
                        input_ids=batch,
                        attention_mask=torch.ones_like(batch),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        **(generate_kwargs or {}),
                    )
            profiler.finish_sequence()
            if dumper is not None:
                dumper.save(number, ids)
    finally:
        Cache.update = original
        if llama is not None:
            llama.apply_rotary_pos_emb = original_rotary
    return profiler


def _stored_length(cache, layer_idx: int) -> int:
    try:
        return int(cache.get_seq_length(layer_idx))
    except TypeError:
        return int(cache.get_seq_length())


def rope_tables(model) -> RopeTables | None:
    from engine.kv_compress.rope import rope_from_config

    config = model.config
    tables = rope_from_config(getattr(config, "text_config", None) or config)
    if tables is None:
        return None
    rotary = getattr(getattr(model, "model", model), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", 1.0)
    return RopeTables(
        rope_theta=tables.rope_theta,
        head_dim=tables.head_dim,
        inv_freq=None if inv_freq is None else tuple(float(v) for v in inv_freq.detach().float().cpu().tolist()),
        attention_scaling=float(scaling) if isinstance(scaling, (int, float)) else 1.0,
    )




def gsm8k_prompts(tokenizer, count: int, shots: int = 5, seed: int = 0) -> tuple[list[torch.Tensor], list[int]]:
    """Few-shot GSM8K prompts in the lm-eval layout, on ``count`` test questions drawn at random."""
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main")
    train = data["train"]
    test = data["test"]
    shot_text = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(shots))
    prompts = []
    chosen = random.Random(seed).sample(range(len(test)), count)
    for i in chosen:
        text = shot_text + f"Question: {test[i]['question']}\nAnswer:"
        prompts.append(torch.tensor(tokenizer(text)["input_ids"]))
    return prompts, chosen


def markdown_tables(summary: dict, *, rel_threshold: str = "0.3", fractions=FRACTIONS) -> str:
    """Per layer: share of pairs with rel at most the threshold, for each residual share, per kind and phase."""
    lines = []
    for kind in KINDS:
        for phase in PHASES:
            by_k = summary.get(kind, {}).get(phase)
            if not by_k:
                continue
            layers = sorted(int(layer) for layer in by_k[fractions[0]])
            lines.append(f"### {kind}, {phase}: share of pairs with L2 ||d||/||m|| <= {rel_threshold}, by residual kept")
            lines.append("")
            lines.append("| layer | " + " | ".join(label(f) for f in fractions) + " | pairs |")
            lines.append("|---|" + "---|" * (len(fractions) + 1))
            for layer in layers:
                cells = []
                for fraction in fractions:
                    entry = by_k[fraction][str(layer)]
                    cells.append(f"{entry['frac_rel_at_most'][rel_threshold]:.2f}")
                pairs = by_k[fractions[0]][str(layer)]["n"]
                lines.append(f"| {layer} | " + " | ".join(cells) + f" | {pairs} |")
            lines.append("")
    for kind in KINDS:
        for phase in PHASES:
            tiers = summary.get(kind, {}).get(phase, {}).get("tiers")
            if not tiers:
                continue
            for threshold, by_layer in tiers.items():
                lines.append(
                    f"### {kind}, {phase}: merge a pair if some residual brings L2 ||d||/||m|| <= {threshold} "
                    "(smallest such residual is stored)"
                )
                lines.append("")
                lines.append("| layer | " + " | ".join(label(f) for f in fractions) + " | unmerged | bytes vs dense |")
                lines.append("|---|" + "---|" * (len(fractions) + 2))
                for layer in sorted(by_layer, key=int):
                    entry = by_layer[layer]
                    cells = [f"{entry['share_by_residual'][f]:.2f}" for f in fractions]
                    lines.append(
                        f"| {layer} | " + " | ".join(cells)
                        + f" | {entry['share_unmerged']:.2f} | {entry['bytes_fraction_of_dense']:.2f} |"
                    )
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=PRETRAINED)
    parser.add_argument("--dataset", choices=("wikitext", "gsm8k"), default="wikitext", help="wikitext: prefill only, one forward pass per chunk; gsm8k: prefill and decode")
    parser.add_argument("--samples", "--questions", dest="samples", type=int, default=30, help="WikiText chunks or GSM8K questions, drawn at random")
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens per WikiText chunk")
    parser.add_argument("--shots", type=int, default=5, help="GSM8K only")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the chunks or questions")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="GSM8K only; WikiText never generates")
    parser.add_argument("--dump-kv", type=Path, default=None, help="directory to save exact K, V and queries for sampled windows of every sequence")
    parser.add_argument("--dump-windows", type=int, default=8, help="random windows kept per sequence in the raw dump; 0 keeps whole sequences")
    parser.add_argument("--dump-window", type=int, default=16, help="consecutive tokens per dump window (even)")
    parser.add_argument("--dump-queries", action=argparse.BooleanOptionalAction, default=True, help="also dump queries (about 2/3 of the dump on Llama 3.1 8B)")
    parser.add_argument("--sample-pairs", type=int, default=2000, help="random pairs kept per (kind, phase, layer) with their full curve; 0 to skip")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu; default is the best one available")
    parser.add_argument("--dtype", default=None, help="bfloat16, float16 or float32; default bfloat16 (float32 on cpu)")
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float32 if device == "cpu" else torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    if args.dataset == "wikitext":
        prompts, chosen = wikitext_chunks(tokenizer, args.samples, args.seq_len, args.seed)
        max_new_tokens, generate_kwargs = 0, None
    else:
        prompts, chosen = gsm8k_prompts(tokenizer, args.samples, args.shots, args.seed)
        max_new_tokens, generate_kwargs = args.max_new_tokens, {"stop_strings": ["Question:"], "tokenizer": tokenizer}
    rope = rope_tables(model)
    profiler = run_profile(
        model,
        prompts,
        max_new_tokens=max_new_tokens,
        rope=rope,
        dump_dir=args.dump_kv,
        dump_windows=args.dump_windows,
        dump_window=args.dump_window,
        dump_seed=args.seed,
        dump_queries=args.dump_queries,
        generate_kwargs=generate_kwargs,
        sample_pairs=args.sample_pairs,
    )
    source = {
        "dataset": args.dataset,
        "samples": args.samples,
        "seed": args.seed,
        "chosen": chosen,  # WikiText chunk indices of --seq-len tokens, or GSM8K test indices
        **({"seq_len": args.seq_len} if args.dataset == "wikitext" else {"shots": args.shots, "max_new_tokens": max_new_tokens}),
    }
    summary = profiler.summary()
    if args.dump_kv is not None:
        (args.dump_kv / "meta.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    **source,
                    "dtype": str(dtype),
                    "dump_windows": args.dump_windows,
                    "dump_window": args.dump_window,
                    "dump_queries": args.dump_queries,
                    "rope": None
                    if rope is None
                    else {
                        "rope_theta": rope.rope_theta,
                        "head_dim": rope.head_dim,
                        "inv_freq": list(rope.inv_freq) if rope.inv_freq else None,
                        "attention_scaling": rope.attention_scaling,
                    },
                    "layout": "keys/values [layers, kv_heads, kept positions, head_dim]; queries [layers, heads, kept positions, head_dim]; 'positions' lists the kept token positions; RoPE applied",
                },
                indent=1,
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        **source,
        "fractions": list(FRACTIONS),
        "ks": list(profiler.ks),
        "head_dim": profiler.head_dim,
        "rel": "L2: ||d||_2 / ||m||_2",
        "cos_thresholds": list(COS_THRESHOLDS),
        "rel_thresholds": list(REL_THRESHOLDS),
        "tier_thresholds": list(TIER_THRESHOLDS),
    }
    (args.out / "profile.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    import numpy

    if args.sample_pairs > 0:
        numpy.savez_compressed(args.out / "profile_samples.npz", **profiler.sample_arrays())
    numpy.savez_compressed((args.dump_kv or args.out) / "profile_norms.npz", **profiler.norm_arrays())
    (args.out / "profile_norm_stats.json").write_text(json.dumps(profiler.norm_stats(), separators=(",", ":")))
    (args.out / "profile_moves.json").write_text(json.dumps(profiler.moves_dump(), separators=(",", ":")))
    (args.out / "profile_hist.json").write_text(json.dumps(profiler.histogram_dump(), separators=(",", ":")))
    (args.out / "profile_kneeded.json").write_text(json.dumps(profiler.kneeded_dump(), separators=(",", ":")))
    (args.out / "profile.md").write_text(markdown_tables(summary))
    print(markdown_tables(summary))
    print(f"wrote {args.out}/profile.json and profile.md")


if __name__ == "__main__":
    main()
