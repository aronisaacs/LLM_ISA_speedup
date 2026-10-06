#!/usr/bin/env python3
"""How similar are adjacent token pairs, per layer, in prefill and in decode?

Read-only profiling. No compression is applied. A hook on Cache.update sees the
exact K/V a normal run stores, and every aligned pair (positions 2i, 2i+1) is
compared per KV head. Pairs from the prompt are the prefill phase. Pairs made
only of generated tokens are the decode phase.

For each pair, with m = (a + b) / 2 and d = (a - b) / 2:
  cos    cosine of a and b
  rel    |d| / |m|, the size of the difference against the shared part
and the same two numbers again after the k features with the largest |d| are
set aside (a and b both take the value m there, as if those entries were kept in
an exact residual). k = 0 is the plain pair.

Keys are measured twice: as stored (RoPE applied) and with RoPE undone.

  python compression_topics/spatial/scripts/pair_similarity_profile.py --questions 30
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

from engine.kv_compress.rope import RopeTables, apply_rope  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_similarity"
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
KS = (0, 8, 16, 32)
COS_THRESHOLDS = (0.5, 0.75, 0.9, 0.95, 0.99)
REL_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
REL_BINS = 400  # rel in [0, 4]
COS_BINS = 400  # cos in [-1, 1]
REL_MAX = 4.0
KINDS = ("k_rope", "k_plain", "v")
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

    def __init__(self, rope: RopeTables | None, ks: tuple[int, ...] = KS) -> None:
        self.rope = rope
        self.ks = ks
        self.hist: dict[tuple[int, str, str, int], Histograms] = {}
        self._generated: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}

    def observe(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, start: int) -> None:
        """Called with the new K/V of one update. The first update of a sequence is scored now, later ones buffered."""
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
            for k in self.ks:
                cos, rel = pair_metrics(pairs[0], pairs[1], k)
                self.hist.setdefault((layer_idx, kind, phase, k), Histograms()).add(cos, rel)

    def summary(self) -> dict:
        out: dict = {}
        for (layer_idx, kind, phase, k), histograms in sorted(self.hist.items()):
            out.setdefault(kind, {}).setdefault(phase, {}).setdefault(str(k), {})[str(layer_idx)] = histograms.summary()
        return out


def run_profile(model, prompts: list[torch.Tensor], *, max_new_tokens: int, rope: RopeTables | None, ks=KS, generate_kwargs=None) -> Profiler:
    """Generate greedily for each prompt (1-D token ids) while a hook scores every cache update."""
    from transformers.cache_utils import Cache

    profiler = Profiler(rope, ks)
    original = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        length = _stored_length(self, layer_idx)
        profiler.observe(layer_idx, key_states, value_states, length)
        return original(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update = update
    try:
        for ids in prompts:
            batch = ids.unsqueeze(0).to(model.device)
            with torch.no_grad():
                model.generate(
                    input_ids=batch,
                    attention_mask=torch.ones_like(batch),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    **(generate_kwargs or {}),
                )
            profiler.finish_sequence()
    finally:
        Cache.update = original
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


def gsm8k_prompts(tokenizer, count: int, shots: int = 5) -> list[torch.Tensor]:
    """Few-shot GSM8K prompts in the lm-eval layout ("Question: ...\\nAnswer: ...")."""
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main")
    train = data["train"]
    test = data["test"]
    shot_text = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(shots))
    prompts = []
    for i in range(count):
        text = shot_text + f"Question: {test[i]['question']}\nAnswer:"
        prompts.append(torch.tensor(tokenizer(text)["input_ids"]))
    return prompts


def markdown_tables(summary: dict, *, rel_threshold: str = "0.3", ks=KS) -> str:
    """Per layer: share of pairs with rel at most the threshold, for each k, per kind and phase."""
    lines = []
    for kind in KINDS:
        for phase in PHASES:
            by_k = summary.get(kind, {}).get(phase)
            if not by_k:
                continue
            layers = sorted(int(layer) for layer in by_k[str(ks[0])])
            lines.append(f"### {kind}, {phase}: share of pairs with |d|/|m| <= {rel_threshold}")
            lines.append("")
            lines.append("| layer | " + " | ".join(f"k={k}" for k in ks) + " | pairs |")
            lines.append("|---|" + "---|" * (len(ks) + 1))
            for layer in layers:
                cells = []
                for k in ks:
                    entry = by_k[str(k)][str(layer)]
                    cells.append(f"{entry['frac_rel_at_most'][rel_threshold]:.2f}")
                pairs = by_k[str(ks[0])][str(layer)]["n"]
                lines.append(f"| {layer} | " + " | ".join(cells) + f" | {pairs} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=PRETRAINED)
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    prompts = gsm8k_prompts(tokenizer, args.questions, args.shots)
    profiler = run_profile(
        model,
        prompts,
        max_new_tokens=args.max_new_tokens,
        rope=rope_tables(model),
        generate_kwargs={"stop_strings": ["Question:"], "tokenizer": tokenizer},
    )
    summary = profiler.summary()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "questions": args.questions,
        "shots": args.shots,
        "max_new_tokens": args.max_new_tokens,
        "ks": list(KS),
        "cos_thresholds": list(COS_THRESHOLDS),
        "rel_thresholds": list(REL_THRESHOLDS),
    }
    (args.out / "profile.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    (args.out / "profile.md").write_text(markdown_tables(summary))
    print(markdown_tables(summary))
    print(f"wrote {args.out}/profile.json and profile.md")


if __name__ == "__main__":
    main()
