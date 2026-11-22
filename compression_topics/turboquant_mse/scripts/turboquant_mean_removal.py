#!/usr/bin/env python3
"""Does removing the shared direction of K / V before TurboQuant-mse buy bits? Read-only measurement.

Same model, text and streaming as turboquant_code_locality.py (Llama 3.1 8B, --chunks WikiText-2 test chunks of
--seq-len tokens, one decoder layer at a time on MPS, bfloat16). Every layer sees exact K/V; the variants below are
applied to that layer's K or V and scored against exact attention with the layer's own queries.

Each variant stores a mean m (per token position, see below) and quantizes the residual x - m with turboquant_mse
(uniform bits, signs="rope_pairs", Hadamard, Gaussian Lloyd-Max, own fp16 norm); the reconstruction is
x^ = m + Q(x - m), which is exactly what the read path would compute (score = q.m + q.Q(x - m)).

K (stored after RoPE):
  plain       m = 0 (today's TurboQuant)
  const       m = one constant vector, the mean post-RoPE key of the KV head (as FlagRot / OptR). q.m is the same for
              every key of a query, so softmax ignores it: no correction term, no storage, no read work
  const@P     the same constant m (calibrated on positions 0..seq_len-1) used on keys at positions P..P+seq_len-1:
              q and k are both turned by RoPE's extra angle P (exact attention unchanged), so only the mismatch of a
              short-context mean at long positions is measured
  global      m_t = R_t mu, mu the mean pre-RoPE key of the KV head, R_t RoPE at position t (option a)
  pairs<n>    the same with mu kept only in its n RoPE pairs (i, i + 64) of largest energy (option c)
  local<G>    m = mean of the post-RoPE keys of an aligned group of G tokens, stored fp16 (option b)
  local<G>q4  the same with the group mean itself stored as 4-bit TurboQuant (plus its fp16 norm)
V: plain, global (mu the mean value), local<G>, local<G>q4.
Means are estimated without BOS (position 0); BOS is always quantized with m = 0 (its own group). mu is leave-one-
chunk-out (from the other chunks), as an offline calibration would be. Local groups start at position 1; the last,
shorter group uses its own mean.

Per (variant, bits, layer, KV head):
  mse    sum ||x - x^||^2 / sum ||x||^2 over positions 1..
  kl     K variants: mean KL(exact attention || attention with K^), V exact (causal, the 4 query heads of the KV head,
         --queries sampled query positions per chunk)
  out    mean ||o^ - o|| / ||o|| of the attention output (the other side exact)
Per (layer, KV head): energy share of each mean, 1 - sum||x - m||^2 / sum||x||^2, and the share of ||mu||^2 (K, pre-RoPE)
in its top 1/2/4/8/16 RoPE pairs.

Storage per coordinate (bits): b + 16/128 (norm); local<G> adds 16/G, local<G>q4 adds (4*128 + 16)/(128 G); mu is one
vector per KV head and layer (not counted).

Check (always): the exact attention recomputed here from the captured q, K, V vs the layer's own attention output
(o_proj input), relative error per layer.

  python compression_topics/turboquant_mse/scripts/turboquant_mean_removal.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.turboquant_mse.algorithms import turboquant_mse as tq  # noqa: E402
from compression_topics.turboquant_mse.scripts.turboquant_code_locality import MODEL_8B, stream_kv  # noqa: E402

OUT = ROOT / "compression_topics" / "turboquant_mse" / "mean_removal" / "llama31"
PLAIN_BITS = (1, 2, 3, 4, 5, 6)  # the reference curve for "equivalent bits"
GROUPS = (8, 32, 128)
PAIRS = (4, 8, 16)
OFFSETS = (2048, 8192, 32768, 129024)  # const@P: positions the calibrated constant mean is used at
Q4_GROUPS = (32, 128)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x, cos, sin):
    """HF RoPE on [..., T, D] with cos/sin [T, D] (float32)."""
    return x * cos + rotate_half(x) * sin


def quant(x, layer, target, bits):
    return tq.apply(x.float().contiguous(), layer_idx=layer, target=target, bits=bits, signs="rope_pairs")


def loo_mean(x):
    """Leave-one-chunk-out mean over positions 1.. : [C, H, 1, D] (chunk c's mean from the other chunks)."""
    sums = x[:, :, 1:].float().sum(2, keepdim=True)
    count = x.shape[2] - 1
    return (sums.sum(0, keepdim=True) - sums) / (count * (x.shape[0] - 1))


def group_index(tokens, size):
    idx = torch.zeros(tokens, dtype=torch.long)
    idx[1:] = torch.arange(tokens - 1) // size + 1
    return idx


def group_mean(x, size):
    """Mean of aligned groups of ``size`` positions from 1 on, expanded to every position (BOS: 0)."""
    c, h, t, d = x.shape
    idx = group_index(t, size)
    groups = int(idx.max()) + 1
    sums = torch.zeros(c, h, groups, d).index_add_(2, idx, x.float())
    counts = torch.bincount(idx, minlength=groups).float().view(1, 1, groups, 1)
    means = sums / counts
    means[:, :, 0] = 0
    return means, idx


def top_pairs(mu, n):
    """mu [C, H, 1, D] with only its n RoPE pairs (i, i + D/2) of largest energy kept."""
    half = mu.shape[-1] // 2
    energy = mu[..., :half] ** 2 + mu[..., half:] ** 2
    keep = torch.zeros_like(energy, dtype=torch.bool)
    keep.scatter_(-1, energy.topk(n, dim=-1).indices, True)
    return mu * torch.cat((keep, keep), dim=-1)


def pair_concentration(mu, ns=(1, 2, 4, 8, 16)):
    half = mu.shape[-1] // 2
    energy = (mu[..., :half] ** 2 + mu[..., half:] ** 2).squeeze(2)  # [C, H, half]
    srt = energy.sort(-1, descending=True).values.cumsum(-1) / energy.sum(-1, keepdim=True)
    return {str(n): srt[..., n - 1].mean(0).tolist() for n in ns}


def per_head(x):
    return x.float().transpose(0, 1).reshape(x.shape[1], -1).mean(1).tolist()


def energy_share(x, m):
    x = x[:, :, 1:].float()
    m = m[:, :, 1:].float() if m.shape[2] > 1 else m.float()
    num = ((x - m) ** 2).sum(-1).transpose(0, 1).reshape(x.shape[1], -1).sum(1)
    den = (x ** 2).sum(-1).transpose(0, 1).reshape(x.shape[1], -1).sum(1)
    return (1 - num / den).tolist()


def rel_mse(x, xh):
    x, xh = x[:, :, 1:].float(), xh[:, :, 1:].float()
    num = ((x - xh) ** 2).sum(-1).transpose(0, 1).reshape(x.shape[1], -1).sum(1)
    den = (x ** 2).sum(-1).transpose(0, 1).reshape(x.shape[1], -1).sum(1)
    return (num / den).tolist()


class Attention:
    """Causal attention of the sampled queries of one layer, exact and with one side replaced."""

    def __init__(self, q, k, v, pos, device):
        self.device, self.pos = device, pos
        self.q = q[:, :, pos].float().to(device)  # [C, Hq, S, D]
        self.group = q.shape[1] // k.shape[1]
        t = k.shape[2]
        self.mask = (torch.arange(t).view(1, -1) > pos.view(-1, 1)).to(device)  # [S, T]
        self.scale = 1 / math.sqrt(q.shape[-1])
        self.k, self.v = k.float().to(device), v.float().to(device)
        self.logp = self._logp(self.k)
        self.p = self.logp.exp()
        self.o = self._out(self.p, self.v)

    def _rep(self, x):
        return x.repeat_interleave(self.group, dim=1)

    def _logp(self, k):
        s = (self.q @ self._rep(k).transpose(-1, -2)) * self.scale
        return torch.log_softmax(s.masked_fill(self.mask, float("-inf")), dim=-1)

    def _out(self, p, v):
        return p @ self._rep(v)

    def _kv_head(self, x):  # [C, Hq, S] -> per KV head
        c, hq, s = x.shape
        return x.view(c, hq // self.group, self.group, s).transpose(0, 1).reshape(hq // self.group, -1).mean(1).tolist()

    def _out_err(self, o):
        return (o - self.o).norm(dim=-1) / self.o.norm(dim=-1).clamp_min(1e-30)

    def with_k(self, kh):
        logp = self._logp(kh.float().to(self.device))
        kl = (self.p * (self.logp - logp)).nan_to_num(0.0).sum(-1)  # masked keys: p = 0
        return {"kl": self._kv_head(kl.cpu()), "out": self._kv_head(self._out_err(self._out(logp.exp(), self.v)).cpu())}

    def with_v(self, vh):
        return {"out": self._kv_head(self._out_err(self._out(self.p, vh.float().to(self.device))).cpu())}

    def check(self, attn_out):
        """Relative error of the recomputed exact output vs the layer's own (o_proj input)."""
        c, t, _ = attn_out.shape
        own = attn_out.view(c, t, -1, self.q.shape[-1]).transpose(1, 2)[:, :, self.pos].float().to(self.device)
        return float(((self.o - own).norm(dim=-1) / own.norm(dim=-1)).mean())


def k_variants(k, k_pre, layer, cos, sin, bits_list, offsets):
    """Yield (name, bits, K^) for every K variant, and the energy stats.

    ``offsets`` {P: (cos_P, sin_P)}: RoPE of the extra angle P (a [1, D] row each).
    """
    mu = loo_mean(k_pre)
    stats = {"pairs_share": pair_concentration(mu)}
    const = loo_mean(k).expand(-1, -1, k.shape[2], -1).clone()
    const[:, :, 0] = 0
    means = {"const": const, "global": rope(mu, cos, sin)}
    for n in PAIRS:
        means[f"pairs{n}"] = rope(top_pairs(mu, n), cos, sin)
    for g in GROUPS:
        gm, idx = group_mean(k, g)
        means[f"local{g}"] = gm.half().float()[:, :, idx]
        if g in Q4_GROUPS:
            means[f"local{g}q4"] = quant(gm, layer, "k", 4)[:, :, idx]
    for name, m in means.items():
        m = m.clone() if m.shape[2] > 1 else m.expand(-1, -1, k.shape[2], -1).clone()
        m[:, :, 0] = 0
        means[name] = m
        stats[f"energy_{name}"] = energy_share(k, m)
    stats["energy_global_pre"] = energy_share(k_pre, mu)
    for b in PLAIN_BITS:
        yield "plain", b, quant(k, layer, "k", b), stats
    for name, m in means.items():
        for b in bits_list:
            yield name, b, m + quant(k.float() - m, layer, "k", b), stats
    for p, (cp, sp) in offsets.items():
        shifted = rope(k.float(), cp, sp)  # the keys as they would sit at positions p..
        for b in bits_list:
            kh = const + quant(shifted - const, layer, "k", b)
            yield f"const@{p}", b, rope(kh, cp, -sp), stats  # turned back: same scores as q and k both shifted


def v_variants(v, layer, bits_list):
    means = {"global": loo_mean(v).expand(-1, -1, v.shape[2], -1).clone()}
    for g in GROUPS:
        gm, idx = group_mean(v, g)
        means[f"local{g}"] = gm.half().float()[:, :, idx]
        if g in Q4_GROUPS:
            means[f"local{g}q4"] = quant(gm, layer, "v", 4)[:, :, idx]
    stats = {}
    for name, m in means.items():
        m[:, :, 0] = 0
        stats[f"energy_{name}"] = energy_share(v, m)
    for b in PLAIN_BITS:
        yield "plain", b, quant(v, layer, "v", b), stats
    for name, m in means.items():
        for b in bits_list:
            yield name, b, m + quant(v.float() - m, layer, "v", b), stats


def run(args):
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from engine.eval_runner.text_chunks import wikitext_chunks

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B, local_files_only=True)
    chunks, chosen = wikitext_chunks(tokenizer, args.chunks, args.seq_len, args.seed)
    config = AutoConfig.from_pretrained(MODEL_8B, local_files_only=True)
    rotary = LlamaRotaryEmbedding(config)
    positions = torch.arange(args.seq_len).unsqueeze(0)
    cos, sin = (t[0].float() for t in rotary(torch.zeros(1, dtype=torch.float32), positions))
    gen = torch.Generator().manual_seed(args.seed)
    offsets = {}
    for p in OFFSETS:
        cp, sp = rotary(torch.zeros(1, dtype=torch.float32), torch.tensor([[p]]))
        offsets[p] = (cp[0].float(), sp[0].float())
    pos = (torch.randperm(args.seq_len - 1, generator=gen)[: args.queries] + 1).sort().values
    result = {"model": MODEL_8B, "chunks": args.chunks, "seq_len": args.seq_len, "seed": args.seed, "chosen": chosen,
              "queries": args.queries, "bits": args.bits, "plain_bits": list(PLAIN_BITS), "groups": list(GROUPS),
              "pairs": list(PAIRS), "offsets": list(OFFSETS), "q4_groups": list(Q4_GROUPS), "layers": []}
    start = time.time()
    for i, k, k_pre, v, q, attn_out in stream_kv(MODEL_8B, chunks, device, layers=args.layers, extras=True):
        att = Attention(q, k, v, pos, device)
        row = {"layer": i, "check_rel_err": att.check(attn_out), "k": {}, "v": {}}
        for name, b, kh, stats in k_variants(k, k_pre, i, cos, sin, args.bits, offsets):
            row["k"].setdefault(name, {})[str(b)] = {"mse": rel_mse(k, kh), **att.with_k(kh)}
            row["k_stats"] = stats
        for name, b, vh, stats in v_variants(v, i, args.bits):
            row["v"].setdefault(name, {})[str(b)] = {"mse": rel_mse(v, vh), **att.with_v(vh)}
            row["v_stats"] = stats
        result["layers"].append(row)
        del att
        if device.type == "mps":
            torch.mps.empty_cache()
        kl = lambda n, b: np.mean(row["k"][n][str(b)]["kl"])  # noqa: E731
        print(f"layer {i:2d}  check {row['check_rel_err']:.2e}  K energy const {np.mean(row['k_stats']['energy_const']):.2f}"
              f" global {np.mean(row['k_stats']['energy_global']):.2f}"
              f" local32 {np.mean(row['k_stats']['energy_local32']):.2f}  KL@2b plain {kl('plain', 2):.4f} "
              f"const {kl('const', 2):.4f} const@129024 {kl('const@129024', 2):.4f} global {kl('global', 2):.4f} local32 {kl('local32', 2):.4f}  [{time.time() - start:.0f} s]", flush=True)
    result["seconds"] = round(time.time() - start, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "mean_removal.json"
    path.write_text(json.dumps(result) + "\n")
    print(f"wrote {path}")
    summary(result)


def storage(name, b):
    extra = 0.0
    if name.startswith("local"):
        g = int(name[5:].removesuffix("q4"))
        extra = (4 * 128 + 16) / (128 * g) if name.endswith("q4") else 16 / g
    return b + 16 / 128 + extra


def equiv_bits(value, curve):
    """Bits of plain TurboQuant with the same error: linear in log(error) between neighbouring plain points."""
    bits, vals = zip(*sorted(curve.items()))
    logs = np.log(vals)
    x = math.log(value)
    for j in range(len(bits) - 1):
        if logs[j] >= x >= logs[j + 1]:
            return bits[j] + (logs[j] - x) / (logs[j] - logs[j + 1]) * (bits[j + 1] - bits[j])
    j = 0 if x > logs[0] else len(bits) - 2  # outside the curve: extend the end segment
    return bits[j] + (logs[j] - x) / (logs[j] - logs[j + 1]) * (bits[j + 1] - bits[j])


def summary(result):
    layers = result["layers"]
    print(f"\ncheck: recomputed vs the layer's own attention output, max rel err "
          f"{max(r['check_rel_err'] for r in layers):.2e}")
    for side, metrics in (("k", ("kl", "out", "mse")), ("v", ("out", "mse"))):
        names = list(layers[0][side])
        mean = {n: {b: {m: float(np.mean([r[side][n][b][m] for r in layers])) for m in metrics}
                    for b in layers[0][side][n]} for n in names}
        print(f"\n{side.upper()}  energy share of the mean: " + ", ".join(
            f"{n} {np.mean([r[side + '_stats']['energy_' + n] for r in layers]):.2f}" for n in names
            if "energy_" + n in layers[0][side + "_stats"]))
        if side == "k":
            print("   K pre-RoPE global (check) "
                  f"{np.mean([r['k_stats']['energy_global_pre'] for r in layers]):.2f};  share of ||mu||^2 in top "
                  "1/2/4/8/16 pairs " + " ".join(f"{np.mean([r['k_stats']['pairs_share'][n] for r in layers]):.2f}"
                                               for n in ("1", "2", "4", "8", "16")))
        for m in metrics:
            curve = {int(b): mean["plain"][b][m] for b in mean["plain"]}
            loo = [abs(equiv_bits(curve[b], {c: v for c, v in curve.items() if c != b}) - b)
                   for b in sorted(curve)[1:-1]]
            print(f"  {m}: plain " + " ".join(f"{b}b {curve[b]:.3g}" for b in sorted(curve)) +
                  f"   (equivalent-bits leave-one-out error, max {max(loo):.2f} bits)")
            for n in names:
                if n == "plain":
                    continue
                cells = []
                for b in result["bits"]:
                    e = equiv_bits(mean[n][str(b)][m], curve)
                    cells.append(f"{b}b {mean[n][str(b)][m]:.3g} = {e:.2f} eq. bits for {storage(n, b):.2f} stored")
                print(f"    {n:11s} " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--queries", type=int, default=256, help="sampled query positions per chunk")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--layers", type=int, default=None, help="first N layers only (quick test)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summary", type=Path, help="print the summary of an existing mean_removal.json")
    args = parser.parse_args()
    if args.summary:
        summary(json.loads(args.summary.read_text()))
    else:
        run(args)


if __name__ == "__main__":
    main()
