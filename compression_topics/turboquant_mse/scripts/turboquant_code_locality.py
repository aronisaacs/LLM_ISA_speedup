#!/usr/bin/env python3
"""How similar are the TurboQuant-mse codes of tokens close to each other? Read-only profiling.

Llama 3.1 8B on --chunks random WikiText-2 test chunks of --seq-len tokens (engine's wikitext_chunks, the same
draw as the other profiles). The model is streamed one decoder layer at a time (HF's own LlamaDecoderLayer, weights
read from the cached safetensors), so it runs in a few GB on a 16 GB Mac (MPS, bfloat16). Nothing is compressed:
every layer sees exact K/V; the codes are what turboquant_mse would store for that K/V.

Views, per layer and KV head:
  k        keys as stored (after RoPE)
  k_pre    keys before RoPE (k_proj output), quantized with the same rotation as k
  v        values
Codes: uniform --bits (default 2 3 4), signs="rope_pairs", Hadamard, Gaussian Lloyd-Max, no outlier split
(turboquant_mse.codes). Position 0 (BOS) is left out of every comparison.

Per (view, bits, layer, KV head):
  match[lag]   share of the d coordinates whose level index is equal at tokens t and t - lag (lag 1, 2, 4 .. 1024)
  match_perm   the same for each token against a random other token of the same chunk (seeded permutation)
  match_chance sum_i p_i^2 averaged over coordinates, p_i the share of tokens at level i (chance agreement from the
               level frequencies alone)
  step         lag 1: share of coordinates whose index moves by 0, 1, 2 .. 2^bits - 1 levels (step_perm: random token)
  groups       aligned groups of 3 and 4 consecutive tokens: how each coordinate's codes split (all agree, one
               differs, ...), how far the odd code is, residual entries against the group's most common code
  planes       lag 1 (and perm): agreement of the sign plane and of each size-bit plane (the FSACC storage form:
               index i < M is sign 1, size M-1-i; i >= M is sign 0, size i-M; M = 2^(bits-1))
Per (view, layer, KV head), independent of bits:
  cos[lag], cos_perm   mean cosine similarity of the exact vectors (the rotation is the same for every token, so this
                       is also the cosine of the rotated vectors before snapping)
  norm_rel             lag 1: |n_t - n_(t-1)| / ((n_t + n_(t-1)) / 2) of the stored fp16 norms (mean, median)

Checks (run them first):
  --check-stream   streaming vs the whole model in memory, on Llama-3.2-1B (same LlamaDecoderLayer path), all layers
  --check-golden   8B layer 0 streamed on the snapshot's first tokens vs shared_core/snapshots/
                   llama31-layer0-real-kv-wikitext.bin (K, V), and codes() vs the golden TurboQuant codes
                   (llama31-layer0-turboquant-{k,v}{2,3,4}.bin), from the snapshot K/V and from the streamed K/V

  python compression_topics/turboquant_mse/scripts/turboquant_code_locality.py --check-stream
  python compression_topics/turboquant_mse/scripts/turboquant_code_locality.py --check-golden
  python compression_topics/turboquant_mse/scripts/turboquant_code_locality.py
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.turboquant_mse.algorithms import turboquant_mse as tq  # noqa: E402

MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_1B = "meta-llama/Llama-3.2-1B-Instruct"
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
GROUPS = (3, 4)  # aligned groups of consecutive tokens (after BOS) for group_stats
SNAPSHOTS = ROOT.parent / "KV_compression_simulations" / "shared_core" / "snapshots"
OUT = ROOT / "compression_topics" / "turboquant_mse" / "code_locality" / "llama31"


# ---------------------------------------------------------------- streaming the model one layer at a time

class _Recorder:
    """Stands in for the cache: keeps the K (after RoPE) and V the attention would store."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self.k, self.v = key_states, value_states
        return key_states, value_states


class _Weights:
    def __init__(self, model_id: str):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        self.dir = Path(hf_hub_download(model_id, "config.json", local_files_only=True)).parent
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.dir / "model.safetensors", "pt") as f:
                self.where = {name: "model.safetensors" for name in f.keys()}
        self._open = safe_open
        self._files: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        file = self.where[name]
        if file not in self._files:
            self._files[file] = self._open(self.dir / file, "pt")
        return self._files[file].get_tensor(name)

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{i}."
        return {name[len(prefix):]: self.get(name) for name in self.where if name.startswith(prefix)}


def stream_kv(model_id: str, chunks: list[torch.Tensor], device: torch.device, dtype=torch.bfloat16,
              layers: int | None = None, extras: bool = False):
    """Yield (layer, k, k_pre, v) per decoder layer, each [chunks, kv_heads, tokens, head_dim] on the CPU.

    With ``extras`` also q (after RoPE, [chunks, heads, tokens, head_dim]) and the attention output before o_proj
    ([chunks, tokens, heads * head_dim]), as the layer computed them.
    """
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import (LlamaDecoderLayer, LlamaRotaryEmbedding,
                                                          apply_rotary_pos_emb)

    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    config._attn_implementation = "sdpa"
    weights = _Weights(model_id)
    embed = weights.get("model.embed_tokens.weight")
    hidden = [embed[ids].unsqueeze(0).to(device=device, dtype=dtype) for ids in chunks]
    del embed
    rotary = LlamaRotaryEmbedding(config, device=device)
    tokens = chunks[0].shape[0]
    positions = torch.arange(tokens, device=device).unsqueeze(0)
    cos_sin = rotary(hidden[0], positions)
    kv_heads, head_dim = config.num_key_value_heads, getattr(config, "head_dim", None) or \
        config.hidden_size // config.num_attention_heads
    for i in range(config.num_hidden_layers if layers is None else layers):
        with torch.device("meta"):
            layer = LlamaDecoderLayer(config, i)
        layer.load_state_dict({k: w.to(dtype) for k, w in weights.layer(i).items()}, assign=True, strict=True)
        layer.to(device).eval()
        pre = {}
        hooks = [layer.self_attn.k_proj.register_forward_hook(lambda m, a, out: pre.__setitem__("k", out))]
        if extras:
            hooks.append(layer.self_attn.q_proj.register_forward_hook(lambda m, a, out: pre.__setitem__("q", out)))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a: pre.__setitem__("o", a[0])))
        ks, kps, vs, qs, outs = [], [], [], [], []
        with torch.inference_mode():
            for c, h in enumerate(hidden):
                rec = _Recorder()
                hidden[c] = layer(h, attention_mask=None, position_ids=positions, past_key_values=rec,
                                  position_embeddings=cos_sin)
                ks.append(rec.k.cpu())
                vs.append(rec.v.cpu())
                kps.append(pre["k"].view(1, tokens, kv_heads, head_dim).transpose(1, 2).cpu())
                if extras:
                    q = pre["q"].view(1, tokens, -1, head_dim).transpose(1, 2)
                    qs.append(apply_rotary_pos_emb(q, q, *cos_sin)[0].cpu())
                    outs.append(pre["o"].cpu())
        for hook in hooks:
            hook.remove()
        del layer
        if device.type == "mps":
            torch.mps.empty_cache()
        if extras:
            yield i, torch.cat(ks), torch.cat(kps), torch.cat(vs), torch.cat(qs), torch.cat(outs)
        else:
            yield i, torch.cat(ks), torch.cat(kps), torch.cat(vs)


def full_kv(model_id: str, chunks: list[torch.Tensor], device: torch.device, dtype=torch.bfloat16):
    """The same K / K before RoPE / V from the whole model in memory (for --check-stream)."""
    from transformers import AutoModelForCausalLM, DynamicCache

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=True,
                                                 attn_implementation="sdpa").to(device).eval()
    cfg = model.config
    out = {i: ([], [], []) for i in range(cfg.num_hidden_layers)}
    for ids in chunks:
        pre = {}
        hooks = [layer.self_attn.k_proj.register_forward_hook(
            lambda m, a, o, i=i: pre.__setitem__(i, o)) for i, layer in enumerate(model.model.layers)]
        cache = DynamicCache(config=cfg)
        with torch.inference_mode():
            model.model(ids.unsqueeze(0).to(device), past_key_values=cache, use_cache=True)
        for h in hooks:
            h.remove()
        for i in range(cfg.num_hidden_layers):
            k, v = cache.layers[i].keys, cache.layers[i].values
            out[i][0].append(k.cpu())
            out[i][1].append(pre[i].view(1, ids.shape[0], k.shape[1], k.shape[3]).transpose(1, 2).cpu())
            out[i][2].append(v.cpu())
    del model
    return {i: tuple(torch.cat(t) for t in triple) for i, triple in out.items()}


# ---------------------------------------------------------------- statistics

def _planes(index: torch.Tensor, bits: int) -> list[torch.Tensor]:
    """Sign plane and size-bit planes (FSACC storage form) of level indices."""
    m = 1 << (bits - 1)
    idx = index.long()
    size = torch.where(idx < m, m - 1 - idx, idx - m)
    return [idx < m] + [((size >> j) & 1).bool() for j in range(bits - 1)]


def _per_head(x: torch.Tensor) -> list[float]:
    """Mean over everything except dim 1 (KV head)."""
    return x.float().transpose(0, 1).reshape(x.shape[1], -1).mean(dim=1).tolist()


def group_stats(index: torch.Tensor, bits: int, size: int) -> dict:
    """Aligned groups of ``size`` consecutive tokens: per coordinate, how the group's codes split.

    pattern  share of (group, coordinate) per split, as sorted counts of equal codes, e.g. "3+1" (one token
             differs from the other three), "2+2", "4" (all agree)
    odd_step for the size-1 + 1 split: share whose odd code is 1, 2, .. levels from the majority code
    residual share of (token, coordinate) entries that differ from the group's most common code (a 2+2 tie
             counts 2 of 4)
    """
    c, h, t, d = index.shape
    n = t // size
    grp = index[:, :, : n * size].reshape(c, h, n, size, d).long()
    counts = torch.stack([(grp == level).sum(3) for level in range(1 << bits)], dim=-1)  # [c, h, n, d, L]
    top = counts.sort(dim=-1, descending=True).values
    patterns = {}
    for key in _partitions(size):
        parts = list(map(int, key.split("+")))
        match = torch.ones(top.shape[:-1], dtype=torch.bool)
        for j, part in enumerate(parts):
            match &= top[..., j] == part
        if len(parts) < top.shape[-1]:
            match &= top[..., len(parts)] == 0
        patterns[key] = _per_head(match)
    majority = counts.argmax(-1)  # [c, h, n, d]
    one_off = top[..., 0] == size - 1
    odd = torch.where(grp != majority.unsqueeze(3), grp, majority.unsqueeze(3)).sub(majority.unsqueeze(3)).abs().amax(3)
    odd_step = [(_per_head(one_off & (odd == s)) if s else 0.0) for s in range(1 << bits)]
    share = one_off.float().transpose(0, 1).reshape(h, -1).mean(1).clamp_min(1e-30)
    odd_step = [[0.0] * h] + [(torch.tensor(odd_step[s]) / share).tolist() for s in range(1, 1 << bits)]
    residual = _per_head((size - top[..., 0]).float() / size)
    return {"pattern": patterns, "odd_step": odd_step, "residual": residual, "groups": c * n}


def _partitions(size: int) -> list[str]:
    def parts(n, most):
        if n == 0:
            yield []
        for k in range(min(n, most), 0, -1):
            for rest in parts(n - k, k):
                yield [k] + rest
    return ["+".join(map(str, p)) for p in parts(size, size)]


def view_stats(x: torch.Tensor, layer_idx: int, target: str, bits_list, perm: torch.Tensor) -> dict:
    """x [chunks, kv_heads, tokens, d] exact vectors (HF order). Tokens from 1 on (BOS left out)."""
    x = x[:, :, 1:].float()
    tokens = x.shape[2]
    unit = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    out = {"cos": {}, "bits": {}}
    for lag in LAGS:
        if lag < tokens:
            out["cos"][str(lag)] = _per_head((unit[:, :, lag:] * unit[:, :, :-lag]).sum(-1))
    out["cos_perm"] = _per_head((unit * unit[:, :, perm]).sum(-1))
    for bits in bits_list:
        index, norm = tq.codes(x, layer_idx=layer_idx, target=target, bits=bits)
        if bits == bits_list[0]:
            n = norm.float()
            rel = (n[:, :, 1:] - n[:, :, :-1]).abs() / ((n[:, :, 1:] + n[:, :, :-1]) / 2).clamp_min(1e-30)
            out["norm_rel_mean"] = _per_head(rel)
            out["norm_rel_median"] = rel.transpose(0, 1).reshape(rel.shape[1], -1).median(dim=1).values.tolist()
        b = {"match": {}}
        for lag in LAGS:
            if lag < tokens:
                b["match"][str(lag)] = _per_head(index[:, :, lag:] == index[:, :, :-lag])
        b["match_perm"] = _per_head(index == index[:, :, perm])
        levels = 1 << bits
        p = torch.stack([(index == level).float().mean(dim=(0, 2)) for level in range(levels)], dim=-1)  # [h, d, L]
        b["match_chance"] = (p * p).sum(-1).mean(-1).tolist()
        b["level_share"] = p.mean(1).tolist()  # [h, L]
        step = (index[:, :, 1:].long() - index[:, :, :-1].long()).abs()
        b["step"] = [_per_head(step == s) for s in range(levels)]  # [s][h]
        step = (index.long() - index[:, :, perm].long()).abs()
        b["step_perm"] = [_per_head(step == s) for s in range(levels)]
        planes, planes_perm = [], []
        mine = _planes(index, bits)
        for plane in mine:
            planes.append(_per_head(plane[:, :, 1:] == plane[:, :, :-1]))
            planes_perm.append(_per_head(plane == plane[:, :, perm]))
        b["planes"], b["planes_perm"] = planes, planes_perm  # plane 0 sign, plane j size bit j-1
        b["groups"] = {str(g): group_stats(index, bits, g) for g in GROUPS}
        out["bits"][str(bits)] = b
    return out


def profile(args) -> None:
    from transformers import AutoTokenizer

    from engine.eval_runner.text_chunks import wikitext_chunks

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B, local_files_only=True)
    chunks, chosen = wikitext_chunks(tokenizer, args.chunks, args.seq_len, args.seed)
    perm = _derangement(args.seq_len - 1, args.seed)
    result = {
        "model": MODEL_8B, "dtype": "bfloat16", "device": args.device, "text": "WikiText-2 raw test",
        "chunks": args.chunks, "seq_len": args.seq_len, "seed": args.seed, "chosen": chosen,
        "bits": args.bits, "lags": list(LAGS), "signs": "rope_pairs", "rotation": "hadamard",
        "codebook": "gaussian", "bos_left_out": True, "layers": [],
    }
    start = time.time()
    for i, k, k_pre, v in stream_kv(MODEL_8B, chunks, device, layers=args.layers):
        row = {"layer": i}
        for name, x, target in (("k", k, "k"), ("k_pre", k_pre, "k"), ("v", v, "v")):
            row[name] = view_stats(x, i, target, args.bits, perm)
        result["layers"].append(row)
        line = "  ".join(
            f"{name} " + "/".join(f"{np.mean(row[name]['bits'][str(b)]['match']['1']):.3f}" for b in args.bits)
            for name in ("k", "k_pre", "v"))
        print(f"layer {i:2d}  lag-1 match ({'/'.join(map(str, args.bits))} bits): {line}  "
              f"[{time.time() - start:.0f} s]", flush=True)
    result["seconds"] = round(time.time() - start, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "profile.json"
    path.write_text(json.dumps(result) + "\n")
    print(f"wrote {path}")
    summary(result)


def _derangement(n: int, seed: int) -> torch.Tensor:
    """A seeded permutation of n positions with no fixed point (a token is never paired with itself)."""
    gen = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(n, generator=gen)
        if not (perm == torch.arange(n)).any():
            return perm


def summary(result: dict) -> None:
    layers = result["layers"]
    print("\nmean over layers and KV heads")
    for name in ("k", "k_pre", "v"):
        cos1 = np.mean([r[name]["cos"]["1"] for r in layers])
        cosp = np.mean([r[name]["cos_perm"] for r in layers])
        print(f"{name:6s} cosine lag 1 {cos1:.3f}, random {cosp:.3f}")
        for b in result["bits"]:
            s = [r[name]["bits"][str(b)] for r in layers]
            m = {lag: np.mean([x["match"][lag] for x in s]) for lag in ("1", "2", "8", "64", "1024")}
            print(f"  {b} bits  match lag 1/2/8/64/1024 " + " ".join(f"{v:.3f}" for v in m.values()) +
                  f"  random {np.mean([x['match_perm'] for x in s]):.3f}"
                  f"  chance {np.mean([x['match_chance'] for x in s]):.3f}"
                  f"  step<=1 {np.mean([np.add(x['step'][0], x['step'][1]) for x in s]):.3f}")


# ---------------------------------------------------------------- checks

def check_stream(args) -> None:
    from transformers import AutoTokenizer

    from engine.eval_runner.text_chunks import wikitext_chunks

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_1B, local_files_only=True)
    chunks, _ = wikitext_chunks(tokenizer, 2, 512, args.seed)
    full = full_kv(MODEL_1B, chunks, device)
    worst = 0.0
    for i, k, k_pre, v in stream_kv(MODEL_1B, chunks, device):
        diffs = [(a.float() - b.float()).abs().max().item() for a, b in zip((k, k_pre, v), full[i])]
        worst = max(worst, *diffs)
        print(f"layer {i:2d}  max |stream - full|  k {diffs[0]:.3g}  k_pre {diffs[1]:.3g}  v {diffs[2]:.3g}")
    print(f"--check-stream: worst {worst:.3g} ({'PASS, identical' if worst == 0 else 'differs, see above'})")


def _real_kv(tokens: int):
    path = SNAPSHOTS / "llama31-layer0-real-kv-wikitext.bin"
    with open(path, "rb") as f:
        if f.read(8) != b"KVREAL02":
            sys.exit(f"{path}: not a KVREAL02 file")
        n, n_embd, n_gqa, head_dim = struct.unpack("<4q", f.read(32))
    ids = np.memmap(path, dtype=np.int32, mode="r", offset=40, shape=(n,))
    k_off = 40 + 4 * n + 2 * n * n_embd
    k = np.memmap(path, dtype=np.float16, mode="r", offset=k_off, shape=(n, n_gqa))
    v = np.memmap(path, dtype=np.float16, mode="r", offset=k_off + 2 * n * n_gqa, shape=(n, n_gqa))

    def heads(a):  # [tokens, gqa] -> [1, kv_heads, tokens, head_dim]
        return torch.from_numpy(np.array(a[:tokens])).view(tokens, -1, head_dim).transpose(0, 1).unsqueeze(0)
    k_hf = heads(k)
    k_hf = torch.cat((k_hf[..., 0::2], k_hf[..., 1::2]), dim=-1)  # ggml pair order -> HF order
    return torch.from_numpy(np.array(ids[:tokens])).long(), k_hf, heads(v)


def _golden_index(target: str, bits: int, tokens: int) -> torch.Tensor:
    path = SNAPSHOTS / f"llama31-layer0-turboquant-{target}{bits}.bin"
    with open(path, "rb") as f:
        if f.read(8) != b"TQGOLD01":
            sys.exit(f"{path}: not a TQGOLD01 file")
        n, heads, level_x10, record, _ = struct.unpack("<5q", f.read(40))
    assert level_x10 == 10 * bits and record == 16 * bits
    planes = np.memmap(path, dtype=np.uint8, mode="r", offset=48, shape=(heads, n, bits, 16))
    b = np.unpackbits(np.array(planes[:, :tokens]), axis=-1, bitorder="little").astype(np.int64)  # [h, t, bits, 128]
    sign, size = b[:, :, 0], sum(b[:, :, j] << (j - 1) for j in range(1, bits))
    m = 1 << (bits - 1)
    return torch.from_numpy(np.where(sign == 1, m - 1 - size, m + size)).unsqueeze(0)  # [1, h, t, 128]


def check_golden(args) -> None:
    tokens = args.seq_len
    ids, k_snap, v_snap = _real_kv(tokens)
    _, k, _, v = next(stream_kv(MODEL_8B, [ids], torch.device(args.device), layers=1))
    for name, mine, snap in (("k", k, k_snap), ("v", v, v_snap)):
        d = (mine.float() - snap.float()).abs()
        rel = d.norm(dim=-1) / snap.float().norm(dim=-1).clamp_min(1e-30)
        print(f"{name}: streamed vs snapshot, max |diff| {d.max():.3g}, median relative L2 {rel.median():.3g}, "
              f"exactly equal {(d == 0).float().mean():.4f}")
    for target, snap, mine in (("k", k_snap, k), ("v", v_snap, v)):
        for bits in (2, 3, 4):
            gold = _golden_index(target, bits, tokens)
            from_snap, _ = tq.codes(snap.float(), layer_idx=0, target=target, bits=bits)
            from_stream, _ = tq.codes(mine.float(), layer_idx=0, target=target, bits=bits)
            print(f"{target}{bits}: codes() on snapshot vs golden {(from_snap.long() == gold).float().mean():.5f}, "
                  f"on streamed K/V vs golden {(from_stream.long() == gold).float().mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--layers", type=int, default=None, help="first N layers only (quick test)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check-stream", action="store_true")
    parser.add_argument("--check-golden", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.check_stream:
        check_stream(args)
    elif args.check_golden:
        check_golden(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()
