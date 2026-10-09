#!/usr/bin/env python3
"""Stage A for ``group_rd``: how much does a per-block mode menu beat one fixed (g, r) per slot,
and how much of that survives a small menu and coarser decisions? Keys and values.

No perplexity runs. ``capture`` makes one prefill pass of Llama 3.1 8B over the
16 WikiText screening chunks that ``group_calibrated_study`` uses (pool of 80,
seed 0, offset 0), saves every layer's keys and values, and averages squared
query coordinates per layer and KV head (summed over the query heads that
share it) into ``query_weights.pt``, the file ``group_rd`` reads for
``distortion='query'``. ``analyze`` needs no model. Per slot (key layers with
distortions cosine and query; value layers with cosine and squared) it traces
distortion against stored bits for:

  full_g<4|16|64>     all 41 modes, one mode per 4 / 16 / 64 tokens of a head
  full_auto           41 modes, residual positions as 7-bit indices where that is
                      smaller than the 128-bit mask (r <= 16); still one size per mode
  default_g<4|16|64>  group_rd.DEFAULT_MENU (8 modes, 3-bit code)
  default_auto        DEFAULT_MENU with auto masks
  top<N>              the N modes (D included) the full menu uses most, one menu for
                      every layer, chosen at --menu-saving (--top, default 6 and 8)
  pairs_r<r>          dense or pair with r residual entries (group_calibrated, g = 2)
  quads_r<r>          dense or quad with r residual entries (group_calibrated, g = 4)

Every mode has one fixed size; there are no per-group thresholds. One price
per layer is shared by all chunks, so a curve is the layer's operating curve;
savings are interpolated on its lower convex envelope. Bits include mode codes
(byte-packed, per decision unit) and the slot header. best_fixed is the best
pairs_r / quads_r curve at each saving, i.e. the per-slot choice
``group_calibrated_study`` makes, scored by the same proxy. Pages are 64
tokens of one head: the summary counts distinct modes per page.

  python compression_topics/spatial/scripts/group_rd_offline.py              # plan only
  python compression_topics/spatial/scripts/group_rd_offline.py --execute --stage capture
  python compression_topics/spatial/scripts/group_rd_offline.py --execute --stage analyze
  python compression_topics/spatial/scripts/group_rd_offline.py --execute --stage menus

``menus`` is a fast re-check on the captures (vectorized over prices; minutes):
candidate 8-format menus at one format per 64-token page, keys with
query-weighted error and values with squared error, against all 41 formats
chosen per 4 tokens and against the best fixed format per layer. Candidates
come from ``CANDIDATE_MENUS`` or ``--menus`` (JSON: name -> list of formats).
Flagged pair formats such as ``P32/d`` let each block pick which pair it merges.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.models import LLAMA31_8B
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.algorithms.storage import metadata_bits
from engine.eval_runner.files import write_json
from engine.kv_compress.rope import RopeTables

OUT = ROOT / "compression_topics/spatial/figures/group_rd_offline"
DISTORTIONS = {"k": ("cosine", "query"), "v": ("cosine", "squared")}
GRANULARITIES = (4, 16, 64)
TARGETS = tuple(round(0.05 * i, 2) for i in range(1, 16))  # 5% .. 75%
PAGE = 64
PAGE_SAVINGS = (.3, .5, .7)
CANDIDATE_MENUS = {
    "default": list(group_rd.DEFAULT_MENU),
    "data_top8": ["D", "Q16", "Q32", "Q0", "Q8", "P32+32", "P32+0", "P32+16"],
    "flag_a": ["D", "P32/d", "P32+32", "P32/0", "Q32", "Q16", "Q8", "Q0"],
    "flag_b": ["D", "P32/d", "P32+32", "P32/0", "P0+0", "Q32", "Q16", "Q0"],
    "flag_c": ["D", "P32/d", "P16/d", "P32+32", "Q32", "Q16", "Q8", "Q0"],
    "flag_d": ["D", "P32/d", "P32+32", "P16/0", "Q32", "Q16", "Q8", "Q0"],
}
MENU_SLOTS = (("k", "query"), ("v", "squared"))
MENU_SAVINGS = (.3, .35, .4, .45, .5, .55, .6, .65, .7)
LAMBDA_SPAN = (-4, 4, 129)  # decades around the layer's typical price, and points


def plan(out, *, chunks=16, pool=80, seed=0, seq_len=2048, model_args=LLAMA31_8B,
         residuals=group_rd.RESIDUALS, tops=(6, 8), menu_saving=.5, keys_dir=None):
    keys_dir = Path(keys_dir or out / "keys")
    return {"stage": "A", "model_args": model_args, "chunks": chunks, "seq_len": seq_len,
            "sampling": {"pool_chunks": pool, "offset": 0, "chunks": chunks, "seed": seed, "seq_len": seq_len},
            "residuals": list(residuals), "modes": group_rd.mode_names(residuals),
            "default_menu": list(group_rd.DEFAULT_MENU),
            "configs": config_names(residuals, tops), "granularities": list(GRANULARITIES),
            "distortions": {target: list(kinds) for target, kinds in DISTORTIONS.items()}, "targets": list(TARGETS), "page_tokens": PAGE,
            "top_menus": {"sizes": list(tops), "chosen_at_saving": menu_saving},
            "keys_dir": str(keys_dir), "query_weights": str(out / "query_weights.pt"),
            "rope": str(out / "rope.json"), "accounting": group_rd.ACCOUNTING,
            "keys_bytes_estimate": 2 * chunks * 32 * 8 * seq_len * 128 * 2}  # keys and values


def config_names(residuals, tops):
    names = [f"{menu}_g{g}" for menu in ("full", "default") for g in GRANULARITIES]
    names += ["full_auto", "default_auto"] + [f"top{n}" for n in tops]
    return names + [f"pairs_r{r}" for r in residuals] + [f"quads_r{r}" for r in residuals]


def parse_model_args(text):
    fields = dict(item.split("=", 1) for item in text.split(","))
    return fields["pretrained"], getattr(torch, fields.get("dtype", "bfloat16"))


def capture(manifest, device=None):
    """One prefill per chunk: keys and values to disk, query second moments to query_weights.pt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache
    import transformers.models.llama.modeling_llama as llama
    from compression_topics.spatial.scripts.pair_similarity_profile import rope_tables
    from engine.eval_runner.chunks import partition
    from engine.eval_runner.text_chunks import wikitext_chunks

    name, dtype = parse_model_args(manifest["model_args"])
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    sampling = manifest["sampling"]
    prompts, chosen = wikitext_chunks(tokenizer, sampling["pool_chunks"], sampling["seq_len"], sampling["seed"])
    prompts, chosen = partition(prompts, chosen, sampling)
    rope = rope_tables(model)
    keys_dir = Path(manifest["keys_dir"])
    keys_dir.mkdir(parents=True, exist_ok=True)
    Path(manifest["rope"]).write_text(json.dumps({"rope_theta": rope.rope_theta, "head_dim": rope.head_dim,
        "inv_freq": list(rope.inv_freq) if rope.inv_freq else None, "attention_scaling": rope.attention_scaling}))
    layers, values, pending, sums, counts = {}, {}, {}, {}, {}
    original_update, original_rotary = Cache.update, llama.apply_rotary_pos_emb

    def rotary(query, key, *args, **kwargs):
        query_embed, key_embed = original_rotary(query, key, *args, **kwargs)
        pending["query"] = query_embed
        return query_embed, key_embed

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        layers[layer_idx] = key_states.detach().to("cpu", torch.bfloat16)[0]
        values[layer_idx] = value_states.detach().to("cpu", torch.bfloat16)[0]
        query = pending.pop("query").float()  # [1, Hq, T, D]
        kv_heads = key_states.shape[1]
        grouped = query[0].reshape(kv_heads, -1, *query.shape[-2:])
        sums[layer_idx] = sums.get(layer_idx, 0) + grouped.square().sum((1, 2)).cpu()
        counts[layer_idx] = counts.get(layer_idx, 0) + query.shape[-2]
        return original_update(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update, llama.apply_rotary_pos_emb = update, rotary
    try:
        for number, (ids, chunk) in enumerate(zip(prompts, chosen)):
            batch = ids.unsqueeze(0).to(device)
            with torch.no_grad():
                model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=True)
            keys = torch.stack([layers[i] for i in sorted(layers)])  # [L, Hkv, T, D]
            stored = torch.stack([values[i] for i in sorted(values)])
            torch.save({"keys": keys, "values": stored, "chunk": chunk}, keys_dir / f"chunk_{chunk:05d}.pt")
            layers.clear()
            values.clear()
            print(f"[capture {number + 1}/{len(prompts)}] chunk {chunk}: keys {tuple(keys.shape)}", flush=True)
    finally:
        Cache.update, llama.apply_rotary_pos_emb = original_update, original_rotary
    weights = torch.stack([sums[i] / counts[i] for i in sorted(sums)])  # [L, Hkv, D]
    torch.save({"weights": weights, "chunks": chosen, "model": name,
                "definition": "mean over positions of squared post-RoPE query, summed over each KV group"},
               manifest["query_weights"])


def load_rope(path):
    data = json.loads(Path(path).read_text())
    return RopeTables(rope_theta=data["rope_theta"], head_dim=data["head_dim"],
                      inv_freq=tuple(data["inv_freq"]) if data["inv_freq"] else None,
                      attention_scaling=data["attention_scaling"])


def restrict(names, keep):
    """Column indices of ``keep`` inside ``names``, in menu order."""
    keep = set(keep)
    return [i for i, name in enumerate(names) if name in keep]


def config_columns(config, names, residuals, top_menus):
    """(mode columns, granularity, mask) of a configuration over the full menu."""
    family, _, rest = config.partition("_")
    if family in ("full", "default"):
        columns = list(range(len(names))) if family == "full" else restrict(names, group_rd.DEFAULT_MENU)
        return (columns, BLOCK_TOKENS, "auto") if rest == "auto" else (columns, int(rest[1:]), "bitmap")
    if family.startswith("top"):
        return restrict(names, top_menus[int(family[3:])]), BLOCK_TOKENS, "bitmap"
    r = int(rest[1:])
    return restrict(names, group_rd.mode_names((r,), family)), BLOCK_TOKENS, "bitmap"


BLOCK_TOKENS = group_rd.BLOCK


def curve(tables, bits, lams, granularity, overhead):
    """Total distortion and bits per price over chunks; ``tables`` are [1, H, blocks, modes]."""
    distortion = torch.zeros(len(lams), dtype=torch.float64)
    stored = torch.zeros(len(lams), dtype=torch.float64)
    for table in tables:
        menu = _as_menu(table, bits)
        for index, lam in enumerate(lams):
            choice = group_rd.select(menu, lam, granularity)
            distortion[index] += menu.distortion.gather(-1, choice.unsqueeze(-1)).sum()
            stored[index] += bits[choice].sum() + overhead
    return distortion, stored


def envelope(saving, distortion, targets):
    """Distortion at each target saving on the lower convex envelope; None beyond reach."""
    points = sorted(set(zip(saving, distortion)))
    hull = []
    for point in points:  # lower hull of (saving, distortion), increasing saving
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    values = []
    for target in targets:
        if not hull or target < hull[0][0] - 1e-12 or target > hull[-1][0] + 1e-12:
            values.append(None)
            continue
        for (s0, d0), (s1, d1) in zip(hull, hull[1:] + [hull[-1]]):
            if s0 - 1e-12 <= target <= s1 + 1e-12:
                values.append(d0 if s1 == s0 else d0 + (d1 - d0) * (target - s0) / (s1 - s0))
                break
    return values


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def saving_at(saving, distortion, level):
    """Largest saving whose envelope distortion is at most ``level``."""
    best = None
    grid = [i / 1000 for i in range(1001)]
    for target, value in zip(grid, envelope(saving, distortion, grid)):
        if value is not None and value <= level + 1e-12:
            best = target
    return best


def distinct_per_page(choice, page_blocks=PAGE // group_rd.BLOCK):
    """Distinct modes in each page of ``page_blocks`` blocks of one head; choice is [1, H, blocks]."""
    blocks = choice.shape[-1] // page_blocks * page_blocks
    pages = choice[..., :blocks].reshape(-1, page_blocks)
    return [len(set(row.tolist())) for row in pages]


class _as_menu:
    """Minimal Menu stand-in for ``group_rd.select`` on a stored distortion table."""
    def __init__(self, table, bits):
        self.distortion, self.bits = table.double(), bits


def _lambdas(tables, bits, quad):
    """Price grid around the distortion per bit of merging a block as the cheapest quad."""
    finite = torch.cat([t[..., quad].flatten() for t in tables]).double()
    finite = finite[torch.isfinite(finite)]
    typical = float(finite.mean()) / float(bits[0]) if finite.numel() else 1e-6
    low, high, count = LAMBDA_SPAN
    return [0.] + [max(typical, 1e-30) * 10 ** (low + (high - low) * i / (count - 1)) for i in range(count)] + [None]


def _price_for(lams, stored, dense, target):
    """Smallest grid price whose bits reach ``target`` saving (the minimum-bits end if none)."""
    for lam, bits in zip(lams, stored.tolist()):
        if 1 - bits / dense >= target:
            return lam
    return None


def analyze(manifest, device="cpu"):
    out = Path(manifest["query_weights"]).parent
    rope = load_rope(manifest["rope"])
    weights = torch.load(manifest["query_weights"], weights_only=True)["weights"]
    residuals = tuple(manifest["residuals"])
    names = group_rd.mode_names(residuals)
    files = sorted(Path(manifest["keys_dir"]).glob("chunk_*.pt"))
    if not files:
        raise SystemExit("no captured keys; run --stage capture first")
    # (target, distortion) -> layer -> [chunk tables]
    tables = {(target, kind): {} for target, kinds in DISTORTIONS.items() for kind in kinds}
    dense = 0
    for path in files:
        captured = torch.load(path, weights_only=True)
        layers, heads, length, dim = captured["keys"].shape
        full = length // group_rd.BLOCK * group_rd.BLOCK
        dense += heads * length * dim * 16  # one slot, all chunks
        for (target, kind), by_layer in tables.items():
            source = captured["keys" if target == "k" else "values"]
            for layer in range(layers):
                body = source[layer, :, :full].unsqueeze(0).to(device).float()
                menu = group_rd.build_menu(body, rope_tables=rope if target == "k" else None,
                                           residuals=residuals, menu=None, distortion=kind,
                                           weights=weights[layer] if kind == "query" else None)
                by_layer.setdefault(layer, []).append(menu.distortion.float().cpu())
        print(f"[analyze] menus for {path.name}", flush=True)
    tail = (length % group_rd.BLOCK) * heads * dim * 16
    blocks = full // group_rd.BLOCK
    targets = list(manifest["targets"])
    full_bits = group_rd.bit_table(names, residuals, dim, "bitmap")
    full_overhead = metadata_bits(heads * blocks, group_rd.mode_bits(len(names))) + tail
    lams, usage = {}, {target: Counter() for target in DISTORTIONS}
    # Data-driven small menus per target: the modes the full menu uses most at the menu saving.
    for (target, kind), by_layer in tables.items():
        for layer, chunk_tables in by_layer.items():
            lams[target, kind, layer] = _lambdas(chunk_tables, full_bits, names.index(f"Q{residuals[0]}"))
            _, stored = curve(chunk_tables, full_bits, lams[target, kind, layer], 4, full_overhead)
            lam = _price_for(lams[target, kind, layer], stored, dense, manifest["top_menus"]["chosen_at_saving"])
            for table in chunk_tables:
                usage[target].update(names[i] for i in group_rd.select(_as_menu(table, full_bits), lam).flatten().tolist())
    top_menus = {}
    for target, counts in usage.items():
        ranked = [name for name, _ in counts.most_common() if name != "D"]
        ranked += [name for name in names if name != "D" and name not in ranked]  # unused modes last
        top_menus[target] = {size: ["D"] + ranked[:size - 1] for size in manifest["top_menus"]["sizes"]}
    report = {"targets": targets, "default_menu": list(group_rd.DEFAULT_MENU),
              "top_menus": {target: {str(k): v for k, v in menus.items()} for target, menus in top_menus.items()},
              "mode_usage": {target: dict(counts.most_common()) for target, counts in usage.items()},
              "layers": {}, "pages": {}}
    for (target, kind), by_layer in tables.items():
        label = f"{target}/{kind}"
        for layer, chunk_tables in by_layer.items():
            prices = lams[target, kind, layer]
            row, curves = {}, {}
            for config in manifest["configs"]:
                columns, granularity, mask = config_columns(config, names, residuals, top_menus[target])
                subset = [names[i] for i in columns]
                bits = group_rd.bit_table(subset, residuals, dim, mask)
                units = heads * math.ceil(blocks / (granularity // group_rd.BLOCK))
                overhead = metadata_bits(units, group_rd.mode_bits(len(subset))) + tail
                d, b = curve([t[..., columns] for t in chunk_tables], bits, prices, granularity, overhead)
                curves[config] = ((1 - b / dense).tolist(), d.tolist())
                row[config] = {"distortion": envelope(*curves[config], targets), "max_saving": max(curves[config][0])}
            fixed = [c for c in row if c.startswith(("pairs_", "quads_"))]
            best = [min(((row[c]["distortion"][i], c) for c in fixed if row[c]["distortion"][i] is not None),
                        default=(None, None)) for i in range(len(targets))]
            row["best_fixed"] = {"distortion": [b[0] for b in best], "config": [b[1] for b in best]}
            # Saving given up by coarser decisions at the distortion per-4 decisions reach.
            for family in ("full", "default"):
                for granularity in GRANULARITIES[1:]:
                    row[f"{family}_g{granularity}"]["saving_lost_vs_g4"] = [
                        None if level is None or (reach := saving_at(*curves[f"{family}_g{granularity}"], level)) is None
                        else target_saving - reach
                        for target_saving, level in zip(targets, row[f"{family}_g4"]["distortion"])]
            report["layers"].setdefault(label, {})[str(layer)] = row
            # Distinct modes per 64-token page for the default menu, per 4 tokens.
            columns = restrict(names, group_rd.DEFAULT_MENU)
            bits = group_rd.bit_table([names[i] for i in columns], residuals, dim, "bitmap")
            overhead = metadata_bits(heads * blocks, group_rd.mode_bits(len(columns))) + tail
            restricted = [t[..., columns] for t in chunk_tables]
            _, stored = curve(restricted, bits, prices, 4, overhead)
            for saving in PAGE_SAVINGS:
                lam = _price_for(prices, stored, dense, saving)
                counts = Counter()
                for table in restricted:
                    counts.update(distinct_per_page(group_rd.select(_as_menu(table, bits), lam)))
                report["pages"].setdefault(label, {}).setdefault(str(saving), {})[str(layer)] = dict(sorted(counts.items()))
    write_json(out / "stage_a.json", report)
    (out / "stage_a.md").write_text(summary(report))
    print(summary(report))


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _gain(layers, i, config, reference):
    """Median over layers of 1 - D(config) / D(reference) at target ``i``."""
    gains = []
    for row in layers.values():
        a, b = row[config]["distortion"][i], row[reference]["distortion"][i]
        if a is not None and b:
            gains.append(1 - a / b)
    return _median(gains)


def summary(report):
    targets = report["targets"]
    tops = sorted({int(size) for menus in report["top_menus"].values() for size in menus})
    columns = [("41 modes vs best fixed", "full_g4", "best_fixed"),
               ("8-mode default vs best fixed", "default_g4", "best_fixed"),
               ("8-mode default vs 41", "default_g4", "full_g4")]
    columns += [(f"top {n} vs 41", f"top{n}", "full_g4") for n in tops]
    columns += [("auto masks (default menu)", "default_auto", "default_g4")]
    lines = ["# group_rd stage A", "",
             "Gain = median over layers of 1 - D(config) / D(reference) at equal stored bits",
             "(positive: config has less distortion). Lost saving = saving a coarser decision",
             "gives up at the distortion per-4-token decisions reach, default menu.", "",
             f"Default menu: {', '.join(report['default_menu'])}"]
    lines += [f"Top {size} menu ({target}): {', '.join(menu)}" for target, menus in report["top_menus"].items()
              for size, menu in sorted(menus.items(), key=lambda x: int(x[0]))]
    lines.append("")
    for kind, layers in report["layers"].items():
        header = ["saving"] + [c[0] for c in columns] + ["g16 lost", "g64 lost"]
        lines += [f"## {'keys' if kind.startswith('k/') else 'values'}, distortion {kind.split('/')[1]}", "", "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        for i, target in enumerate(targets):
            cells = [_fmt(_gain(layers, i, config, reference)) for _, config, reference in columns]
            for granularity in (16, 64):
                cells.append(_fmt(_median([row[f"default_g{granularity}"]["saving_lost_vs_g4"][i]
                                           for row in layers.values()]), points=True))
            lines.append(f"| {target:.0%} | " + " | ".join(cells) + " |")
        lines += ["", "Distinct modes per 64-token page (default menu, all layers, heads, chunks):", ""]
        for target, by_layer in report["pages"][kind].items():
            total = Counter()
            for counts in by_layer.values():
                total.update({int(k): v for k, v in counts.items()})
            pages = sum(total.values())
            mean = sum(k * v for k, v in total.items()) / pages if pages else 0
            lines.append(f"- saving {float(target):.0%}: mean {mean:.2f}; "
                         + ", ".join(f"{k} modes {v / pages:.1%}" for k, v in sorted(total.items())))
        lines.append("")
    return "\n".join(lines)


def _fmt(value, points=False):
    if value is None:
        return "-"
    return f"{value * 100:+.1f} pts" if points else f"{value:+.1%}"


def menu_columns(menu, names):
    """Columns of ``menu`` as a function of the full ordered table (flagged = better orientation)."""
    def column(table, name):
        if "/" not in name:
            return table[..., names.index(name)]
        a, b = name[1:].split("/")
        return torch.minimum(table[..., names.index(f"P{a}+{b}")], table[..., names.index(f"P{b}+{a}")])
    return lambda table: torch.stack([column(table, name) for name in group_rd.mode_names(menu=menu)], -1)


def fast_curve(table, bits, lams, granularity, overhead, batch=16):
    """Total distortion and bits per price; ``table`` is [chunks, H, blocks, modes] on any device."""
    per = granularity // group_rd.BLOCK
    chunks, heads, blocks, modes = table.shape
    units = table.reshape(chunks, heads, blocks // per, per, modes).sum(-2)
    finite = torch.isfinite(units)
    unit_bits = bits.to(table) * per
    distortion, stored = [], []
    for start in range(0, len(lams), batch):
        group = lams[start:start + batch]
        rows = []
        for lam in group:
            rows.append(unit_bits.expand_as(units) + torch.where(finite, 0., float("inf")) if lam is None
                        else units + lam * unit_bits)
        cost = torch.stack(rows)
        choice = cost.argmin(-1)
        d = units.unsqueeze(0).expand_as(cost).gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        distortion += d.flatten(1).sum(-1).tolist()
        stored += (unit_bits[choice].flatten(1).sum(-1) + overhead * chunks).tolist()
    return distortion, stored


def menu_check(manifest, device="cpu", menus=None, savings=MENU_SAVINGS):
    """Candidate menus at 64-token pages vs 41 formats per block and the best fixed format per layer."""
    menus = menus or CANDIDATE_MENUS
    out = Path(manifest["query_weights"]).parent
    rope = load_rope(manifest["rope"])
    weights = torch.load(manifest["query_weights"], weights_only=True)["weights"]
    residuals = tuple(manifest["residuals"])
    names = group_rd.mode_names(residuals)
    files = sorted(Path(manifest["keys_dir"]).glob("chunk_*.pt"))
    if not files:
        raise SystemExit("no captured keys; run --stage capture first")
    tables = {slot: {} for slot in MENU_SLOTS}
    for path in files:
        captured = torch.load(path, weights_only=True)
        layers, heads, length, dim = captured["keys"].shape
        full = length // 64 * 64
        for target, kind in MENU_SLOTS:
            source = captured["keys" if target == "k" else "values"]
            for layer in range(layers):
                body = source[layer, :, :full].unsqueeze(0).to(device).float()
                menu = group_rd.build_menu(body, rope_tables=rope if target == "k" else None, residuals=residuals,
                                           menu=None, distortion=kind, weights=weights[layer] if kind == "query" else None)
                tables[target, kind].setdefault(layer, []).append(menu.distortion[0].float())
        print(f"[menus] built {path.name}", flush=True)
    blocks = full // group_rd.BLOCK
    dense = len(files) * heads * blocks * group_rd.BLOCK * dim * 16
    configs = {f"menu:{name}": (menu, 64) for name, menu in menus.items()}
    configs["full41_g4"] = (names, 4)
    configs["full41_g64"] = (names, 64)
    for r in residuals:
        configs[f"fixed_pairs_r{r}"] = (group_rd.mode_names((r,), "pairs"), 4)
        configs[f"fixed_quads_r{r}"] = (group_rd.mode_names((r,), "quads"), 4)
    report = {"savings": list(savings), "menus": menus, "slots": {}}
    for (target, kind), by_layer in tables.items():
        label = f"{target}/{kind}"
        rows = {}
        for layer, chunk_tables in by_layer.items():
            table = torch.stack(chunk_tables)  # [chunks, H, blocks, 41]
            full_bits = group_rd.bit_table(names, residuals, dim)
            lams = _lambdas([table.cpu()], full_bits, names.index(f"Q{residuals[0]}"))
            row = {}
            for config, (menu, granularity) in configs.items():
                menu_names = group_rd.mode_names(residuals, menu=menu) if config.startswith("menu:") else menu
                sub = menu_columns(menu_names, names)(table) if config.startswith("menu:") else \
                    table[..., [names.index(n) for n in menu_names]]
                bits = group_rd.bit_table(menu_names, residuals, dim)
                overhead = metadata_bits(heads * blocks * group_rd.BLOCK // granularity, group_rd.mode_bits(len(menu_names)))
                d, b = fast_curve(sub, bits, lams, granularity, overhead)
                row[config] = envelope([1 - x / dense for x in b], d, savings)
            row["best_fixed"] = [min((row[c][i] for c in row if c.startswith("fixed_") and row[c][i] is not None),
                                     default=None) for i in range(len(savings))]
            rows[str(layer)] = {c: v for c, v in row.items() if not c.startswith("fixed_")}
        report["slots"][label] = rows
        print(f"[menus] evaluated {label}", flush=True)
    write_json(out / "menus.json", report)
    text = menu_summary(report)
    (out / "menus.md").write_text(text)
    print(text)


def menu_summary(report):
    savings = report["savings"]
    lines = ["# group_rd menu re-check (one format per 64-token page)", "",
             "Error reduction at equal stored bits, median over layers (positive = less error).", ""]
    lines += [f"- `{name}`: {', '.join(menu)}" for name, menu in report["menus"].items()]
    for label, rows in report["slots"].items():
        configs = [c for c in next(iter(rows.values())) if c.startswith("menu:")] + ["full41_g64", "full41_g4"]
        for reference, title in (("best_fixed", "vs the best fixed format per layer"),
                                 ("full41_g4", "vs all 41 formats chosen per 4 tokens")):
            shown = [c for c in configs if c != reference]
            lines += ["", f"## {'keys' if label.startswith('k') else 'values'} ({label.split('/')[1]} error), {title}", "",
                      "| saving | " + " | ".join(c.replace("menu:", "") for c in shown) + " |",
                      "|---|" + "---|" * len(shown)]
            for i, saving in enumerate(savings):
                cells = []
                for c in shown:
                    values = [1 - row[c][i] / row[reference][i] for row in rows.values()
                              if row[c][i] is not None and row[reference][i]]
                    cells.append(f"{_median(values):+.1%}" if values else "–")
                lines.append(f"| {saving:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--keys-dir", type=Path, default=None, help="captured keys and values (about 4 GB); default <out>/keys")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--pool-chunks", type=int, default=80, help="pool group_calibrated_study samples from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--model-args", default=LLAMA31_8B)
    parser.add_argument("--top", type=int, nargs="+", default=[6, 8], help="sizes of data-driven menus, D included")
    parser.add_argument("--menu-saving", type=float, default=.5, help="saving at which mode usage picks those menus")
    parser.add_argument("--stage", choices=("capture", "analyze", "all", "menus"), default="all")
    parser.add_argument("--menus", type=json.loads, default=None, help="JSON object: menu name -> list of formats")
    parser.add_argument("--device", default=None)
    parser.add_argument("--execute", action="store_true", help="run the stage; otherwise only write the plan")
    args = parser.parse_args()
    if not 1 <= args.chunks <= args.pool_chunks or min(args.top) < 2 or not 0 < args.menu_saving < 1:
        parser.error("need 1 <= chunks <= pool-chunks, top sizes >= 2 and 0 < menu-saving < 1")
    out = args.out.resolve()
    manifest = plan(out, chunks=args.chunks, pool=args.pool_chunks, seed=args.seed, seq_len=args.seq_len,
                    model_args=args.model_args, tops=tuple(sorted(set(args.top))), menu_saving=args.menu_saving,
                    keys_dir=args.keys_dir)
    write_json(out / "plan.json", manifest)
    print(f"Plan: capture keys of {args.chunks} WikiText chunks ({args.seq_len} tokens; pool {args.pool_chunks}, "
          f"seed {args.seed}, the group_calibrated screening chunks) and query second moments.")
    print(f"Default menu ({len(group_rd.DEFAULT_MENU)} modes): {', '.join(group_rd.DEFAULT_MENU)}; "
          f"full menu {len(manifest['modes'])} modes.")
    print(f"Configs: {', '.join(manifest['configs'])}; distortions: keys {', '.join(DISTORTIONS['k'])}, "
          f"values {', '.join(DISTORTIONS['v'])}; "
          f"savings {TARGETS[0]:.0%}..{TARGETS[-1]:.0%}.")
    print(f"Keys: {manifest['keys_dir']} (~{manifest['keys_bytes_estimate'] / 1e9:.1f} GB). Output: {out}")
    if not args.execute:
        print("Plan only. Add --execute (with --stage capture | analyze | all | menus) when ready to run.")
        return
    if args.stage in ("capture", "all"):
        capture(manifest, args.device)
    if args.stage == "menus":
        menu_check(manifest, args.device or "cpu", args.menus)
        return
    if args.stage in ("analyze", "all"):
        analyze(manifest, args.device or "cpu")


if __name__ == "__main__":
    main()
