#!/usr/bin/env python3
"""Small reconstruction/bytes/controller pilot on existing dense 8B KV captures.

No model inference, accuracy evaluation, query weighting, attention importance,
or quantization. Both separate and shared quad/eight masks are tested online.
Equal per-slot targets are a diagnostic,
not a replacement for independently calibrated layer allocation in a full run.
"""
from dataclasses import replace
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms import hierarchy
from engine.eval_runner.files import write_json
from engine.eval_runner.timing import TimingRecord
from engine.kv_compress.rope import RopeTables

CAPTURES = ROOT / 'compression_topics/spatial/figures/group_rd_offline'
OUT = ROOT / 'compression_topics/spatial/figures/hierarchy_per_head_pilot'
VARIANTS = ((2, 'separate', 'pairs'), (4, 'separate', 'pairs_quads_separate'),
            (4, 'shared', 'pairs_quads_shared'), (8, 'separate', 'pairs_quads_eights_separate'),
            (8, 'shared', 'pairs_quads_eights_shared'))


def measure(tensor, *, target, rope, budgets, progress=None, format_progress=None):
    tables = {layout: hierarchy.build_table(tensor, head_decisions='per_head', target=target, rope_tables=rope, max_group=8, mask_layout=layout)
              for layout in ('separate', 'shared')}
    if format_progress:
        # Same residual count: isolate mask flexibility from the controller's
        # changed merge decisions at a matched storage budget.
        for size in (4,8):
            separate, shared = tables['separate'], tables['shared']
            paired = [(a[(size,start)][1], b[(size,start)][1])
                      for aa,bb in zip(separate.rows,shared.rows) for a,b in zip(aa,bb) for start in range(0,8,size)]
            finite = [(a,b) for a,b in paired if math.isfinite(a) and math.isfinite(b)]
            means = [sum(p[i] for p in finite)/len(finite)*8/size for i in (0,1)] if finite else [None,None]
            format_progress(dict(group_size=size,residual_entries=16,valid_groups=len(finite),total_groups=len(paired),
                separate_relative_error=means[0],shared_relative_error=means[1],
                separate_payload_bytes=separate.payload[(size,16)]/8,
                shared_payload_bytes=shared.payload[(size,16)]/8,
                saved_mask_bytes=(separate.payload[(size,16)]-shared.payload[(size,16)])/8,
                scope='same-format comparison across KV heads; payload excludes shared tags and slot header'))
    rows = []
    for size, layout, variant in VARIANTS:
        subset = hierarchy.HeadWindowTable([replace(t,max_group=size) for t in tables[layout].tables])
        for budget in budgets:
            started = time.monotonic()
            try:
                result = hierarchy.run(subset, budget)
            except ValueError as error:
                if 'cannot be met' not in str(error):
                    raise
                row = dict(variant=variant, mask_layout=layout, target_saving=budget, status='unreachable', reason=str(error))
            else:
                history = result.pop('history')
                row = dict(variant=variant, status='complete', **result,
                           checkpoints=[r for r in history if r['tokens'] in (32,64,128,256,512,1024,1536)],
                           post_startup_max_deviation=max((abs(r['cumulative_saving']-budget) for r in history[4:]), default=None),
                           within_three_points=abs(result['measured_saving']-budget)<=.03)
            row['selection_seconds'] = time.monotonic()-started
            rows.append(row)
            if progress:
                progress(row)
    return rows


def comparisons(rows):
    grouped = {}
    for row in rows:
        if row['status'] != 'complete':
            continue
        key = (row['chunk'], row['layer'], row['target'], row['tokens'], row['target_saving'])
        grouped.setdefault(key, {})[row['variant']] = row
    output = []
    for key, variants in grouped.items():
        for left, right in (('pairs','pairs_quads_separate'), ('pairs','pairs_quads_shared'),
                            ('pairs_quads_separate','pairs_quads_eights_separate'),
                            ('pairs_quads_shared','pairs_quads_eights_shared'),
                            ('pairs_quads_separate','pairs_quads_shared'),
                            ('pairs_quads_eights_separate','pairs_quads_eights_shared')):
            if left not in variants or right not in variants:
                continue
            a, b = variants[left], variants[right]
            gap = b['measured_saving']-a['measured_saving']
            output.append(dict(chunk=key[0], layer=key[1], target=key[2], tokens=key[3], target_saving=key[4],
                reference=left, comparison=right, saving_gap_points=100*gap,
                matched_within_one_point=abs(gap)<=.01,
                relative_error_change=(b['mean_relative_squared_error']/a['mean_relative_squared_error']-1)
                                      if a['mean_relative_squared_error'] else None))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=CAPTURES)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--layers', type=int, nargs='+', default=[0,8,16,24,31])
    parser.add_argument('--prefixes', type=int, nargs='+', default=[32,128,512,1536])
    parser.add_argument('--budgets', type=float, nargs='+', default=[.3,.4,.5,.6,.7,.8])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args()
    if args.chunks < 1 or any(n < 32 or n % 8 for n in args.prefixes) or any(not 0 < b < 1 for b in args.budgets):
        parser.error('need positive chunks, complete >=32-token prefixes and budgets in (0,1)')
    if not args.layers or any(n < 0 or n >= 32 for n in args.layers):
        parser.error('layer indices must lie in 0..31')
    files = sorted((args.captures/'keys').glob('chunk_*.pt'))[:args.chunks]
    if len(files) < args.chunks:
        parser.error('existing dense captures missing; this pilot will not launch a full model capture automatically')
    manifest = json.loads((args.captures/'plan.json').read_text())
    if 'Llama-3.1-8B' not in manifest['model_args']:
        parser.error('pilot expects the dense Llama-3.1-8B captures')
    rope_data = json.loads((args.captures/'rope.json').read_text())
    if rope_data.get('inv_freq'):
        rope_data['inv_freq'] = tuple(rope_data['inv_freq'])
    rope = RopeTables(**rope_data)
    protocol = dict(model_args=manifest['model_args'], captures=[str(p.resolve()) for p in files],
                    sampling=manifest['sampling'], layers=args.layers, prefixes=args.prefixes, budgets=args.budgets,
                    menus={str(g):list(menu) for g,menu in hierarchy.MENUS.items()}, window_tokens=8,
                    accounting={layout:hierarchy.accounting(layout) for layout in ('separate','shared')},
                    variants=[dict(max_group=size, mask_layout=layout, name=name) for size,layout,name in VARIANTS], scope=__doc__)
    plan_path = args.out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != protocol:
        parser.error('different pilot protocol exists; choose a fresh --out')
    write_json(plan_path, protocol)
    rows = [];format_rows = [];started = time.monotonic();completed = 0
    total = len(files)*len(args.layers)*2*len(args.prefixes)
    for path in files:
        captured = torch.load(path, weights_only=True, map_location='cpu')
        if captured['keys'].shape[:2] != (32,8) or captured['keys'].shape[-1] != 128 or captured['values'].shape != captured['keys'].shape:
            raise ValueError('capture must contain matching dense 32-layer/8-head/128-dim keys and values')
        if max(args.prefixes) > captured['keys'].shape[-2]:
            raise ValueError('capture is shorter than the requested prefix')
        for layer in args.layers:
            for target, field in (('k','keys'),('v','values')):
                for tokens in args.prefixes:
                    body = captured[field][layer:layer+1, :, :tokens].to(args.device)
                    metadata = dict(chunk=int(captured['chunk']), layer=layer, target=target, tokens=tokens)
                    with TimingRecord(args.out, name='hierarchy_probe', run=plan_path, worker=0, identity=metadata):
                        def progress(row):
                            rows.append({**metadata, **row})
                            write_json(args.out/'preliminary.json', dict(status='preliminary', completed_probes=completed,
                                total_probes=total, elapsed_seconds=time.monotonic()-started, rows=rows))
                        measure(body, target=target, rope=rope, budgets=args.budgets, progress=progress,
                                format_progress=lambda row:format_rows.append({**metadata,**row}))
                    completed += 1
                    elapsed = time.monotonic()-started
                    print(f'[preliminary] {completed}/{total} probes; chunk {captured["chunk"]}, layer {layer}, {target}, '
                          f'{tokens} tokens; elapsed {elapsed:.1f}s; estimated remaining {elapsed/completed*(total-completed):.1f}s', flush=True)
    write_json(args.out/'summary.json', dict(status='complete', protocol=protocol, rows=rows,
        comparisons=comparisons(rows), format_comparisons=format_rows, elapsed_seconds=time.monotonic()-started,
        warning='Reconstruction proxy only, not accuracy. Compare distortion only where observed savings match. '
                'Per-slot diagnostic targets do not perform layer allocation. Dense tail tokens are unchanged.'))
    print(f'Complete: {args.out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
