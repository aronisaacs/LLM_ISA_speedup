#!/usr/bin/env python3
"""Small cached-vector proxy experiment; no model accuracy or calibration sweep.

First captured chunk selects fixed families; second chunk is evaluated.
Importance decisions use even prefill queries; attention-weighted scoring uses
odd queries. This is a within-prefill proxy, NOT future-query or task accuracy.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import statistics
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.algorithms.storage import metadata_bits
from compression_topics.spatial.scripts.adaptive_ablation_study import MENU, FAMILIES, ARMS
from compression_topics.spatial.scripts.group_rd_offline import load_rope
from engine.eval_runner.files import write_json


def subset(table, names):
    indices = [table.names.index(n) for n in group_rd.mode_names((0, 8, 16, 32), menu=names)]
    return replace(table, names=[table.names[i] for i in indices], distortion=table.distortion[..., indices],
                   bits=table.bits[indices], is_quad=table.is_quad[indices], first=table.first[indices],
                   second=table.second[indices], quad=table.quad[indices], swap=table.swap[..., indices])


def choose(table, saving, dense_bits):
    overhead = metadata_bits(group_rd.units(table), group_rd.mode_bits(len(table.names)))
    lam = group_rd.solve_lambda(table, dense_bits * (1 - saving) - overhead)
    choice = group_rd.select(table, lam)
    return choice, float(table.bits[choice].sum()) + overhead


def loss(table, choice):
    return float(table.distortion.gather(-1, choice.unsqueeze(-1)).sum())


def test_slot(calibration, heldout, scoring, seen, savings):
    rows = []
    dense = heldout.distortion.shape[1] * heldout.distortion.shape[2] * 4 * 128 * 16
    cal_dense = calibration.distortion.shape[1] * calibration.distortion.shape[2] * 4 * 128 * 16
    for saving in savings:
        options = []
        for family, names in FAMILIES.items():
            table = subset(calibration, names)
            try:
                choice, _ = choose(table, saving, cal_dense)
            except ValueError:
                continue
            options.append((loss(table, choice), family))
        if not options:
            raise ValueError('no fixed family can meet calibration saving')
        _, family = min(options)
        decisions = {'fixed': subset(heldout, FAMILIES[family]), 'adaptive': heldout, 'importance': seen}
        for arm in ARMS:
            table = decisions[arm]
            choice, stored = choose(table, saving, dense)
            score_table = subset(scoring, table.names)
            plain_table = subset(heldout, table.names)
            rows.append({'arm': arm, 'budget': saving, 'measured_saving': 1 - stored / dense,
                         'fixed_family': family if arm == 'fixed' else None,
                         'attention_weighted_error': loss(score_table, choice),
                         'unweighted_error': loss(plain_table, choice),
                         'mode_counts': {n: int((choice == i).sum()) for i, n in enumerate(table.names)}})
    return rows


def summarize(rows):
    result = []
    for target in ('k', 'v'):
        for budget in sorted({r['budget'] for r in rows}):
            chosen = [r for r in rows if r['target'] == target and r['budget'] == budget]
            for arm in ARMS:
                arm_rows = [r for r in chosen if r['arm'] == arm]
                if not arm_rows:
                    continue
                fixed = {r['layer']: r for r in chosen if r['arm'] == 'fixed'}
                gains = [1 - r['attention_weighted_error'] / fixed[r['layer']]['attention_weighted_error']
                         for r in arm_rows if fixed[r['layer']]['attention_weighted_error'] > 0]
                result.append({'target': target, 'budget': budget, 'arm': arm,
                               'median_attention_error_reduction_vs_fixed': statistics.median(gains) if gains else None,
                               'mean_measured_saving': statistics.mean(r['measured_saving'] for r in arm_rows)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-plan', type=Path, default=ROOT / 'compression_topics/spatial/figures/group_rd_offline/plan.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'compression_topics/spatial/figures/adaptive_ablation_quick')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 12, 20, 28])
    parser.add_argument('--savings', type=float, nargs='+', default=[.3, .4, .5])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if any(not 0 < b < .7 for b in args.savings):
        parser.error('savings must be between zero and 70%')
    manifest = json.loads(args.capture_plan.read_text())
    files = sorted(Path(manifest['keys_dir']).glob('chunk_*.pt'))[:2]
    if len(files) < 2:
        parser.error('need two captured chunks; no capture or full evaluation is started automatically')
    first, second = [torch.load(f, map_location='cpu', weights_only=True) for f in files]
    if first['chunk'] == second['chunk']:
        parser.error('calibration and held-out chunks must differ')
    weights = torch.load(manifest['query_weights'], map_location='cpu', weights_only=True)['weights']
    rope = load_rope(manifest['rope'])
    rows = []
    started = time.monotonic()
    provenance = {'capture_plan': str(args.capture_plan), 'chunks': [p['chunk'] for p in (first, second)],
                  'layers': args.layers, 'menu': list(MENU), 'families': {k: list(v) for k, v in FAMILIES.items()},
                  'scoring': 'held-out captured chunk; odd-prefill-query attention error; importance from even queries',
                  'limitations': 'proxy only; no layer allocation, task accuracy, or future decode queries; query weights shared from source capture'}
    for layer in args.layers:
        if not 0 <= layer < first['keys'].shape[0]:
            parser.error('requested layer not present in captures')
        for target in ('k', 'v'):
            key = 'keys' if target == 'k' else 'values'
            tables = []
            for capture in (first, second):
                body = capture[key][layer].unsqueeze(0).to(args.device).float()
                body = body[..., :body.shape[-2] // 4 * 4, :]
                opts = dict(rope_tables=rope if target == 'k' else None, residuals=(0, 8, 16, 32),
                            menu=MENU, distortion='query' if target == 'k' else 'squared',
                            weights=weights[layer].to(args.device) if target == 'k' else None,
                            select_by='query' if target == 'k' else 'deviation')
                plain = group_rd.build_menu(body, **opts)
                if capture is first:
                    tables.append(plain)
                else:
                    importance = capture['importance'][layer, 1:3, :, :body.shape[-2]].to(args.device).float()
                    seen, unseen = [w.unsqueeze(0) / w.mean().clamp_min(1e-12) for w in importance]
                    tables.extend([plain, group_rd.build_menu(body, token_weights=unseen, **opts),
                                   group_rd.build_menu(body, token_weights=seen, **opts)])
            results = test_slot(*tables, args.savings)
            rows.extend({**r, 'target': target, 'layer': layer} for r in results)
            report = {'status': 'preliminary', 'provenance': provenance, 'rows': rows, 'summary': summarize(rows)}
            write_json(args.out / 'results.partial.json', report)
            print(f'[proxy] layer {layer} {target}; {time.monotonic()-started:.1f}s elapsed', flush=True)
            for r in report['summary']:
                if r['target'] == target:
                    print(json.dumps(r), flush=True)
            del tables
    report.update(status='complete', elapsed_seconds=time.monotonic()-started)
    write_json(args.out / 'results.json', report)
    print(f"Complete. Results: {args.out / 'results.json'}. No accuracy calculations were run.", flush=True)

if __name__ == '__main__':
    main()
