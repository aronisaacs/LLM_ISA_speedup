#!/usr/bin/env python3
"""Fixed-format online pairs at 10/20/30/40% (presentation rungs 1 and 2).

* no_residual: each pair is dense or merged with no residual;
* fixed_residual: one residual count (0/16/32) per layer K/V slot, chosen by
  calibration; dense is always allowed.

Same online controller, preprocessing and accounting as the main ladder.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms.formats import MENUS, max_saving
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json

ARMS = ('no_residual', 'fixed_residual')
OUT = ROOT / 'compression_topics/spatial/figures/fixed_rungs'


def plan(out, *, arms=ARMS, layers=32):
    p = base.base_plan(out, layers=layers, budgets=(.1, .2, .3, .4))
    p.update(version=1, protocol='online_fixed_pairs_v2', arms=list(arms), max_group=2,
             local_budgets=[.1, .2, .25, .3, .35, .4, .43, .47, .49], residuals=list(MENUS[2]),
             fixed_format_scope='one residual count per layer K/V slot; dense always allowed')
    return p


def candidates(p):
    rows = []
    for arm in p['arms']:
        for layer in range(p['layers']):
            for target in ('k', 'v'):
                for residual in ([0] if arm == 'no_residual' else p['residuals']):
                    config = SpatialConfig(max_group=2, fixed_residual=residual, control=p['control'])
                    for budget in p['local_budgets']:
                        if budget >= max_saving(p['head_dim'], 2, residual):
                            continue
                        rows.append(dict(name=f'{arm}_L{layer:02d}_{target}_b{round(100 * budget):02d}_r{residual}',
                                         arm=arm, layer=layer, target=target, budget=budget,
                                         residual_entries=residual,
                                         kv=config.pipeline(budget, **base.slot(layer, target))))
    return rows


def allocations(p, calibration):
    return [s for arm in p['arms'] for s in base.allocate(p, calibration, arm)]


def summarize(out, run, *, require_complete=False):
    return base.summarize(out, run, require_complete=require_complete, arm_pairs=(('no_residual', 'fixed_residual'),),
                          scope='online fixed-format pairs; full C-Eval 5-shot',
                          comparison='independent calibration and measured layer allocation per arm')


def run_stage(p, stage):
    return base.run_stage(p, stage, candidates=candidates, arms=p['arms'], allocations=allocations,
                          tasks=base.task_run, summarizer=summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--from', dest='start', choices=base.STAGES, default='screen')
    parser.add_argument('--through', choices=base.STAGES, default='summary')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    p = plan(args.out.resolve(), arms=tuple(dict.fromkeys(args.arms)))
    base.write_plan(p)
    print(f'Online fixed rungs: 10/20/30/40%; {len(candidates(p))} candidates.', flush=True)
    if args.execute:
        base.run_stages(p, args.start, args.through, run_stage)


if __name__ == '__main__':
    main()
