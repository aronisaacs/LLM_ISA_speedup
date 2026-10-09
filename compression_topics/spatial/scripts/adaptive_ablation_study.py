#!/usr/bin/env python3
"""Controlled fixed/adaptive/attention-importance comparison on causal continuations.

All arms share group_rd, query-weighted keys, RoPE, norms, bitmap masks and
16-bit accounting. Only format restrictions and token importance differ.
Each arm is independently calibrated on TRAIN. VALIDATION and TEST score only
512 tokens after a 1536-token prefill; decode appends remain dense.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts import presentation_ladder_study as clean
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration, execute_chunks, perplexity
from engine.layer_select.greedy.calibrated import allocate

OUT = ROOT / 'compression_topics/spatial/figures/adaptive_ablation'
STAGES = ('weights', 'pilot', 'screen', 'refine', 'allocate', 'validate', 'test', 'tasks', 'summary')
ARMS = ('fixed', 'adaptive', 'importance')
MENU = ('D', 'P32/d', 'P32+32', 'P8/d', 'P8+8', 'Q16', 'Q8', 'Q0')
FAMILIES = {'pair32': ('D', 'P32/d', 'P32+32'), 'pair8': ('D', 'P8/d', 'P8+8'),
            'quad16': ('D', 'Q16'), 'quad8': ('D', 'Q8'), 'quad0': ('D', 'Q0')}
LABELS = {'fixed': 'Fixed pair/quad family per layer/K/V slot',
          'adaptive': 'Adaptive eight-format blocks, no importance',
          'importance': 'Adaptive eight-format blocks + prefill attention importance'}


def study_plan(out, **kwargs):
    plan = clean.study_plan(out, **kwargs)
    plan.update(version=1, protocol='controlled_rd_train_validation_test_continuations_v1',
                accounting=group_rd.ACCOUNTING, menu=list(MENU),
                families={k: list(v) for k, v in FAMILIES.items()}, labels=LABELS,
                scoring_prefix=1536, test_chunks=64,
                controls='query-ranked key residuals/query distortion; values squared/deviation; RoPE; norms; bitmap; decode dense')
    return plan


def policy(plan, layer, target, budget, arm, family, digest):
    if arm not in ARMS or (arm == 'fixed' and family not in FAMILIES):
        raise ValueError('unknown arm or fixed family')
    step = {'method': 'group_rd', 'accounting': group_rd.ACCOUNTING,
            'k_layers': [layer] if target == 'k' else [], 'v_layers': [layer] if target == 'v' else [],
            'saving': budget, 'residuals': [0, 8, 16, 32],
            'menu': list(FAMILIES[family] if arm == 'fixed' else MENU),
            'granularity': 4, 'mask': 'bitmap', 'decode': False,
            'importance': 'prefill_attention' if arm == 'importance' else 'none',
            'distortion': 'query' if target == 'k' else 'squared',
            'select_by': 'query' if target == 'k' else 'deviation'}
    if target == 'k':
        step.update(query_weights=str(Path(plan['out']) / 'query_weights.pt'), query_weights_sha256=digest)
    parse_kv_spec({'pipeline': [step]})
    return step


def format_ceiling(names, dim=128):
    # An asymptotic upper bound; finite prefill accounting is checked by pilot/calibration.
    bits = group_rd.bit_table(names, (0, 8, 16, 32), dim)
    return 1 - (float(bits.min()) + group_rd.mode_bits(len(names))) / (4 * dim * 16)


def candidates(plan, digest):
    rows = []
    for arm in ARMS:
        families = FAMILIES if arm == 'fixed' else {'all': MENU}
        for family, names in families.items():
            for layer in range(plan['layers']):
                for target in ('k', 'v'):
                    for budget in plan['local_budgets']:
                        if budget >= format_ceiling(names, plan['head_dim']):
                            continue
                        name = f'{arm}_{family}_L{layer:02d}_{target}_b{round(100*budget):02d}'
                        rows.append({'name': name, 'arm': arm, 'family': family, 'layer': layer,
                                     'target': target, 'budget': budget,
                                     'kv': {'pipeline': [policy(plan, layer, target, budget, arm, family, digest)]}})
    return rows


def sampling(plan, stage):
    if stage == 'test':
        return {'split': 'test', 'pool_chunks': plan['test_chunks'], 'offset': 0,
                'chunks': plan['test_chunks'], 'seed': plan['seed'], 'seq_len': plan['seq_len'],
                'scoring_prefix': plan['scoring_prefix']}
    result = clean.sampling(plan, stage)
    if stage != 'weights':
        result['scoring_prefix'] = plan['scoring_prefix']
    return result


def chunk_run(plan, configs, stage):
    return {'model': 'hf', 'model_args': plan['model_args'], 'batch_size': 1,
            'model_shape': {'layers': plan['layers'], 'head_dim': plan['head_dim']},
            'preliminary_results': True, 'sampling': sampling(plan, stage), 'configurations': configs}


def load(path):
    return json.loads(path.read_text())


def winners(calibration, arm, by_family=False):
    groups = {}
    for row in calibration['candidates']:
        if row['arm'] != arm:
            continue
        key = (row['layer'], row['target'], row['budget']) + ((row['family'],) if by_family else ())
        if key not in groups or (row['ppl'], row['name']) < (groups[key]['ppl'], groups[key]['name']):
            groups[key] = row
    return list(groups.values())


def allocations(plan, calibration):
    selections = []
    for arm in ARMS:
        conservative = {**calibration, 'selected': [
            {**r, 'measured_calibration_compression': r['compression'], 'compression': r['budget']}
            for r in winners(calibration, arm)]}
        for budget in plan['global_budgets']:
            choice = allocate(conservative, plan['layers'], ('k', 'v'), budget)
            choice.update(arm=arm, tag=f'{arm}_b{round(100*budget):02d}')
            selections.append(choice)
            print(f"[allocation] {choice['tag']}: {choice['compression']:.3%} assigned saving", flush=True)
    return selections


def summarize(plan):
    out = Path(plan['out'])
    report = {'status': 'preliminary', 'protocol': plan['protocol'], 'labels': LABELS,
              'scoring': '512 held-out continuation tokens after 1536-token prefill; prefill storage savings',
              'rows': [], 'dense': {}}
    for stage in ('validate', 'test'):
        path = out / stage / 'dense.json'
        if path.exists():
            report['dense'][stage + '_continuation_ppl'] = perplexity(load(path))
    dense_task = out / 'tasks/dense_ceval.json'
    if dense_task.exists():
        report['dense']['ceval_accuracy'] = load(dense_task)['results']['ceval-valid']['acc,none']
    for choice in load(out / 'selections.json') if (out / 'selections.json').exists() else []:
        row = {'arm': choice['arm'], 'budget': choice['budget'], 'assigned_saving': choice['compression']}
        for stage in ('validate', 'test'):
            path = out / stage / (choice['tag'] + '.json')
            if path.exists():
                d = load(path)
                row[stage + '_continuation_ppl'] = perplexity(d)
                row[stage + '_prefill_saving'] = d['storage']['targets']['kv']['compression']
        path = out / 'tasks' / (choice['tag'] + '_ceval.json')
        if path.exists():
            d = load(path)
            row['ceval_accuracy'] = d['results']['ceval-valid']['acc,none']
            row['ceval_prefill_saving'] = d['storage']['targets']['kv']['compression']
            row['ceval_delta_points'] = 100 * (row['ceval_accuracy'] - report['dense']['ceval_accuracy'])
            if row['ceval_prefill_saving'] < row['budget'] - .001:
                raise ValueError('final C-Eval misses storage target')
        report['rows'].append(row)
    if len(report['rows']) == len(ARMS) * len(plan['global_budgets']) and all(
            'ceval_accuracy' in r and 'test_continuation_ppl' in r and 'validate_continuation_ppl' in r
            for r in report['rows']):
        report['status'] = 'complete'
    write_json(out / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_stage(plan, stage):
    out = Path(plan['out'])
    if stage == 'weights':
        clean.capture_weights(plan)
        return
    if stage not in ('pilot', 'summary') and not (out / 'pilot_passed.json').exists():
        raise RuntimeError('run and pass storage pilot first')
    digest = hashlib.sha256((out / 'query_weights.pt').read_bytes()).hexdigest()
    rows = candidates(plan, digest)
    if stage in ('pilot', 'screen', 'refine', 'validate', 'test'):
        dest = out / stage
        configs = [configuration('dense', {'pipeline': []}, dest, WIKITEXT_FULL)]
        if stage == 'pilot':
            for arm in ARMS:
                spec = {'pipeline': [policy(plan, l, t, .4, arm, 'quad16', digest)
                                     for l in range(plan['layers']) for t in ('k', 'v')]}
                configs.append(configuration('pilot_' + arm, spec, dest, WIKITEXT_FULL))
        elif stage in ('screen', 'refine'):
            if stage == 'refine':
                names = set(load(out / 'shortlist.json'))
                rows = [r for r in rows if r['name'] in names]
            configs += [configuration(r['name'], r['kv'], dest, WIKITEXT_FULL) for r in rows]
        else:
            configs += [configuration(r['tag'], r['kv'], dest, WIKITEXT_FULL) for r in load(out / 'selections.json')]
        path = out / (stage + '_run.json')
        write_json(path, chunk_run(plan, configs, stage))
        execute_chunks(path)
        if stage in ('screen', 'refine'):
            measured = clean.measure_rows(plan, rows, stage)
            write_json(out / ('screening.json' if stage == 'screen' else 'calibration.json'), measured)
            if stage == 'screen':
                # Keep every fixed family; selecting a family happens only after fresh refinement.
                names = sorted({r['name'] for arm in ARMS for r in winners(measured, arm, by_family=True)})
                write_json(out / 'shortlist.json', names)
                print(f'[screen] {len(names)} finalists; refinement uses disjoint TRAIN chunks', flush=True)
        else:
            checks = []
            for c in configs[1:]:
                d = load(Path(c['output_path']))
                if d['simulation']['kv'] != c['kv']:
                    raise ValueError('stale policy result')
                budget = .4 if stage == 'pilot' else next(r['budget'] for r in load(out / 'selections.json') if r['tag'] == c['name'])
                saving = d['storage']['targets']['kv']['compression']
                if saving < budget - .001:
                    raise ValueError(f"{c['name']} misses measured prefill storage target")
                checks.append({'name': c['name'], 'budget': budget, 'saving': saving,
                               'continuation_ppl': perplexity(d)})
            write_json(out / ('pilot_passed.json' if stage == 'pilot' else stage + '_checks.json'), checks)
            print(json.dumps(checks, indent=2), flush=True)
            # No quality cutoff: poor performance is a requested experimental result.
        return
    if stage == 'allocate':
        write_json(out / 'selections.json', allocations(plan, load(out / 'calibration.json')))
    elif stage == 'tasks':
        if not (out / 'validate_checks.json').exists() or not (out / 'test_checks.json').exists():
            raise RuntimeError('complete continuation validation/test before full C-Eval')
        configs = [configuration('dense_ceval', {'pipeline': []}, out / 'tasks', CEVAL_VALID_5SHOT)]
        configs += [configuration(r['tag'] + '_ceval', r['kv'], out / 'tasks', CEVAL_VALID_5SHOT)
                    for r in load(out / 'selections.json')]
        path = out / 'tasks_run.json'
        write_json(path, {'model': 'hf', 'model_args': plan['model_args'], 'batch_size': 1,
                         'preliminary_results': True, 'configurations': configs})
        subprocess.run([sys.executable, str(ROOT / 'engine/multi_run.py'), '--run', str(path),
                        '--skip-existing', '--write-results', '--results-root', str(out)], cwd=ROOT, check=True)
        summarize(plan)
    else:
        summarize(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--from', dest='start', choices=STAGES, default='weights')
    parser.add_argument('--through', choices=STAGES, default='summary')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if STAGES.index(args.start) > STAGES.index(args.through):
        parser.error('invalid stage order')
    plan = study_plan(args.out.resolve())
    path = args.out.resolve() / 'plan.json'
    if path.exists() and load(path) != plan:
        parser.error('existing output differs; use a new --out')
    write_json(path, plan)
    print(f"Controlled three-arm study: {len(candidates(plan, 'capture-pending'))} single-slot candidates; "
          '4 screen + 12 refinement TRAIN chunks; 16 VALIDATION + 64 TEST continuations; '
          '16 full C-Eval evaluations. Global savings 10/20/30/40/50%.', flush=True)
    print('8-format menu; fixed family vs adaptive vs importance. Query weighting held constant. '
          'RoPE/norms/accounting shared; decode appends dense. Preliminary files written per chunk.', flush=True)
    if not args.execute:
        print('Plan only; submit the Slurm launcher to execute.', flush=True)
        return
    for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through) + 1]:
        print(f'=== Controlled ablation stage: {stage} ===', flush=True)
        run_stage(plan, stage)

if __name__ == '__main__':
    main()
