#!/usr/bin/env python3
"""Calibrate and evaluate the complete prefill policy, including token importance.

Fresh single-slot continuation-loss curves allocate the whole-KV budget.
The same eight-format policy is used in pilot, calibration and final tasks.
No generated-token compression or full-sequence importance-weighted perplexity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts.group_rd_offline import CANDIDATE_MENUS
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration, execute_chunks, measurement, perplexity
from engine.layer_select.greedy.calibrated import allocate

OUT = ROOT / 'compression_topics/spatial/figures/group_rd_integrated'
WEIGHTS = ROOT / 'compression_topics/spatial/figures/group_rd_offline/query_weights.pt'
STAGES = ('pilot', 'screen', 'refine', 'allocate', 'validate', 'tasks', 'summary')
GLOBAL_BUDGETS = (.1, .2, .3, .4, .5)


def policy(layer, target, saving, weights_path, weights_sha256):
    step = {'method': 'group_rd', 'accounting': group_rd.ACCOUNTING,
            'k_layers': [layer] if target == 'k' else [],
            'v_layers': [layer] if target == 'v' else [], 'saving': saving,
            'menu': list(CANDIDATE_MENUS['flag_d']), 'granularity': 4,
            'importance': 'prefill_attention', 'decode': False,
            'distortion': 'query' if target == 'k' else 'squared',
            'select_by': 'query' if target == 'k' else 'deviation'}
    if target == 'k':
        step.update(query_weights=str(weights_path), query_weights_sha256=weights_sha256)
    parse_kv_spec({'pipeline': [step]})
    return step


def plan(out, *, layers=32, budgets=(.1, .25, .4, .55, .7), global_budgets=GLOBAL_BUDGETS,
         screen_chunks=4, refine_chunks=12, validation_chunks=16, seed=17,
         model_args=LLAMA31_8B, weights_path=WEIGHTS):
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    rows = []
    for layer in range(layers):
        for target in ('k', 'v'):
            for budget in budgets:
                name = f'L{layer:02d}_{target}_b{round(budget * 100):02d}'
                kv = {'pipeline': [policy(layer, target, budget, weights_path, digest)]}
                rows.append({'name': name, 'layer': layer, 'target': target, 'budget': budget, 'kv': kv})
    return {'version': 1, 'layers': layers, 'local_budgets': list(budgets),
            'global_budgets': list(global_budgets), 'screen_chunks': screen_chunks,
            'refine_chunks': refine_chunks, 'validation_chunks': validation_chunks,
            'seed': seed, 'seq_len': 2048, 'scoring_prefix': 1536, 'model_args': model_args,
            'query_weights': str(weights_path), 'query_weights_sha256': digest,
            'policy': 'flag_d/query_residuals/prefill_attention/decode_dense_v1',
            'candidates': rows, 'out': str(out)}


def chunk_run(manifest, configs, *, offset, chunks):
    return {'model': 'hf', 'model_args': manifest['model_args'], 'batch_size': 1,
            'configurations': configs, 'preliminary_results': True,
            'model_shape': {'layers': manifest['layers'], 'head_dim': 128},
            # Fixed pool shared by pilot/screen/refine/validation; offsets disjoint.
            'sampling': {'pool_chunks': 4 + manifest['screen_chunks'] + manifest['refine_chunks'] +
                         manifest['validation_chunks'], 'offset': offset, 'chunks': chunks,
                         'seed': manifest['seed'], 'seq_len': manifest['seq_len'],
                         'scoring_prefix': manifest['scoring_prefix']}}


def dense(out):
    return configuration('dense', {'pipeline': []}, out, WIKITEXT_FULL)


def load(path):
    return json.loads(path.read_text())


def measured_curves(manifest, stage):
    out = Path(manifest['out']) / stage
    def chunks(payload, name):
        scores = list(payload['chunk_scores'])
        if stage == 'refine':
            screened = load(Path(manifest['out']) / 'screen' / f'{name}.json')
            if screened['simulation']['kv'] != payload['simulation']['kv']:
                raise ValueError('screen/refine policies differ')
            scores = screened['chunk_scores'] + scores
        if len({row['chunk'] for row in scores}) != len(scores):
            raise ValueError('calibration partitions overlap')
        return scores

    def score(scores):
        return math.exp(sum(r['nll'] * r['tokens'] for r in scores) / sum(r['tokens'] for r in scores))

    baseline_scores = chunks(load(out / 'dense.json'), 'dense')
    baseline = score(baseline_scores)
    rows = []
    for row in manifest['candidates']:
        payload = load(out / f"{row['name']}.json")
        if payload['simulation']['kv'] != row['kv']:
            raise ValueError(f"stale policy result: {row['name']}")
        measured = measurement(payload)
        if measured['compression'] < row['budget'] - .001:
            raise ValueError(f"candidate misses measured byte budget: {row['name']}")
        scores = chunks(payload, row['name'])
        if [r['chunk'] for r in scores] != [r['chunk'] for r in baseline_scores]:
            raise ValueError('candidate and dense calibration chunks differ')
        if stage == 'refine':
            earlier = load(Path(manifest['out']) / 'screen' / f"{row['name']}.json")
            stats = list(payload['gate_stats'].values()) + list(earlier['gate_stats'].values())
            measured['compression'] = 1 - sum(r['stored_bits'] for r in stats) / sum(r['dense_bits'] for r in stats)
        ppl = score(scores)
        differences = [r['nll'] - dense_row['nll'] for r, dense_row in zip(scores, baseline_scores)]
        rows.append({**row, **measured, 'ppl': ppl, 'delta_nll': math.log(ppl / baseline),
                     'paired_delta_nll_standard_error': statistics.stdev(differences) / math.sqrt(len(differences)),
                     'chunk_scores': scores})
    # Remove byte/error-dominated choices. No interpolation or extrapolation:
    # every retained rung is an actually evaluated policy at an actual saving.
    selected = []
    for layer in range(manifest['layers']):
        for target in ('k', 'v'):
            slot = [r for r in rows if (r['layer'], r['target']) == (layer, target)]
            frontier = [r for r in slot if not any(
                other['budget'] >= r['budget'] and other['ppl'] <= r['ppl'] and
                (other['budget'] > r['budget'] or other['ppl'] < r['ppl'])
                for other in slot if other is not r)]
            # Deduplicate equal compression; allocator requires strictly increasing bytes saved.
            by_rate = {}
            for row in sorted(frontier, key=lambda r: (r['ppl'], r['name'])):
                by_rate.setdefault(row['budget'], row)
            selected.extend(by_rate.values())
    return {'dense_ppl': baseline, 'candidates': rows, 'selected': selected,
            'scoring': '512 continuation tokens following 1536-token prefill',
            'policy': manifest['policy']}


def allocate_policy(calibration, layers, budget):
    # Do not rely on accidental discrete oversaving on calibration prompts to
    # meet the global budget on other prompt lengths. Each assigned saving is
    # a lower bound; retain measured rates alongside it for the report.
    conservative = {**calibration, 'selected': [
        {**row, 'measured_calibration_compression': row['compression'], 'compression': row['budget']}
        for row in calibration['selected']]}
    selection = allocate(conservative, layers, ('k', 'v'), budget)
    selection['measured_calibration_compression'] = sum(
        row['measured_calibration_compression'] for row in selection['assignment']) / (2 * layers)
    return selection


def summarize(manifest):
    out = Path(manifest['out'])
    report = {'policy': manifest['policy'], 'status': 'preliminary', 'rows': []}
    validation_dense = out / 'validate/dense.json'
    ceval_dense = out / 'tasks/dense_ceval.json'
    base_ppl = perplexity(load(validation_dense)) if validation_dense.exists() else None
    base_acc = load(ceval_dense)['results']['ceval-valid']['acc,none'] if ceval_dense.exists() else None
    report.update(dense_continuation_token_ppl=base_ppl, dense_ceval_accuracy=base_acc)
    selections_path = out / 'selections.json'
    for selection in load(selections_path) if selections_path.exists() else []:
        row = {'budget': selection['budget'], 'planned_prefill_kv_saving': selection['compression']}
        path = out / 'validate' / f"{selection['tag']}.json"
        if path.exists():
            result = load(path)
            row.update(continuation_token_ppl=perplexity(result),
                       measured_prefill_kv_saving=result['storage']['targets']['kv']['compression'])
        path = out / 'tasks' / f"{selection['tag']}_ceval.json"
        if path.exists():
            result = load(path)
            accuracy = result['results']['ceval-valid']['acc,none']
            row.update(ceval_accuracy=accuracy, ceval_delta_points=100 * (accuracy - base_acc) if base_acc is not None else None,
                       ceval_measured_kv_saving=(result.get('storage') or {}).get('targets', {}).get('kv', {}).get('compression'))
        report['rows'].append(row)
    if report['rows'] and all('ceval_accuracy' in r and 'continuation_token_ppl' in r for r in report['rows']):
        report['status'] = 'complete'
    write_json(out / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_stage(manifest, stage, pilot_max_delta_nll=.25):
    out = Path(manifest['out'])
    destination = out / stage
    if stage in ('screen', 'refine', 'allocate', 'validate'):
        gate_path = out / 'pilot_gate.json'
        if not gate_path.exists() or not load(gate_path)['passed']:
            raise RuntimeError('run and pass the integrated pilot before calibration')
    if stage == 'pilot':
        spec = {'pipeline': [policy(layer, target, .4, manifest['query_weights'], manifest['query_weights_sha256'])
                             for layer in range(manifest['layers']) for target in ('k', 'v')]}
        configs = [dense(destination), configuration('integrated_b40', spec, destination, WIKITEXT_FULL)]
        run = chunk_run(manifest, configs, offset=0, chunks=4)
    elif stage in ('screen', 'refine'):
        configs = [dense(destination)] + [configuration(r['name'], r['kv'], destination, WIKITEXT_FULL)
                                         for r in manifest['candidates']]
        offset = 4 if stage == 'screen' else 4 + manifest['screen_chunks']
        run = chunk_run(manifest, configs, offset=offset, chunks=manifest[stage + '_chunks'])
    elif stage == 'allocate':
        calibration = measured_curves(manifest, 'refine')
        write_json(out / 'calibration.json', calibration)
        selections = []
        for budget in manifest['global_budgets']:
            selection = allocate_policy(calibration, manifest['layers'], budget)
            selection['tag'] = f'integrated_b{round(budget * 100):02d}'
            selections.append(selection)
            print(f"Allocated {budget:.0%}: guaranteed assigned saving {selection['compression']:.2%}; "
                  f"{len(selection['assignment'])}/{2 * manifest['layers']} slots compressed", flush=True)
        write_json(out / 'selections.json', selections)
        return
    elif stage == 'validate':
        selections = load(out / 'selections.json')
        configs = [dense(destination)] + [configuration(r['tag'], r['kv'], destination, WIKITEXT_FULL) for r in selections]
        run = chunk_run(manifest, configs, offset=4 + manifest['screen_chunks'] + manifest['refine_chunks'],
                        chunks=manifest['validation_chunks'])
    elif stage == 'tasks':
        gate_path = out / 'validation_gate.json'
        if not gate_path.exists() or not all(r['passed'] for r in load(gate_path)['rows']):
            raise RuntimeError('full C-Eval requires a passed combined-model validation gate')
        selections = load(out / 'selections.json')
        configs = [configuration('dense_ceval', {'pipeline': []}, destination, CEVAL_VALID_5SHOT)]
        configs += [configuration(r['tag'] + '_ceval', r['kv'], destination, CEVAL_VALID_5SHOT,
                                  kv_budget=r['budget'], policy=manifest['policy']) for r in selections]
        run = {'model': 'hf', 'model_args': manifest['model_args'], 'batch_size': 1, 'configurations': configs}
        path = out / 'tasks_run.json'
        write_json(path, run)
        subprocess.run([sys.executable, str(ROOT / 'engine/multi_run.py'), '--run', str(path),
                        '--skip-existing', '--write-results', '--results-root', str(ROOT)], cwd=ROOT, check=True)
        summarize(manifest)
        return
    else:
        summarize(manifest)
        return
    path = out / f'{stage}_run.json'
    write_json(path, run)
    execute_chunks(path)
    if stage == 'pilot':
        dense_ppl = perplexity(load(destination / 'dense.json'))
        payload = load(destination / 'integrated_b40.json')
        delta = math.log(perplexity(payload) / dense_ppl)
        saving = payload['storage']['targets']['kv']['compression']
        result = {'delta_nll': delta, 'measured_prefill_saving': saving,
                  'max_delta_nll': pilot_max_delta_nll, 'passed': delta <= pilot_max_delta_nll and saving >= .399}
        write_json(out / 'pilot_gate.json', result)
        print(f'Pilot results: {result}', flush=True)
        if not result['passed']:
            raise RuntimeError('pilot gate failed; stopped before the layer sweep; inspect pilot results')
    elif stage == 'screen':
        write_json(out / 'screening.json', measured_curves(manifest, stage))
    elif stage == 'validate':
        report = summarize(manifest)
        gates = []
        for row in report['rows']:
            delta = math.log(row['continuation_token_ppl'] / report['dense_continuation_token_ppl'])
            gates.append({'budget': row['budget'], 'delta_nll': delta,
                          'saving': row['measured_prefill_kv_saving'],
                          'quality_warning': delta > pilot_max_delta_nll,
                          'passed': row['measured_prefill_kv_saving'] >= row['budget'] - .001})
        write_json(out / 'validation_gate.json', {'max_delta_nll': pilot_max_delta_nll, 'rows': gates})
        for row in gates:
            if row['quality_warning']:
                print(f"Quality warning at {row['budget']:.0%}: delta NLL {row['delta_nll']:.4f}; "
                      'continuing the requested full compression sweep.', flush=True)
        if not all(row['passed'] for row in gates):
            raise RuntimeError('combined-model byte accounting gate failed; stopped before full C-Eval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--local-budgets', type=float, nargs='+', default=[.1, .25, .4, .55, .7])
    parser.add_argument('--global-budgets', type=float, nargs='+', default=list(GLOBAL_BUDGETS))
    parser.add_argument('--screen-chunks', type=int, default=4)
    parser.add_argument('--refine-chunks', type=int, default=12)
    parser.add_argument('--validation-chunks', type=int, default=16)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--from', dest='start', choices=STAGES, default='pilot')
    parser.add_argument('--through', choices=STAGES, default='summary')
    parser.add_argument('--pilot-max-delta-nll', type=float, default=.25)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if STAGES.index(args.start) > STAGES.index(args.through):
        parser.error('--from must not follow --through')
    if any(n < 2 for n in (args.screen_chunks, args.refine_chunks, args.validation_chunks)):
        parser.error('need at least two chunks per stage')
    if any(not 0 < b <= .7 or abs(b * 100 - round(b * 100)) > 1e-8
           for b in args.local_budgets + args.global_budgets):
        parser.error('budgets must be whole percentages in (0,70%]')
    if not math.isfinite(args.pilot_max_delta_nll) or args.pilot_max_delta_nll < 0:
        parser.error('pilot threshold must be finite and nonnegative')
    manifest = plan(args.out.resolve(), budgets=tuple(sorted(set(args.local_budgets))),
                    global_budgets=tuple(sorted(set(args.global_budgets))), screen_chunks=args.screen_chunks,
                    refine_chunks=args.refine_chunks, validation_chunks=args.validation_chunks, seed=args.seed)
    path = args.out.resolve() / 'manifest.json'
    if path.exists() and load(path) != manifest:
        parser.error('existing output directory has a different policy or plan; use a new --out')
    write_json(path, manifest)
    print(f"Plan: 4-chunk pilot; {len(manifest['candidates'])} layer/K/V/budget configurations on "
          f"{args.screen_chunks}+{args.refine_chunks} disjoint chunks; fresh allocation; "
          f"{args.validation_chunks} held-out continuations; full C-Eval. Global savings {args.global_budgets}.", flush=True)
    print('Scoring: 1536-token prefill + 512-token continuation; importance uses prefill only; decode stays dense.', flush=True)
    if not args.execute:
        print('Plan only. Add --execute to run on the DGX.', flush=True)
        return
    for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through) + 1]:
        print(f'=== Integrated stage: {stage} ===', flush=True)
        run_stage(manifest, stage, args.pilot_max_delta_nll)


if __name__ == '__main__':
    main()
