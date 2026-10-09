#!/usr/bin/env python3
"""Clean four-rung study, with fresh per-layer calibration in every rung.

1 raw aligned pairs; 2 directional pairs/residuals/norms; 3 query-ranked key
residuals; 4 fixed pairs/quads selected per layer/K/V slot. No token importance,
no per-block format menu, no old calibration reuse. Decode appends are dense.
Calibration/query weights use TRAIN, validation uses VALIDATION, final WikiText
uses TEST; C-Eval valid is full 5-shot. Fresh ledger lives inside this study.
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
from compression_topics.spatial.algorithms import presentation_spatial as kernel
from engine.eval_runner.files import write_json
from engine.eval_runner.chunks import partition
from engine.eval_runner.text_chunks import wikitext_chunks
from engine.layer_select.calibration import configuration, execute_chunks, measurement, perplexity
from engine.layer_select.greedy.calibrated import allocate
from engine.kv_compress.spec import parse_kv_spec

OUT = ROOT / 'compression_topics/spatial/figures/presentation_ladder_clean'
STAGES = ('weights', 'pilot', 'screen', 'refine', 'allocate', 'validate', 'tasks', 'summary')
LABELS = {1: 'aligned pairs baseline', 2: 'pairs + residuals + norm restoration',
          3: 'query-weighted key residuals', 4: 'calibrated fixed pairs/quads'}


def sampling(plan, stage):
    if stage == 'validate':
        return {'split': 'validation', 'pool_chunks': plan['validation_chunks'], 'offset': 0,
                'chunks': plan['validation_chunks'], 'seed': plan['seed'], 'seq_len': plan['seq_len']}
    offset = 0 if stage == 'pilot' else 4 if stage in ('weights', 'screen') else 4 + plan['screen_chunks']
    count = 4 if stage == 'pilot' else plan['screen_chunks'] + plan['refine_chunks'] if stage == 'weights' else plan[stage + '_chunks']
    return {'split': 'train', 'pool_chunks': 4 + plan['screen_chunks'] + plan['refine_chunks'],
            'offset': offset, 'chunks': count, 'seed': plan['seed'], 'seq_len': plan['seq_len']}


def study_plan(out, *, layers=32, local_budgets=(.1, .25, .4, .49, .6, .7),
               budgets=(.1, .2, .3, .4, .5), residuals=(0, 8, 16, 32), screen_chunks=4,
               refine_chunks=12, validation_chunks=16, seed=29, model_args=LLAMA31_8B):
    return {'version': 1, 'accounting': kernel.ACCOUNTING, 'out': str(out), 'layers': layers,
            'head_dim': 128, 'model_args': model_args, 'local_budgets': list(local_budgets),
            'global_budgets': list(budgets), 'residuals': list(residuals), 'screen_chunks': screen_chunks,
            'refine_chunks': refine_chunks, 'validation_chunks': validation_chunks, 'seed': seed,
            'seq_len': 2048, 'labels': {str(k): v for k, v in LABELS.items()},
            'norm_representation': 'one precomputed 16-bit restoration coefficient per merged token',
            'protocol': 'fresh_train_calibration_validation_holdout_ceval_valid5shot_wikitext_test_v1',
            'ranking': 'worst relative squared reconstruction error', 'finalists': 1}


def step(plan, layer, target, budget, size, residual, directional, selection='deviation', digest=None):
    result = {'method': 'presentation_spatial', 'accounting': kernel.ACCOUNTING,
              'k_layers': [layer] if target == 'k' else [], 'v_layers': [layer] if target == 'v' else [],
              'saving': budget, 'group_size': size, 'residual_entries': residual,
              'directional': directional, 'select_by': selection}
    if selection == 'query':
        result.update(query_weights=str(Path(plan['out']) / 'query_weights.pt'), query_weights_sha256=digest)
    parse_kv_spec({'pipeline': [result]})
    return result


def candidates(plan, digest):
    rows = []
    def add(layer, target, budget, size, residual, directional, selection, rungs):
        if budget > kernel.maximum_saving(residual, plan['head_dim'], size, directional) + 1e-12:
            return
        tag = f"{'dir' if directional else 'raw'}_{selection}_g{size}_L{layer:02d}_{target}_b{round(budget*100):02d}_r{residual:02d}"
        spec = {'pipeline': [step(plan, layer, target, budget, size, residual, directional, selection, digest)]}
        rows.append({'name': tag, 'layer': layer, 'target': target, 'budget': budget,
                     'group_size': size, 'residual_entries': residual, 'rungs': rungs, 'kv': spec})
    for layer in range(plan['layers']):
        for target in ('k', 'v'):
            for budget in plan['local_budgets']:
                add(layer, target, budget, 2, 0, False, 'deviation', [1])
                for residual in plan['residuals']:
                    # Identical candidates are evaluated once within this fresh study.
                    shared = target == 'v' or residual == 0
                    add(layer, target, budget, 2, residual, True, 'deviation', [2, 3, 4] if shared else [2])
                    if not shared:
                        add(layer, target, budget, 2, residual, True, 'query', [3, 4])
                    add(layer, target, budget, 4, residual, True,
                        'query' if target == 'k' and residual else 'deviation', [4])
    return rows


def chunk_run(plan, configs, stage):
    return {'model': 'hf', 'model_args': plan['model_args'], 'batch_size': 1,
            'model_shape': {'layers': plan['layers'], 'head_dim': plan['head_dim']},
            'sampling': sampling(plan, stage), 'preliminary_results': True, 'configurations': configs}


def load(path):
    return json.loads(Path(path).read_text())


def capture_weights(plan):
    """Dense train prefills only; no key dumps, no attention probability calculation."""
    import torch
    import transformers.models.llama.modeling_llama as llama
    from engine.eval_runner.execute import load_model_if_needed
    out = Path(plan['out'])
    path = out / 'query_weights.pt'
    source = {'model_args': plan['model_args'], 'sampling': sampling(plan, 'weights'),
              'layers': plan['layers'], 'head_dim': plan['head_dim']}
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('source') != source:
            raise ValueError('existing weights have a different model or training sample')
        write_json(out / 'weight_shape.json', {'layers': payload['weights'].shape[0],
                                             'kv_heads': payload['weights'].shape[1],
                                             'head_dim': payload['weights'].shape[2]})
        return
    lm, _, _, _ = load_model_if_needed(None, None, {'model': 'hf', 'model_args': plan['model_args'], 'batch_size': 1}, {})
    model = lm.model.eval()
    config = getattr(model.config, 'text_config', model.config)
    if config.model_type != 'llama' or config.num_hidden_layers != plan['layers'] or \
       config.hidden_size // config.num_attention_heads != plan['head_dim']:
        raise ValueError('fresh query-weight capture model does not match study shape')
    sample = source['sampling']
    prompts, ids = wikitext_chunks(lm.tokenizer, sample['pool_chunks'], sample['seq_len'], sample['seed'], split='train')
    prompts, ids = partition(prompts, ids, sample)
    sums, counts, cursor = {}, {}, [0]
    original = llama.apply_rotary_pos_emb
    def rotary(*args, **kwargs):
        q, k = original(*args, **kwargs)
        layer = cursor[0]
        cursor[0] += 1
        grouped = q[0].float().reshape(k.shape[1], q.shape[1] // k.shape[1], q.shape[-2], q.shape[-1])
        energy = grouped.square().sum((1, 2)).cpu()
        sums[layer] = sums.get(layer, 0) + energy
        counts[layer] = counts.get(layer, 0) + q.shape[-2]
        return q, k
    llama.apply_rotary_pos_emb = rotary
    try:
        with torch.inference_mode():
            for number, (prompt, chunk) in enumerate(zip(prompts, ids), 1):
                cursor[0] = 0
                output = model(input_ids=prompt[None].to(model.device), use_cache=False)
                del output
                if cursor[0] != plan['layers']:
                    raise ValueError('query capture did not observe each layer exactly once')
                print(f'[fresh query weights] {number}/{len(prompts)}, train chunk {chunk}', flush=True)
    finally:
        llama.apply_rotary_pos_emb = original
    weights = torch.stack([sums[layer] / counts[layer] for layer in range(plan['layers'])])
    temporary = path.with_suffix('.pt.tmp')
    torch.save({'weights': weights, 'source': source, 'chunk_ids': ids}, temporary)
    temporary.replace(path)
    write_json(out / 'weight_shape.json', {'layers': weights.shape[0], 'kv_heads': weights.shape[1], 'head_dim': weights.shape[2]})


def measure_rows(plan, rows, stage):
    out = Path(plan['out']) / stage
    dense_payload = load(out / 'dense.json')
    baseline = perplexity(dense_payload)
    dense_scores = dense_payload['chunk_scores']
    measured = []
    for row in rows:
        payload = load(out / f"{row['name']}.json")
        if payload['simulation']['kv'] != row['kv']:
            raise ValueError(f"stale policy result {row['name']}")
        scores = payload['chunk_scores']
        if [r['chunk'] for r in scores] != [r['chunk'] for r in dense_scores]:
            raise ValueError('calibration requires identical dense/candidate chunk IDs')
        stats = measurement(payload)
        if stats['shortfall_updates'] or stats['compression'] < row['budget'] - .001:
            raise ValueError(f"candidate cannot meet its finite measured byte budget: {row['name']}")
        differences = [r['nll'] - d['nll'] for r, d in zip(scores, dense_scores)]
        ppl = perplexity(payload)
        measured.append({**row, **stats, 'ppl': ppl, 'delta_nll': math.log(ppl / baseline),
                         'paired_nll_standard_error': statistics.stdev(differences) / math.sqrt(len(differences)),
                         'chunk_scores': scores})
    return {'dense_ppl': baseline, 'candidates': measured}


def winners(calibration, rung, *, by_size=False):
    rows = [r for r in calibration['candidates'] if rung in r['rungs']]
    fields = ('layer', 'target', 'budget', 'group_size') if by_size else ('layer', 'target', 'budget')
    groups = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key not in groups or (row['ppl'], row['residual_entries'], row['group_size']) < \
           (groups[key]['ppl'], groups[key]['residual_entries'], groups[key]['group_size']):
            groups[key] = row
    return list(groups.values())



def budget_cases(plan):
    """Report impossible pair targets explicitly; never substitute best effort."""
    cases = []
    for rung in range(1, 5):
        ceiling = kernel.maximum_saving(0, plan['head_dim'], 4 if rung == 4 else 2, rung != 1)
        for budget in plan['global_budgets']:
            cases.append({'rung': rung, 'label': LABELS[rung], 'budget': budget,
                          'status': 'eligible' if budget < ceiling else 'unattainable',
                          'reason': 'Target exceeds format storage ceiling including metadata.'
                                    if budget >= ceiling else None})
    return cases


def select_allocations(plan, calibration):
    selections = []
    for rung in range(1, 5):
        selected = winners(calibration, rung)
        conservative = {**calibration, 'selected': [
            {**r, 'measured_calibration_compression': r['compression'], 'compression': r['budget']} for r in selected]}
        for budget in [c['budget'] for c in budget_cases(plan)
                       if c['rung'] == rung and c['status'] == 'eligible']:
            assignment = allocate(conservative, plan['layers'], ('k', 'v'), budget)
            assignment.update(rung=rung, label=LABELS[rung], tag=f'rung{rung}_b{round(budget*100):02d}')
            selections.append(assignment)
    return selections


def summary(plan):
    out = Path(plan['out'])
    report = {'status': 'preliminary', 'protocol': plan['protocol'], 'labels': plan['labels'], 'rows': [],
              'unattainable': [c for c in budget_cases(plan) if c['status'] == 'unattainable']}
    dense_paths = {task: out / 'tasks' / f'dense_{task}.json' for task in ('ceval', 'wikitext')}
    dense = {task: load(path)['results']['ceval-valid' if task == 'ceval' else 'wikitext'][
        'acc,none' if task == 'ceval' else 'word_perplexity,none'] for task, path in dense_paths.items() if path.exists()}
    report['dense'] = dense
    selection_path = out / 'selections.json'
    for selection in load(selection_path) if selection_path.exists() else []:
        row = {'rung': selection['rung'], 'label': selection['label'], 'budget': selection['budget'],
               'assigned_saving_lower_bound': selection['compression'],
               'residual_slot_counts': {str(k): sum(r['residual_entries'] == k for r in selection['assignment'])
                                        for k in plan['residuals']},
               'pair_slots': sum(r['group_size'] == 2 for r in selection['assignment']),
               'quad_slots': sum(r['group_size'] == 4 for r in selection['assignment'])}
        for task in ('ceval', 'wikitext'):
            path = out / 'tasks' / f"{selection['tag']}_{task}.json"
            if path.exists():
                payload = load(path)
                metric = payload['results']['ceval-valid' if task == 'ceval' else 'wikitext'][
                    'acc,none' if task == 'ceval' else 'word_perplexity,none']
                row[task] = metric
                row[task + '_measured_kv_saving'] = payload['storage']['targets']['kv']['compression']
                if task in dense:
                    row[task + '_delta'] = metric - dense[task]
        report['rows'].append(row)
    if len(report['rows']) == sum(c['status'] == 'eligible' for c in budget_cases(plan)) and all('ceval' in r and 'wikitext' in r for r in report['rows']):
        report['status'] = 'complete'
    write_json(out / 'summary.json', report)
    if report['status'] == 'complete':
        plot_report(report, out)
    print(json.dumps(report, indent=2), flush=True)
    return report


def plot_report(report, out):
    """Export static scientific figures for slides; no plotting dependency required to evaluate."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Results complete; matplotlib unavailable, so plot-ready data is in summary.json.', flush=True)
        return
    for task, label, filename, factor in (
            ('ceval', 'C-Eval accuracy (%)', 'accuracy_vs_storage', 100),
            ('wikitext', 'WikiText test word perplexity', 'perplexity_vs_storage', 1)):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), layout='constrained')
        for rung in range(1, 5):
            rows = sorted((r for r in report['rows'] if r['rung'] == rung), key=lambda r: r[task + '_measured_kv_saving'])
            ax.plot([100 * r[task + '_measured_kv_saving'] for r in rows],
                    [factor * r[task] for r in rows], marker='o', label=f'{rung}: {LABELS[rung]}')
        ax.axhline(factor * report['dense'][task], color='black', linestyle='--', label='Dense')
        ax.set(xlabel='Measured whole-KV storage saving (%)', ylabel=label,
               title='Fresh calibrated spatial compression')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for suffix in ('svg', 'png'):
            fig.savefig(out / f'{filename}.{suffix}', dpi=180)
        plt.close(fig)


def run_stage(plan, stage):
    out = Path(plan['out'])
    if stage == 'weights':
        capture_weights(plan)
        return
    digest = hashlib.sha256((out / 'query_weights.pt').read_bytes()).hexdigest()
    rows = candidates(plan, digest)
    if stage in ('pilot', 'screen', 'refine', 'validate'):
        destination = out / stage
        configs = [configuration('dense', {'pipeline': []}, destination, WIKITEXT_FULL)]
        if stage == 'pilot':
            for rung in range(1, 5):
                size = 4 if rung == 4 else 2
                residual = 0 if rung == 1 else 16
                spec = {'pipeline': [step(plan, layer, target, .3, size, residual, rung != 1,
                                         'query' if rung >= 3 and target == 'k' else 'deviation', digest)
                                     for layer in range(plan['layers']) for target in ('k', 'v')]}
                configs.append(configuration(f'pilot_rung{rung}', spec, destination, WIKITEXT_FULL))
        elif stage in ('screen', 'refine'):
            if stage == 'refine':
                shortlisted = load(out / 'shortlist.json')
                rows = [r for r in rows if r['name'] in shortlisted]
            configs += [configuration(r['name'], r['kv'], destination, WIKITEXT_FULL) for r in rows]
        else:
            configs += [configuration(r['tag'], r['kv'], destination, WIKITEXT_FULL) for r in load(out / 'selections.json')]
        path = out / f'{stage}_run.json'
        write_json(path, chunk_run(plan, configs, stage))
        execute_chunks(path)
        if stage == 'screen':
            measured = measure_rows(plan, rows, stage)
            write_json(out / 'screening.json', measured)
            chosen = {r['name'] for rung in range(1, 5) for r in winners(measured, rung, by_size=True)}
            write_json(out / 'shortlist.json', sorted(chosen))
            print(f'Shortlisted {len(chosen)}/{len(rows)} candidates for fresh refinement.', flush=True)
        elif stage == 'refine':
            measured = measure_rows(plan, rows, stage)
            write_json(out / 'calibration.json', measured)
        elif stage == 'pilot':
            baseline = perplexity(load(destination / 'dense.json'))
            pilot = []
            for rung in range(1, 5):
                payload = load(destination / f'pilot_rung{rung}.json')
                measured = measurement(payload)
                pilot.append({'rung': rung, 'token_ppl': perplexity(payload), 'dense_token_ppl': baseline,
                              'compression': measured['compression'], 'shortfalls': measured['shortfall_updates']})
            write_json(out / 'pilot.json', pilot)
            if any(r['shortfalls'] or r['compression'] < .299 for r in pilot):
                raise RuntimeError('pilot byte accounting failed; inspect pilot.json before calibration')
        else:
            base = perplexity(load(destination / 'dense.json'))
            validation = []
            for selection in load(out / 'selections.json'):
                payload = load(destination / f"{selection['tag']}.json")
                saving = payload['storage']['targets']['kv']['compression']
                validation.append({'rung': selection['rung'], 'budget': selection['budget'],
                                   'token_ppl': perplexity(payload), 'delta_nll': math.log(perplexity(payload) / base),
                                   'measured_kv_saving': saving})
                if saving < selection['budget'] - .001:
                    raise RuntimeError(f"validation misses bytes for {selection['tag']}")
            write_json(out / 'validation.json', validation)
            print(json.dumps(validation, indent=2), flush=True)
        return
    if stage == 'allocate':
        selections = select_allocations(plan, load(out / 'calibration.json'))
        write_json(out / 'selections.json', selections)
        return
    if stage == 'tasks':
        if not (out / 'validation.json').exists():
            raise RuntimeError('run clean held-out validation before final tasks')
        tasks = {'ceval': CEVAL_VALID_5SHOT, 'wikitext': WIKITEXT_FULL}
        configs = [configuration('dense_' + name, {'pipeline': []}, out / 'tasks', task) for name, task in tasks.items()]
        configs += [configuration(selection['tag'] + '_' + name, selection['kv'], out / 'tasks', task,
                                  rung=selection['rung'], kv_budget=selection['budget'])
                    for selection in load(out / 'selections.json') for name, task in tasks.items()]
        path = out / 'tasks_run.json'
        write_json(path, {'model': 'hf', 'model_args': plan['model_args'], 'batch_size': 1, 'configurations': configs})
        # A separate fresh index makes even dense results independent of old experiments.
        subprocess.run([sys.executable, str(ROOT / 'engine/multi_run.py'), '--run', str(path), '--skip-existing',
                        '--write-results', '--results-root', str(out)], cwd=ROOT, check=True)
    summary(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--screen-chunks', type=int, default=4)
    parser.add_argument('--refine-chunks', type=int, default=12)
    parser.add_argument('--validation-chunks', type=int, default=16)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--from', dest='start', choices=STAGES, default='weights')
    parser.add_argument('--through', choices=STAGES, default='summary')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if STAGES.index(args.start) > STAGES.index(args.through) or min(args.screen_chunks, args.refine_chunks, args.validation_chunks) < 2:
        parser.error('invalid stage order or fewer than two chunks')
    out = args.out.resolve()
    plan = study_plan(out, screen_chunks=args.screen_chunks, refine_chunks=args.refine_chunks,
                      validation_chunks=args.validation_chunks, seed=args.seed)
    path = out / 'plan.json'
    if path.exists() and load(path) != plan:
        parser.error('different clean protocol already exists here; use a new output directory')
    write_json(path, plan)
    count = len(candidates(plan, 'capture-pending'))
    print(f'Fresh four-rung study: {count} single-slot candidates on {args.screen_chunks} TRAIN chunks; '
          f'one finalist per rung/group-size/slot/budget refined on {args.refine_chunks} disjoint TRAIN chunks.', flush=True)
    task_count = 2 + 2 * sum(c['status'] == 'eligible' for c in budget_cases(plan))
    print(f'{args.validation_chunks} VALIDATION chunks; final {task_count} task configurations.', flush=True)
    print('All key variants use RoPE. Whole-KV saving: 10/20/30/40% for all rungs; 50% for rung 4. '
          '50% is unattainable for pair-only rungs 1–3. Fresh local ledger; no old results.', flush=True)
    if not args.execute:
        print('Plan only; add --execute on the DGX.', flush=True)
        return
    for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through) + 1]:
        print(f'=== Clean presentation stage: {stage} ===', flush=True)
        run_stage(plan, stage)


if __name__ == '__main__':
    main()
