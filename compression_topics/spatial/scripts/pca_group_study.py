#!/usr/bin/env python3
"""Coarse four-token PCA study with settings frozen per layer/KV-head/target.

Stages: collect dense directional samples; fit shared bases on fit chunks;
choose a residual size and fixed worst-token cosine-error threshold per head on
separate calibration chunks; screen and validate frozen whole-model policies
on two disjoint WikiText sets; evaluate all policies on full five-shot C-Eval.
Head selection minimizes mean cosine error at the requested calibration budget,
not singleton-head perplexity. No runtime rank/threshold adaptation. Actual
held-out savings can differ from calibrated budgets and are always reported.
Basis bits are charged once per prompt cache, not amortized over requests.
Keys and values have separate per-layer/per-head bases in pre-RoPE coordinates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms import pca_group as algorithm
from compression_topics.spatial.scripts.pair_calibration_chunks import partition
from compression_topics.spatial.scripts.pair_similarity_profile import wikitext_chunks, PRETRAINED
from compression_topics.spatial.scripts.group_calibrated_study import (
    configuration, execute_chunks, execute, write_json, measurement, perplexity)
from engine.kv_compress import install, parse_kv_spec

STAGES = ('collect', 'calibrate', 'screen', 'refine', 'tasks', 'summary')
SIZES = {'rank1': (32, 64, 128), 'shared': (4, 8, 16, 32), 'separate': (0, 8, 16, 32)}


def choose_setting(unit, kind, sizes, budget, reference_length, basis=None):
    """One fixed representation size and threshold for all groups in this head."""
    dim = unit.shape[-1]
    groups = reference_length // 4
    dense_bits = reference_length * dim * 16
    choices = []
    for size in sizes:
        if size > dim:
            continue
        reconstructed, errors = algorithm.reconstruct(unit, kind, size,
            None if basis is None else basis[:, :size])
        merged_bits, _, _ = algorithm.storage(kind, size, dim)
        gain = 4 * dim * 16 - merged_bits
        fixed = algorithm.overhead(kind, size, groups, dim)
        if gain <= 0:
            continue
        fraction = (budget * dense_bits + fixed) / (groups * gain)
        if fraction > 1:
            continue
        needed = max(1, math.ceil(fraction * len(errors) - 1e-10))
        sorted_errors = errors.sort().values
        cutoff = sorted_errors[needed - 1]
        if not torch.isfinite(cutoff):
            continue
        # Threshold itself is stored in fp16; round up to keep the fitted set.
        cutoff16 = cutoff.half()
        if cutoff16.float() < cutoff:
            cutoff16 = torch.nextafter(cutoff16, torch.tensor(float('inf'), dtype=torch.float16))
        threshold = float(cutoff16)
        selected = errors <= threshold
        achieved = (float(selected.float().mean()) * groups * gain - fixed) / dense_bits
        distortion = float(errors[selected].sum() / len(errors))
        choices.append({'kind': kind, 'size': size, 'threshold': threshold,
            'calibration_saving': achieved, 'calibration_merge_fraction': float(selected.float().mean()),
            'mean_cosine_error': distortion, 'reference_length': reference_length})
    if not choices:
        raise ValueError(f'no {kind} candidate reaches budget {budget} at length {reference_length}')
    return min(choices, key=lambda r: (r['mean_cosine_error'], r['size']))


def pipeline(settings, basis_path, basis_digest):
    steps = []
    for slot, heads in sorted(settings.items()):
        target, _, layer = slot.split('_')
        step = {'method': 'pca_quad', 'accounting': algorithm.ACCOUNTING,
                'k_layers': [int(layer)] if target == 'k' else [],
                'v_layers': [int(layer)] if target == 'v' else [], 'head_settings': heads}
        if any(h['kind'] == 'shared' for h in heads):
            step.update(basis_path=str(basis_path), basis_digest=basis_digest)
        steps.append(step)
    return {'pipeline': steps}


def collect(out, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    prompts, chosen = wikitext_chunks(tokenizer, args.pool_chunks, args.seq_len, args.seed)
    prompts, chosen = partition(prompts, chosen, {'seed': args.seed, 'offset': 0,
        'chunks': args.fit_chunks + args.calibration_chunks})
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED, dtype=torch.bfloat16).to('cuda').eval()
    algorithm.SAMPLES.clear()
    spec = parse_kv_spec({'pipeline': [{'method': 'pca_quad_collect', 'k_layers': 'all',
        'v_layers': 'all', 'samples_per_chunk': args.samples_per_chunk, 'seed': args.seed}]})
    uninstall = install(SimpleNamespace(model=model), spec)
    try:
        with torch.inference_mode():
            for i, ids in enumerate(prompts):
                model(input_ids=ids[None].to(model.device), use_cache=True)
                print(f'Dense PCA sample chunk {i+1}/{len(prompts)}', flush=True)
    finally:
        uninstall()
    samples = {slot: {'fit': torch.cat(rows[:args.fit_chunks], 1),
                      'calibration': torch.cat(rows[args.fit_chunks:], 1)}
               for slot, rows in algorithm.SAMPLES.items()}
    torch.save(samples, out / 'samples.pt')
    write_json(out / 'sample_chunks.json', {'fit': chosen[:args.fit_chunks],
        'calibration': chosen[args.fit_chunks:], 'canonical_pre_rope': True})
    algorithm.SAMPLES.clear()


def calibrate(out, args):
    samples = torch.load(out / 'samples.pt', map_location='cpu', weights_only=True)
    bases = {}
    for slot, data in sorted(samples.items()):
        head_bases = []
        for fit in data['fit']:
            unit = torch.nn.functional.normalize(fit.double(), dim=-1)
            residual = (unit - unit.mean(-2, keepdim=True)).reshape(-1, unit.shape[-1])
            _, vectors = torch.linalg.eigh(residual.T @ residual)
            head_bases.append(vectors.flip(-1)[:, :max(SIZES['shared'])].half())
        bases[slot] = torch.stack(head_bases)
    basis_path = (out / 'bases.pt').resolve()
    torch.save(bases, basis_path)
    digest = hashlib.sha256(basis_path.read_bytes()).hexdigest()
    policies = []
    for kind in ['separate'] + args.options:
        for budget in args.budgets:
            settings = {}
            for slot, data in sorted(samples.items()):
                settings[slot] = [choose_setting(torch.nn.functional.normalize(unit.float(), dim=-1),
                    kind, SIZES[kind], budget, args.seq_len,
                    bases[slot][head].float() if kind == 'shared' else None)
                    for head, unit in enumerate(data['calibration'])]
            tag = f'{kind}_b{round(budget * 100):02d}'
            policies.append({'tag': tag, 'kind': kind, 'budget': budget,
                'head_settings': settings, 'kv': pipeline(settings, basis_path, digest)})
            print(f'Calibrated {tag}: {sum(len(h) for h in settings.values())} fixed head settings', flush=True)
    write_json(out / 'policies.json', policies)
    for stage, offset, chunks in (
        ('screen', args.fit_chunks + args.calibration_chunks, args.screen_chunks),
        ('refine', args.fit_chunks + args.calibration_chunks + args.screen_chunks, args.refine_chunks)):
        configs = [configuration('dense_wikitext', {'pipeline': []}, out / stage, WIKITEXT_FULL)]
        configs += [configuration(p['tag'], p['kv'], out / stage, WIKITEXT_FULL,
            calibrated_budget=p['budget'], pca_kind=p['kind']) for p in policies]
        write_json(out / f'{stage}_run.json', {'model': 'hf', 'model_args': LLAMA31_8B,
            'batch_size': 1, 'configurations': configs, 'sampling': {
                'pool_chunks': args.pool_chunks, 'seed': args.seed, 'seq_len': args.seq_len,
                'offset': offset, 'chunks': chunks}})
    configs = [configuration('dense_ceval', {'pipeline': []}, out / 'tasks', CEVAL_VALID_5SHOT)]
    configs += [configuration(p['tag'], p['kv'], out / 'tasks', CEVAL_VALID_5SHOT,
        calibrated_budget=p['budget'], pca_kind=p['kind']) for p in policies]
    write_json(out / 'tasks_run.json', {'model': 'hf', 'model_args': LLAMA31_8B,
        'batch_size': 1, 'configurations': configs})


def summarize(out):
    policies = json.loads((out / 'policies.json').read_text())
    result = {'accounting': algorithm.ACCOUNTING, 'group_size': 4,
        'settings_fixed_per_layer_head_target': True, 'basis_charged_once_per_prompt_cache': True,
        'rows': []}
    for stage in ('screen', 'refine', 'tasks'):
        baseline = json.loads((out / stage / ('dense_ceval.json' if stage == 'tasks' else 'dense_wikitext.json')).read_text())
        for policy in policies:
            payload = json.loads((out / stage / f"{policy['tag']}.json").read_text())
            row = {'stage': stage, 'kind': policy['kind'], 'calibrated_budget': policy['budget'],
                   'actual_whole_kv_saving': measurement(payload)['compression'],
                   'compression_stats': measurement(payload)}
            if stage == 'tasks':
                dense = baseline['results']['ceval-valid']['acc,none']
                row.update(accuracy=payload['results']['ceval-valid']['acc,none'], dense_accuracy=dense)
                row['delta_accuracy'] = row['accuracy'] - dense
            else:
                row.update(token_ppl=perplexity(payload), dense_token_ppl=perplexity(baseline),
                           chunk_scores=payload['chunk_scores'])
            result['rows'].append(row)
    write_json(out / 'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'compression_topics/spatial/figures/pca_quad_heads')
    parser.add_argument('--options', choices=('rank1', 'shared'), nargs='+', default=['rank1', 'shared'])
    parser.add_argument('--budgets', type=float, nargs='+', default=[.2, .4, .6])
    parser.add_argument('--fit-chunks', type=int, default=4)
    parser.add_argument('--calibration-chunks', type=int, default=4)
    parser.add_argument('--samples-per-chunk', type=int, default=32)
    parser.add_argument('--screen-chunks', type=int, default=16)
    parser.add_argument('--refine-chunks', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--from', dest='start', choices=STAGES, default='collect')
    parser.add_argument('--through', choices=STAGES, default='summary')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if STAGES.index(args.start) > STAGES.index(args.through):
        parser.error('--from must not follow --through')
    if min(args.fit_chunks, args.calibration_chunks, args.screen_chunks, args.refine_chunks, args.samples_per_chunk) < 1 or args.seq_len < 4:
        parser.error('chunk/sample counts must be positive, seq-len >= 4')
    args.budgets = sorted(set(args.budgets))
    args.options = sorted(set(args.options))
    if not args.budgets or any(not math.isfinite(b) or not 0 < b <= .7 or abs(b*100-round(b*100)) > 1e-8 for b in args.budgets):
        parser.error('budgets must be whole percentages in (0, 70%]')
    # Baseline must support every requested budget as well.
    dim = 128
    for budget in args.budgets:
        for kind in ['separate'] + args.options:
            if not any((args.seq_len // 4 * (4*dim*16-algorithm.storage(kind, size, dim)[0])
                        - algorithm.overhead(kind, size, args.seq_len//4, dim)) /
                        (args.seq_len*dim*16) >= budget for size in SIZES[kind]):
                parser.error(f'{kind} cannot reach budget {budget} at the reference length')
    args.pool_chunks = args.fit_chunks + args.calibration_chunks + args.screen_chunks + args.refine_chunks
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in vars(args).items() if k not in ('out', 'start', 'through', 'execute')}
    manifest.update(accounting=algorithm.ACCOUNTING, sizes=SIZES, group_size=4,
        calibration='headwise geometric error; frozen policies validated on held-out token PPL',
        basis_storage='once per prompt cache at actual prompt length')
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        parser.error('output directory has a different experiment; use a new --out')
    write_json(path, manifest)
    print(f'PCA options: {args.options}; budgets: {args.budgets}; output: {out}', flush=True)
    print('Separate-residual quad baseline included. Fixed per-head settings; actual held-out savings are reported.', flush=True)
    if not args.execute:
        print('Plan only. Add --execute to run.')
        return
    for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through)+1]:
        print(f'Stage: {stage}', flush=True)
        if stage == 'collect':
            collect(out, args)
        elif stage == 'calibrate':
            calibrate(out, args)
        elif stage in ('screen', 'refine'):
            execute_chunks(out / f'{stage}_run.json')
        elif stage == 'tasks':
            execute(out / 'tasks_run.json', out)
        else:
            summarize(out)


if __name__ == '__main__':
    main()
