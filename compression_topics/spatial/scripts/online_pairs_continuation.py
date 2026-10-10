#!/usr/bin/env python3
"""Small full-model continuation pilot; no C-Eval or fresh layer calibration.

Default: 4 fresh WikiText VALIDATION chunks, 1536 prefill + 512 continuation,
dense and three arms at the saved 30% allocation. Report actual savings, not
assumed matched budgets. Generated/continuation tokens remain dense.
"""
import argparse
from dataclasses import asdict
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from catalog.models import LLAMA31_8B
from catalog.tasks import WIKITEXT_FULL
from compression_topics.spatial.algorithms import online_pairs as online
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration,execute_chunks,perplexity

OUT=ROOT/'compression_topics/spatial/figures/online_pairs_continuation'
ARMS=('fixed_residual','adaptive_fixed_price','adaptive_feedback')


def build_run(out,selections,*,budgets=(.3,),chunks=4,layers=32,model_args=LLAMA31_8B,seed=73,prefix=1536,suffix=512):
    configs=[configuration('dense',{'pipeline':[]},out/'tasks',WIKITEXT_FULL)]
    for budget in budgets:
        selected=next(r for r in selections if r['rung']==2 and abs(r['budget']-budget)<1e-9)
        for arm in ARMS:
            steps=[]
            for row in selected['assignment']:
                target=row['target']
                step={'method':'online_pairs','accounting':online.ACCOUNTING,
                      'k_layers':[row['layer']] if target=='k' else [],'v_layers':[row['layer']] if target=='v' else [],
                      'saving':row['budget'],'feedback':arm=='adaptive_feedback','control':asdict(online.Control()),
                      'fixed_residual':row['residual_entries'] if arm=='fixed_residual' else None}
                steps.append(step)
            spec={'pipeline':steps};parse_kv_spec(spec)
            configs.append(configuration(f'{arm}_b{round(100*budget):02d}',spec,out/'tasks',WIKITEXT_FULL,
                                         arm=arm,global_budget=budget))
    return {'model':'hf','model_args':model_args,'batch_size':1,'preliminary_results':True,
            'model_shape':{'layers':layers,'head_dim':128},'configurations':configs,
            'sampling':{'split':'validation','pool_chunks':chunks,'offset':0,'chunks':chunks,'seed':seed,
                        'seq_len':prefix+suffix,'scoring_prefix':prefix}}


def summarize(out,run):
    results=[];dense_path=out/'tasks/dense.json'
    if not dense_path.exists():raise ValueError('dense baseline not complete')
    dense=json.loads(dense_path.read_text());base=perplexity(dense)
    for config in run['configurations'][1:]:
        path=Path(config['output_path'])
        if not path.exists():raise ValueError(f'missing result {path}')
        payload=json.loads(path.read_text())
        if payload['simulation']['kv']!=config['kv']:raise ValueError('saved policy differs')
        scores=payload['chunk_scores'];baseline=dense['chunk_scores']
        if [r['chunk'] for r in scores]!=[r['chunk'] for r in baseline]:raise ValueError('chunk IDs differ')
        if [r['tokens'] for r in scores]!=[r['tokens'] for r in baseline]:raise ValueError('token counts differ')
        differences=[a['nll']-b['nll'] for a,b in zip(scores,baseline)]
        rows=list(payload['gate_stats'].values())
        results.append({'name':config['name'],**config['metadata'],'continuation_token_ppl':perplexity(payload),
                        'relative_ppl_change':perplexity(payload)/base-1,'paired_mean_delta_nll':statistics.mean(differences),
                        'paired_delta_nll_standard_error':statistics.stdev(differences)/math.sqrt(len(differences)),
                        'actual_prefill_kv_saving':payload['storage']['targets']['kv']['compression'],
                        'max_local_target_deviation':max(r['max_final_target_deviation'] for r in rows),
                        'controller_bound_hits':sum(r['controller_bound_hits'] for r in rows),
                        'chunk_scores':scores})
    report={'status':'complete','dense_continuation_token_ppl':base,'sampling':run['sampling'],'rows':results,
            'scope':'small full-model validation continuation pilot; reused rung-2 allocation; actual savings may differ; no C-Eval',
            'definition':'continuation-only token perplexity, not full WikiText word perplexity'}
    write_json(out/'summary.json',report);print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--selections',type=Path,default=ROOT/'compression_topics/spatial/figures/presentation_ladder_clean_50/selections.json')
    p.add_argument('--budgets',type=float,nargs='+',default=[.3])
    p.add_argument('--chunks',type=int,default=4)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.chunks<2:p.error('need at least two chunks')
    if any(b not in (.1,.2,.3,.4) for b in args.budgets):p.error('use saved 10/20/30/40% allocations')
    run=build_run(args.out.resolve(),json.loads(args.selections.read_text()),budgets=args.budgets,chunks=args.chunks)
    path=args.out.resolve()/'run.json'
    if path.exists() and json.loads(path.read_text())!=run:p.error('different run exists; use new --out')
    write_json(path,run)
    print(f"Small pilot: {len(run['configurations'])} configurations x {args.chunks} chunks; "
          '1536-token prefill + 512-token continuation. No accuracy/calibration sweep.',flush=True)
    if args.execute:
        execute_chunks(path);summarize(args.out.resolve(),run)
    else:print('Plan only; use the Slurm launcher.',flush=True)

if __name__=='__main__':main()
