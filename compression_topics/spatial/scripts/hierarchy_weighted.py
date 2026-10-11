#!/usr/bin/env python3
"""Independently calibrated query residual / next-eight importance / shared-mask ablation.

Default runs through holdout WikiText only. Full C-Eval is an explicit --through
summary choice after inspecting the pilot. Existing unweighted job is untouched.
"""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from compression_topics.spatial.scripts import hierarchy_accuracy as base
from compression_topics.spatial.algorithms.composed import SpatialConfig
from engine.eval_runner.files import write_json
from engine.layer_select.study import measure_candidates
ARMS=('query','query_attention','query_shared','query_attention_shared')
OUT=ROOT/'compression_topics/spatial/figures/hierarchy_per_head_weighted'


def plan(out, budgets=(.3,.5,.7), screen=2, refine=2, validation=2):
    study=base.plan(out,budgets=budgets,screen=screen,refine=refine,validation=validation)
    study.update(version=3,protocol='per_head_absolute_query_error_own_calibration_v3',arms=list(ARMS),
                 query_weights=str(out/'query_weights.pt'),query_definition='train_post_rope_plane_tied_second_moment',
                 key_distortion='absolute_diagonal_query_weighted_squared_error_no_key_norm',
                 importance_definition='next8_queries_local16_causal_softmax_times16_floor0.1_dense_tail',
                 query_chunks=2,query_split='train',query_prefix=1536)
    return study


def candidates(study):
    path=Path(study['query_weights'])
    if not path.exists(): raise ValueError('run query calibration first')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    for arm in study['arms']:
        for row in base.candidates(study):
            attention='attention' in arm
            component=SpatialConfig(representation='hierarchy',control=study['control'],
                mask_layout='shared' if 'shared' in arm else 'separate',
                select_by='query' if row['target']=='k' else 'deviation',
                distortion='query' if row['target']=='k' else 'relative',
                query_weights=str(path) if row['target']=='k' else None,
                query_weights_sha256=digest if row['target']=='k' else None,
                importance='next8_attention' if attention else 'none')
            row.update(arm=arm,name=row['name'].replace('online_g8',arm),
                kv=component.pipeline(row['budget'],k_layers=[row['layer']] if row['target']=='k' else [],
                                      v_layers=[row['layer']] if row['target']=='v' else []))
            rows.append(row)
    return rows


def slot_measurement(payload):
    stats={k:v for k,v in payload.get('gate_stats',{}).items() if '_layer_' in k}
    if len(stats)!=1: raise ValueError('need exactly one calibrated slot')
    row=next(iter(stats.values()))
    if row.get('accounting') not in (base.accounting('separate'),base.accounting('shared')) or row.get('dense_bits',0)<=0:
        raise ValueError('missing hierarchy accounting')
    return dict(compression=1-row['stored_bits']/row['dense_bits'],shortfall_updates=0,stats=stats)


def measure(study,rows,stage):
    # Deliberately dense lookahead tail can undercompress short prompts.
    return measure_candidates(rows,Path(study['out'])/stage,tolerance=.03,budget_rule='absolute',measurement_fn=slot_measurement)


def allocations(study,calibration):
    result=[]
    for arm in study['arms']:
        subset=dict(calibration)
        subset['candidates']=[dict(row,arm=base.ARM) for row in calibration['candidates'] if row['arm']==arm]
        selected=base.select_allocations(study,subset)
        for selection in selected:
            selection.update(arm=arm,tag=f'{arm}_b{round(selection["budget"]*100):02d}')
        result.extend(selected)
    return result


def task_run(study,selections):
    run=base.task_run(study,selections)
    for config,selection in zip(run['configurations'][1:],selections):config['metadata']['arm']=selection['arm']
    return run


def summarize(out,run,*,require_complete=False):
    return base.workflow.summarize(out,run,require_complete=require_complete,
        arm_pairs=(('query','query_attention'),('query','query_shared'),('query_attention','query_attention_shared')),
        scope='query residuals; bounded next-eight attention; shared masks; prefill only',
        comparison='independent layer calibration and measured bytes for each arm')


def calibrate_queries(study):
    """Two fresh dense training prefills; save only small second-moment weights."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.cache_utils import Cache
    from engine.kv_compress.importance import capture_queries
    from engine.eval_runner.text_chunks import wikitext_chunks
    from compression_topics.spatial.scripts.group_rd_offline import parse_model_args
    path=Path(study['query_weights'])
    if path.exists():
        saved=torch.load(path,weights_only=True)
        if saved.get('protocol')!=study: raise ValueError('query weight provenance mismatch')
        return
    name,dtype=parse_model_args(study['model_args'])
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,dtype=dtype).to('cuda').eval()
    prompts,chosen=wikitext_chunks(tokenizer,study['query_chunks'],study['seq_len'],study['seed'],split='train')
    take,uninstall=capture_queries('llama')
    original=Cache.update;sums={};counts={}
    def update(cache,keys,values,layer,*args,**kwargs):
        q=take().detach().float()[...,:study['query_prefix'],:]
        grouped=q.reshape(1,keys.shape[1],-1,q.shape[-2],q.shape[-1])
        sums[layer]=sums.get(layer,0)+grouped.square().sum((0,2,3)).cpu()
        counts[layer]=counts.get(layer,0)+q.shape[-2]
        return original(cache,keys,values,layer,*args,**kwargs)
    Cache.update=update
    try:
        with torch.no_grad():
            for n,prompt in enumerate(prompts):
                model(input_ids=prompt[:study['query_prefix']].unsqueeze(0).to('cuda'),use_cache=True)
                print(f'Query calibration {n+1}/{len(prompts)}',flush=True)
    finally:
        Cache.update=original;uninstall()
    weights=torch.stack([sums[i]/counts[i] for i in range(study['layers'])])
    torch.save(dict(weights=weights,chunks=chosen,protocol=study),path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--arms',choices=ARMS,nargs='+',default=list(ARMS))
    parser.add_argument('--budgets',type=float,nargs='+',default=[.3,.5,.7])
    parser.add_argument('--screen-chunks',type=int,default=2)
    parser.add_argument('--refine-chunks',type=int,default=2)
    parser.add_argument('--validation-chunks',type=int,default=2)
    parser.add_argument('--from',dest='start',choices=base.STAGES,default='screen')
    parser.add_argument('--through',choices=base.STAGES,default='validate')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.budgets or any(b not in base.BUDGETS for b in args.budgets) or min(args.screen_chunks,args.refine_chunks,args.validation_chunks)<2:
        parser.error('budgets 10..80%; at least two chunks per stage')
    if base.STAGES.index(args.start)>base.STAGES.index(args.through):parser.error('invalid stage order')
    out=args.out.resolve();study=plan(out,tuple(sorted(set(args.budgets))),args.screen_chunks,args.refine_chunks,args.validation_chunks)
    study['arms']=list(dict.fromkeys(args.arms))
    path=out/'plan.json'
    if path.exists() and base.workflow.load(path)!=study:parser.error('different protocol; choose fresh output directory')
    write_json(path,study)
    print(f"{len(study['arms'])} arms, each independently calibrated. Query ranking applies to K; next8 importance to K/V. "
          'Final eight-token tail stays dense in attention arms. Default ends after holdout, not full C-Eval.',flush=True)
    if args.execute:
        calibrate_queries(study)
        for stage in base.STAGES[base.STAGES.index(args.start):base.STAGES.index(args.through)+1]:
            print(f'=== {stage} ===',flush=True)
            base.workflow.run_stage(study,stage,candidate_builder=candidates,arms=study['arms'],measure_fn=measure,
                allocation_fn=allocations,task_builder=task_run,summarizer=summarize)

if __name__=='__main__':main()
