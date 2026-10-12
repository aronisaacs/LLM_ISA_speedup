#!/usr/bin/env python3
"""Query-aware online dense-or-quad compression, one 40% accuracy point."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.algorithms.formats import max_saving
from compression_topics.spatial.scripts import study as base,fixed_hierarchy as shared
from engine.eval_runner.files import write_json
ARM='quad_only'
OUT=ROOT/'compression_topics/spatial/figures/quad_only_40'

def plan(out,*,layers=32):
    p=shared.plan(out,layers=layers)
    p.update(protocol='query_online_quad_only_fixed_residual_prefill_v1',arms=[ARM],
        max_group=4,merge_sizes=[4],local_budgets=[.1,.25,.4,.55,.7],residual_combinations=[[0],[16]],
        residual_scope='one count per token per layer K/V slot; heads choose quads independently')
    return p

def candidates(p):
    rows=[]
    for layer in range(p['layers']):
        for target in ('k','v'):
            for residual in (0,16):
                aware=target=='k'
                config=SpatialConfig(max_group=4,merge_sizes=(4,),residual_counts={'4':residual},prefill_only=True,
                    control=p['control'],query_aware=aware,query_weights=p['query_weights'] if aware else None,
                    query_weights_sha256=p['query_weights_sha256'] if aware else None)
                for budget in p['local_budgets']:
                    if budget>=max_saving(p['head_dim'],4,residual):continue
                    rows.append(dict(name=f'quad_L{layer:02d}_{target}_b{round(100*budget):02d}_r{residual}',
                        arm=ARM,layer=layer,target=target,budget=budget,residual_entries=residual,
                        kv=config.pipeline(budget,**base.slot(layer,target))))
    return rows

def allocations(p,cal):return base.allocate(p,cal,ARM)

def tasks(p,selections):
    shared.quality_gate(p)
    return base.task_run(p,selections)

def summarize(out,run,*,require_complete=False):
    report=shared.summarize(out,run,require_complete=require_complete,arm=ARM)
    report['scope']='dense or quad only, fixed residual count per layer K/V, independent head decisions, query-aware keys, prefill-only'
    write_json(Path(out)/('summary.json' if report['status']=='complete' else 'preliminary.json'),report);return report

def run_stage(p,stage):
    if stage=='validate':
        config=base.configuration('reference',shared.reference_selection(p),Path(p['out'])/'validation',base.WIKITEXT_FULL)
        path=Path(p['out'])/'reference_validation_run.json';write_json(path,base.chunk_run(p,[config],'validation'));base.execute_chunks(path)
    result=base.run_stage(p,stage,candidates=candidates,arms=[ARM],allocations=allocations,tasks=tasks,summarizer=summarize)
    if stage=='validate':shared.quality_gate(p)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--from',dest='start',choices=base.STAGES,default='screen')
    parser.add_argument('--through',choices=base.STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();p=plan(args.out.resolve());base.write_plan(p)
    print(f'{len(candidates(p))} candidates: quad-only, fixed 0 or 16 residuals; one 40% point; same controller and query weights.',flush=True)
    if args.execute:base.run_stages(p,args.start,args.through,run_stage)
if __name__=='__main__':main()
