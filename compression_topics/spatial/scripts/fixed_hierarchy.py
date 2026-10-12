#!/usr/bin/env python3
"""40% query-aware hierarchy with calibrated fixed residuals per group size."""
import argparse
import hashlib
from itertools import product
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.algorithms.formats import MENUS,max_saving
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json
ARM='fixed_hierarchy'
REFERENCE=ROOT/'compression_topics/spatial/figures/clean_online_ladder_query_hierarchy'
OUT=ROOT/'compression_topics/spatial/figures/fixed_hierarchy_40'

def plan(out,*,layers=32):
    p=base.base_plan(out,layers=layers,budgets=(.4,),screen=2,refine=12,validation=16)
    p.update(version=1,protocol='query_fixed_group_residuals_prefill_only_v1',arms=[ARM],
        local_budgets=[.1,.25,.4,.55,.7,.8,.85],query_weights=str(REFERENCE/'query_weights.pt'),
        query_weights_sha256=hashlib.sha256((REFERENCE/'query_weights.pt').read_bytes()).hexdigest(),
        residual_combinations=[list(r) for r in product(MENUS[2],MENUS[4],MENUS[8])],
        reference=str(REFERENCE),holdout_ppl_ratio_limit=1.02,accuracy_tolerance_points=1.,prefill_only=True)
    return p

def candidates(p):
    rows=[]
    for layer in range(p['layers']):
        for target in ('k','v'):
            for combo in p['residual_combinations']:
                counts=dict(zip(('2','4','8'),combo));ceiling=max(max_saving(p['head_dim'],g,counts[str(g)]) for g in MENUS)
                aware=target=='k'
                config=SpatialConfig(max_group=8,residual_counts=counts,prefill_only=True,control=p['control'],query_aware=aware,
                    query_weights=p['query_weights'] if aware else None,query_weights_sha256=p['query_weights_sha256'] if aware else None)
                for budget in p['local_budgets']:
                    if budget>=ceiling:continue
                    rows.append(dict(name=f'fixed_L{layer:02d}_{target}_b{round(100*budget):02d}_r'+ '_'.join(map(str,combo)),
                        arm=ARM,layer=layer,target=target,budget=budget,residual_counts=counts,
                        kv=config.pipeline(budget,**base.slot(layer,target))))
    return rows

def reference_selection(p):
    saved=next(s for s in base.load(Path(p['reference'])/'selections.json') if s['budget']==.4)
    steps=[]
    for old in saved['kv']['pipeline']:
        aware=bool(old['k_layers'])
        config=SpatialConfig(max_group=8,prefill_only=True,control=old['control'],query_aware=aware,
            query_weights=p['query_weights'] if aware else None,query_weights_sha256=p['query_weights_sha256'] if aware else None)
        steps+=config.pipeline(old['saving'],k_layers=old['k_layers'],v_layers=old['v_layers'])['pipeline']
    return {'pipeline':steps}

def allocations(p,cal):return base.allocate(p,cal,ARM)

def check_validation(rows,budgets,arms,*,tolerance):
    from engine.layer_select.study import require_matched_savings
    require_matched_savings(rows,budgets,arms,tolerance=tolerance)

def quality_gate(p):
    out=Path(p['out']);ref=base.load(out/'validation/reference.json');fixed=base.load(out/f"validation/{p['arms'][0]}_b40.json")
    a=base.perplexity(ref);b=base.perplexity(fixed)
    saving_ref=ref['storage']['targets']['kv']['compression'];saving_fixed=fixed['storage']['targets']['kv']['compression']
    passed=abs(saving_ref-saving_fixed)<=p['matching_tolerance'] and b<=a*p['holdout_ppl_ratio_limit']
    gate=dict(passed=passed,reference_ppl=a,fixed_ppl=b,ppl_ratio=b/a,reference_saving=saving_ref,fixed_saving=saving_fixed,
        definition='heuristic 2% PPL screen and 1-point byte match; not an accuracy guarantee')
    write_json(out/'quality_gate.json',gate);print(gate,flush=True)
    if not passed:raise RuntimeError('Holdout quality/byte gate failed; full C-Eval not started. Inspect quality_gate.json.')

def tasks(p,selections):
    quality_gate(p)
    return base.task_run(p,selections)

def summarize(out,run,*,require_complete=False,arm=ARM):
    report=base.summarize(out,run,require_complete=require_complete,scope='fixed residuals by group size per layer K/V; query-aware; prefill-only',
        comparison='historical adaptive reference plus reevaluated common holdout; 2 screening versus reference 4, both 12 refinement')
    old=base.load(REFERENCE/'summary.json');ref=next(r for r in old['rows'] if r.get('global_budget')==.4)
    report['historical_reference']=ref
    for r in report['rows']:
        if r.get('arm')==arm:
            report['reference_comparison']=dict(accuracy_difference_points=100*(r['accuracy']-ref['accuracy']),
                saving_difference_points=100*(r['actual_kv_saving']-ref['actual_prefill_kv_saving']),
                within_prespecified_accuracy_tolerance=r['accuracy']>=ref['accuracy']-.01,
                note='interpret accuracy only alongside actual compression; historical reference, not statistical equivalence')
    write_json(Path(out)/('summary.json' if report['status']=='complete' else 'preliminary.json'),report);return report

def run_stage(p,stage):
    if stage=='validate':
        config=base.configuration('reference',reference_selection(p),Path(p['out'])/'validation',base.WIKITEXT_FULL)
        path=Path(p['out'])/'reference_validation_run.json';write_json(path,base.chunk_run(p,[config],'validation'));base.execute_chunks(path)
    result=base.run_stage(p,stage,candidates=candidates,arms=[ARM],allocations=allocations,tasks=tasks,summarizer=summarize,validation_checker=check_validation)
    if stage=='validate':quality_gate(p)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--from',dest='start',choices=base.STAGES,default='screen')
    parser.add_argument('--through',choices=base.STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();p=plan(args.out.resolve());base.write_plan(p)
    print(f'{len(candidates(p))} screening candidates; 12 residual combinations; one 40% C-Eval point after holdout gate.',flush=True)
    if args.execute:base.run_stages(p,args.start,args.through,run_stage)
if __name__=='__main__':main()
