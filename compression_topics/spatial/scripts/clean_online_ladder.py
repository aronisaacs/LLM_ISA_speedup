#!/usr/bin/env python3
"""Per-head online ladder, 30/40/50%: pairs, hierarchy, query-aware hierarchy.

Separate masks; no token importance. Pair-only 50% is explicitly the smallest
feasible format endpoint, not an achieved 50% budget or a controller ablation.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from compression_topics.spatial.scripts import hierarchy_accuracy as base,hierarchy_weighted as weighted
from compression_topics.spatial.algorithms.composed import SpatialConfig
from engine.layer_select.study import select_settings,require_matched_savings
from engine.eval_runner.files import write_json

ARMS=('pairs','hierarchy','query_hierarchy')
BUDGETS=(.3,.4,.5)
OUT=ROOT/'compression_topics/spatial/figures/clean_online_ladder'


def plan(out,*,arms=ARMS,layers=32,screen=4,refine=12,validation=16):
    p=base.plan(out,layers=layers,budgets=BUDGETS,screen=screen,refine=refine,validation=validation)
    p.update(version=1,protocol='per_head_clean_ladder_absolute_query_no_attention_separate_masks_v1',
        arms=list(arms),pair_local_budgets=[.1,.25,.4,.47,.49],
        local_budgets=[.1,.25,.4,.47,.49,.55,.7,.8,.85],
        query_weights=str(out/'query_weights.pt'),query_chunks=2,query_prefix=1536,query_split='train',
        pair_50='minimum_storage_formats_per_head_not_exact_budget',
        pair_menu=[0,16,32],key_query_objective='absolute_diagonal_query_weighted_squared_error')
    return p


def candidates(study):
    import hashlib
    rows=[]
    query='query_hierarchy' in study['arms']
    path=Path(study['query_weights'])
    if query and not path.exists():raise ValueError('fresh query calibration required')
    digest=hashlib.sha256(path.read_bytes()).hexdigest() if query else None
    for arm in study['arms']:
        for layer in range(study['layers']):
            for target in ('k','v'):
                budgets=study['pair_local_budgets']+[.5] if arm=='pairs' else study['local_budgets']
                for budget in budgets:
                    aware=arm=='query_hierarchy' and target=='k'
                    endpoint=arm=='pairs' and budget==.5
                    component=SpatialConfig(representation='hierarchy',max_group=2 if arm=='pairs' else 8,
                        control=study['control'],head_decisions='per_head',mask_layout='separate',importance='none',
                        select_by='query' if aware else 'deviation',distortion='query' if aware else 'relative',
                        query_weights=str(path) if aware else None,query_weights_sha256=digest if aware else None,
                        saturate=endpoint)
                    rows.append(dict(name=f'{arm}_L{layer:02d}_{target}_b{round(100*budget):02d}',
                        arm=arm,layer=layer,target=target,budget=budget,maximum_feasible=endpoint,
                        kv=component.pipeline(budget,k_layers=[layer] if target=='k' else [],v_layers=[layer] if target=='v' else [])))
    return rows


def allocations(study,calibration):
    result=[]
    for arm in study['arms']:
        regular=[dict(r,arm=base.ARM) for r in calibration['candidates'] if r['arm']==arm and not (arm=='pairs' and r['budget']==.5)]
        sub=dict(calibration,candidates=regular);p=dict(study,budgets=[.3,.4] if arm=='pairs' else study['budgets'])
        for selection in base.select_allocations(p,sub):
            selection.update(arm=arm,tag=f'{arm}_b{round(100*selection["budget"]):02d}',maximum_feasible=False)
            result.append(selection)
        if arm=='pairs':
            settings=select_settings(calibration,include=lambda r:r['arm']=='pairs' and r['budget']==.5)
            if len(settings)!=study['layers']*2:raise ValueError('missing pair endpoint calibration')
            pipeline=[step for r in settings for step in r['kv']['pipeline']]
            result.append(dict(arm=arm,budget=.5,tag='pairs_b50_maximum_feasible',maximum_feasible=True,
                kv={'pipeline':pipeline},assignment=settings,compression=sum(r['measured_saving'] for r in settings)/len(settings),
                allocation_basis='measured_calibration_minimum_storage_endpoint'))
    return result


def check_validation(rows,budgets,arms,*,tolerance):
    if len(rows)!=len(budgets)*len(arms):raise ValueError('missing or duplicate validation rows')
    for budget in budgets:
        eligible=[a for a in arms if not(a=='pairs' and budget==.5)]
        if eligible:require_matched_savings([r for r in rows if r['arm'] in eligible],[budget],eligible,tolerance=tolerance)
    if 'pairs' in arms:
        endpoint=[r for r in rows if r['arm']=='pairs' and r['budget']==.5]
        # BF16 D=128: one mean + two scales + one 3-bit tag per head/pair.
        ceiling=1-(16*(128+2)+3)/(2*128*16)
        if len(endpoint)!=1 or not ceiling-tolerance<=endpoint[0]['actual_saving']<=ceiling+1e-8:
            raise ValueError('pair maximum endpoint differs from physical byte ceiling')


def tasks(study,selections):
    run=base.task_run(study,selections)
    for config,selection in zip(run['configurations'][1:],selections):
        config['metadata'].update(arm=selection['arm'],maximum_feasible=selection['maximum_feasible'],
                                  operating_point='maximum_feasible_pairs' if selection['maximum_feasible'] else 'calibrated_budget')
    return run


def summarize(out,run,*,require_complete=False):
    report=base.workflow.summarize(out,run,require_complete=require_complete,
        arm_pairs=(('pairs','hierarchy'),('hierarchy','query_hierarchy')),
        scope='per-head online clean ladder; separate masks; no token importance; prefill-only; full C-Eval 5-shot',
        comparison='independent matched training calibration; pair 50% is maximum feasible endpoint, not equal-budget comparison')
    for row in report['rows']:
        if 'global_budget' in row:row['saving_deviation_points']=100*(row['actual_prefill_kv_saving']-row['global_budget'])
    for comparison in report['comparisons']:
        if comparison['budget']==.5 and comparison['reference_arm']=='pairs':
            comparison.update(matched_within_one_point=False,maximum_feasible_reference=True,
                              caveat='pair endpoint below 50%; not a matched-budget comparison')
    write_json(Path(out)/('summary.json' if report['status']=='complete' else 'preliminary.json'),report)
    return report


def run_stage(study,stage):
    return base.workflow.run_stage(study,stage,candidate_builder=candidates,arms=study['arms'],measure_fn=base.measure,
        allocation_fn=allocations,task_builder=tasks,summarizer=summarize,validation_checker=check_validation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--arms',choices=ARMS,nargs='+',default=list(ARMS))
    parser.add_argument('--from',dest='start',choices=base.STAGES,default='screen')
    parser.add_argument('--through',choices=base.STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if base.STAGES.index(args.start)>base.STAGES.index(args.through):parser.error('invalid stage order')
    out=args.out.resolve();study=plan(out,arms=tuple(dict.fromkeys(args.arms)))
    path=out/'plan.json'
    if path.exists() and base.workflow.load(path)!=study:parser.error('protocol differs; choose fresh --out')
    write_json(path,study)
    print('Clean per-head ladder, 30/40/50%, 4+12 training and 16 holdout chunks, full C-Eval. '
          'Pair 50% is a labeled minimum-storage endpoint (~49.15% ceiling), not exact 50%. '
          'Common pair menu/controller cadence; no attention or shared masks.',flush=True)
    if args.execute:
        if 'query_hierarchy' in study['arms']:weighted.calibrate_queries(study)
        for stage in base.STAGES[base.STAGES.index(args.start):base.STAGES.index(args.through)+1]:
            print(f'=== {stage} ===',flush=True);run_stage(study,stage)

if __name__=='__main__':main()
