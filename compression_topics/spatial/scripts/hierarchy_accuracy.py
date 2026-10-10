#!/usr/bin/env python3
"""Full online pair/quad/eight accuracy, separately calibrated across KV layers.

Separate per-token quad/eight masks, ordinary residual features, RoPE-aligned
keys and stored norm scales. No offline arm, shared masks, query weights,
attention importance or quantization. Compression is prefill-only; decode
appends remain dense. Uses the existing staged accuracy runner and allocator.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT
from compression_topics.spatial.algorithms.composed import SpatialConfig
from compression_topics.spatial.algorithms.hierarchy import MENUS, accounting
from compression_topics.spatial.algorithms.decisions import Control
from compression_topics.spatial.scripts import online_pairs_accuracy as workflow
from engine.eval_runner.cache import simulation_identity, identities_match
from engine.eval_runner.files import write_json
from engine.eval_runner.index import record_simulation
from engine.layer_select.calibration import configuration
from engine.layer_select.study import measure_candidates, select_settings, allocate_settings

OUT = ROOT/'compression_topics/spatial/figures/hierarchy_accuracy'
ARM = 'online_hierarchy'
BUDGETS = tuple(n/10 for n in range(1,9))
STAGES = workflow.STAGES


def plan(out, *, layers=32, budgets=BUDGETS, screen=4, refine=12, validation=16):
    return dict(version=1, out=str(out), layers=layers, head_dim=128, model_args=LLAMA31_8B,
                budgets=list(budgets), local_budgets=[.1,.25,.4,.55,.7,.8,.85],
                screen_chunks=screen, refine_chunks=refine, validation_chunks=validation,
                seed=83, seq_len=2048, scoring_prefix=1536, control=asdict(Control()),
                representation='hierarchy', max_group=8, mask_layout='separate',
                menus={str(g):list(menu) for g,menu in MENUS.items()},
                accounting=accounting('separate'), allocation_basis='measured_calibration_bytes',
                protocol='own_train_calibration_measured_slot_allocation_validation_gate_full_ceval5shot_v1',
                matching_tolerance=.01,
                dense_ceval_source=str(ROOT/'compression_topics/spatial/figures/online_pairs_accuracy_offline/tasks/dense_ceval.json'))


def candidates(study):
    component = SpatialConfig(representation='hierarchy',max_group=8,mask_layout='separate',control=study['control'])
    rows=[]
    for layer in range(study['layers']):
        for target in ('k','v'):
            for budget in study['local_budgets']:
                kv = component.pipeline(budget,k_layers=[layer] if target=='k' else [],
                                        v_layers=[layer] if target=='v' else [])
                rows.append(dict(name=f'online_g8_L{layer:02d}_{target}_b{round(100*budget):02d}',
                                 arm=ARM,layer=layer,target=target,budget=budget,kv=kv))
    return rows


def slot_measurement(payload):
    """Byte-only calibration contract; no invented legacy pair/cosine metrics."""
    stats = {key:value for key,value in payload.get('gate_stats',{}).items() if '_layer_' in key}
    if len(stats)!=1:
        raise ValueError('hierarchy calibration must measure exactly one KV slot')
    row=next(iter(stats.values()))
    if row.get('accounting')!=accounting('separate') or row.get('dense_bits',0)<=0:
        raise ValueError('missing hierarchy byte accounting')
    return dict(compression=1-row['stored_bits']/row['dense_bits'],shortfall_updates=0,stats=stats)


def measure(study, rows, stage):
    return measure_candidates(rows,Path(study['out'])/stage,tolerance=.03,budget_rule='absolute',
                              measurement_fn=slot_measurement)


def select_allocations(study, calibration):
    settings=select_settings(calibration,include=lambda row:row['arm']==ARM)
    # Different requested settings can occasionally produce exactly equal bytes.
    # Keep the better measured quality at that rate before building the curve.
    rates={}
    for row in settings:
        key=(row['layer'],row['target'],row['measured_saving'])
        if key not in rates or row['ppl']<rates[key]['ppl']:rates[key]=row
    selections=[]
    for budget in study['budgets']:
        selection=allocate_settings(calibration,list(rates.values()),study['layers'],('k','v'),budget,
                                    allocation_basis='measured')
        selection.update(arm=ARM,tag=f'online_g8_b{round(100*budget):02d}')
        selections.append(selection)
    return selections


def task_run(study,selections):
    out=Path(study['out'])
    configs=[configuration('dense_ceval',{'pipeline':[]},out/'tasks',CEVAL_VALID_5SHOT)]
    configs += [configuration(s['tag']+'_ceval',s['kv'],out/'tasks',CEVAL_VALID_5SHOT,
                              arm=ARM,global_budget=s['budget']) for s in selections]
    run=dict(model='hf',model_args=study['model_args'],batch_size=1,configurations=configs)
    # Reuse only a proven-identical full dense evaluation; compressed methods
    # always use their own new calibration and policy identities.
    source=Path(study['dense_ceval_source'])
    if source.exists():
        payload=workflow.load(source)
        wanted=simulation_identity(run,configs[0],workflow.parse_kv_spec(configs[0]['kv']))
        if identities_match(payload.get('simulation',{}),wanted) and payload.get('n-samples',{}).get('ceval-valid',{}).get('effective')==1346:
            write_json(configs[0]['output_path'],payload)
            record_simulation(wanted,payload['results'],payload['n-samples'],root=out)
            print('Reusing identity-checked full dense C-Eval baseline.',flush=True)
    return run


def summarize(out,run,*,require_complete=False):
    report=workflow.summarize(out,run,require_complete=require_complete,arm_pairs=(),
        scope='full C-Eval validation 5-shot; own calibrated online pair/quad/eight hierarchy; prefill only',
        comparison='one online method with separate masks; measured layer allocation; no offline/shared-mask arms')
    for row in report['rows']:
        if 'global_budget' in row:
            row['saving_deviation_points']=100*(row['actual_prefill_kv_saving']-row['global_budget'])
            row['within_one_point']=abs(row['saving_deviation_points'])<=1
    write_json(Path(out)/('summary.json' if report['status']=='complete' else 'preliminary.json'),report)
    return report


def run_stage(study,stage):
    return workflow.run_stage(study,stage,candidate_builder=candidates,arms=(ARM,),measure_fn=measure,
                              allocation_fn=select_allocations,task_builder=task_run,summarizer=summarize)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--budgets',type=float,nargs='+',default=list(BUDGETS))
    parser.add_argument('--screen-chunks',type=int,default=4)
    parser.add_argument('--refine-chunks',type=int,default=12)
    parser.add_argument('--validation-chunks',type=int,default=16)
    parser.add_argument('--from',dest='start',choices=STAGES,default='screen')
    parser.add_argument('--through',choices=STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if any(b not in BUDGETS for b in args.budgets) or not args.budgets:
        parser.error('supported global budgets are 10..80%, step 10')
    if min(args.screen_chunks,args.refine_chunks,args.validation_chunks)<2 or STAGES.index(args.start)>STAGES.index(args.through):
        parser.error('invalid stages/chunk counts')
    out=args.out.resolve()
    study=plan(out,budgets=tuple(sorted(set(args.budgets))),screen=args.screen_chunks,
               refine=args.refine_chunks,validation=args.validation_chunks)
    path=out/'plan.json'
    if path.exists() and workflow.load(path)!=study:parser.error('different protocol exists; choose fresh --out')
    write_json(path,study)
    print(f"{len(candidates(study))} single-slot target settings; {args.screen_chunks}+{args.refine_chunks} "
          f"disjoint train chunks; {args.validation_chunks} holdout chunks; {len(study['budgets'])} full C-Eval compressed policies.",flush=True)
    print('Online pairs/quads/eights, separate token masks, own K/V layer calibration. '
          'Measured calibration bytes drive allocation; holdout must be within 1 point of target. '
          'Preliminary results and timing records are saved. Decode appends remain dense.',flush=True)
    if args.execute:
        for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through)+1]:
            print(f'=== {stage} ===',flush=True);run_stage(study,stage)
    else:print('Plan only; submit hierarchy_accuracy_dgx.sh.')


if __name__=='__main__':main()
