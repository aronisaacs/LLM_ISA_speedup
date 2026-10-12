#!/usr/bin/env python3
"""Online fixed-format rungs 1/2, independently calibrated at 10/20/30/40%."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from compression_topics.spatial.scripts import hierarchy_accuracy as base
from compression_topics.spatial.algorithms.composed import SpatialConfig
from compression_topics.spatial.algorithms.storage import vector_group_bits
from engine.eval_runner.files import write_json

ARMS=('no_residual','fixed_residual')
OUT=ROOT/'compression_topics/spatial/figures/online_fixed_rungs'

def plan(out,*,arms=ARMS,layers=32):
    p=base.plan(out,layers=layers,budgets=(.1,.2,.3,.4))
    p.update(version=1,protocol='online_fixed_per_head_rope_norm_eight_window_v1',arms=list(arms),
             max_group=2,local_budgets=[.1,.2,.25,.3,.35,.4,.43,.47,.49],
             residuals=[0,16,32],menus={'2':[0,16,32]},
             fixed_format_scope='one residual count per layer K/V slot; dense always allowed')
    return p

def candidates(p):
    rows=[]
    for arm in p['arms']:
        for layer in range(p['layers']):
            for target in ('k','v'):
                for residual in ([0] if arm=='no_residual' else p['residuals']):
                    ceiling=1-(vector_group_bits(residual,p['head_dim'],2,include_norms=True)+3)/(2*p['head_dim']*16)
                    component=SpatialConfig(representation='hierarchy',max_group=2,residual_entries=residual,control=p['control'])
                    for budget in p['local_budgets']:
                        if budget>=ceiling:continue
                        rows.append(dict(name=f'{arm}_L{layer:02d}_{target}_b{round(100*budget):02d}_r{residual}',
                            arm=arm,layer=layer,target=target,budget=budget,residual_entries=residual,
                            kv=component.pipeline(budget,k_layers=[layer] if target=='k' else [],v_layers=[layer] if target=='v' else [])))
    return rows

def allocations(p,calibration):
    result=[]
    for arm in p['arms']:
        subset=dict(calibration,candidates=[dict(r,arm=base.ARM) for r in calibration['candidates'] if r['arm']==arm])
        for selection in base.select_allocations(p,subset):
            selection.update(arm=arm,tag=f'{arm}_b{round(100*selection["budget"]):02d}')
            result.append(selection)
    return result

def tasks(p,selections):
    run=base.task_run(p,selections)
    for config,selection in zip(run['configurations'][1:],selections):config['metadata']['arm']=selection['arm']
    return run

def summarize(out,run,*,require_complete=False):
    report=base.workflow.summarize(out,run,require_complete=require_complete,
        arm_pairs=(('no_residual','fixed_residual'),),scope='online per-head fixed-format pairs; RoPE and norm restoration; prefill-only; full C-Eval 5-shot',
        comparison='independent train calibration and measured layer allocation; common eight-token controller; no whole-prefill ranking')
    write_json(Path(out)/('summary.json' if report['status']=='complete' else 'preliminary.json'),report)
    return report

def run_stage(p,stage):
    return base.workflow.run_stage(p,stage,candidate_builder=candidates,arms=p['arms'],measure_fn=base.measure,
        allocation_fn=allocations,task_builder=tasks,summarizer=summarize)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--from',dest='start',choices=base.STAGES,default='screen')
    parser.add_argument('--through',choices=base.STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if base.STAGES.index(args.start)>base.STAGES.index(args.through):parser.error('invalid stage order')
    p=plan(args.out.resolve(),arms=tuple(dict.fromkeys(args.arms)));path=Path(p['out'])/'plan.json'
    if path.exists() and base.workflow.load(path)!=p:parser.error('protocol differs; choose fresh --out')
    write_json(path,p)
    print('Online fixed rungs: 10/20/30/40%; independent 4+12 train and 16 holdout calibration; full C-Eval. '+str(len(candidates(p)))+' candidates.',flush=True)
    if args.execute:
        for stage in base.STAGES[base.STAGES.index(args.start):base.STAGES.index(args.through)+1]:
            print(f'=== {stage} ===',flush=True);run_stage(p,stage)
if __name__=='__main__':main()
