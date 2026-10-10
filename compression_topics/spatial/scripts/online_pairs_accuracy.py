#!/usr/bin/env python3
"""Independently calibrated fixed, online and offline adaptive pair accuracy study.

Common train screen/refinement and validation chunks; full C-Eval 5-shot.
The fixed arm is clean rung 2 (global prefill ranking, per-head merge masks).
The adaptive arm uses one pair format across heads and local feedback. Both
use ordinary residual features, RoPE keys and norm restoration, dense appends.
Offline adaptive searches a price over the full prefill, with the same pair
menu/reconstruction/accounting as online. It is a reference, not an accuracy
optimum. Actual representation metadata is counted separately for each method.
"""
import argparse
from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms import online_pairs as online
from compression_topics.spatial.algorithms import presentation_spatial as fixed
from engine.eval_runner.cache import simulation_identity, identities_match
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration, execute_chunks, perplexity
from engine.layer_select.study import measure_candidates, select_settings, allocate_settings, require_matched_savings

OUT = ROOT / 'compression_topics/spatial/figures/online_pairs_accuracy_offline'
ARMS = ('fixed', 'adaptive', 'offline')
STAGES = ('screen', 'refine', 'allocate', 'validate', 'tasks', 'summary')


def plan(out, *, layers=32, budgets=(.1,.2,.3,.4), screen=4, refine=12, validation=16):
    return {'version': 3, 'out': str(out), 'layers': layers, 'head_dim': 128,
            'model_args': LLAMA31_8B, 'budgets': list(budgets), 'local_budgets': [.1,.25,.4,.47],
            'residuals': [0,8,16,32], 'screen_chunks': screen, 'refine_chunks': refine,
            'validation_chunks': validation, 'seed': 83, 'seq_len': 2048, 'scoring_prefix':1536,
            'control':asdict(online.Control()), 'fixed_accounting':fixed.ACCOUNTING,
            'adaptive_accounting':online.ACCOUNTING,
            'protocol':'independent_train_calibration_shared_samples_validation_gate_full_ceval5shot',
            'matching_tolerance': .01}


def candidates(plan):
    rows=[]
    for arm in ARMS:
        for layer in range(plan['layers']):
            for target in ('k','v'):
                for budget in plan['local_budgets']:
                    for residual in plan['residuals'] if arm=='fixed' else [None]:
                        if arm=='fixed' and budget>fixed.maximum_saving(residual):
                            continue
                        step={'k_layers':[layer] if target=='k' else [],
                              'v_layers':[layer] if target=='v' else [],'saving':budget}
                        if arm=='fixed':
                            step.update(method='presentation_spatial',accounting=fixed.ACCOUNTING,
                                        group_size=2,residual_entries=residual,directional=True,select_by='deviation')
                        else:
                            step.update(method='online_pairs',accounting=online.ACCOUNTING,
                                        feedback=True,fixed_residual=None,
                                        **({'decision':'offline'} if arm=='offline' else {'decision':'online','control':plan['control']}))
                        spec={'pipeline':[step]};parse_kv_spec(spec)
                        tag=f'{arm}_L{layer:02d}_{target}_b{round(100*budget):02d}_r{residual}'
                        rows.append({'name':tag,'arm':arm,'layer':layer,'target':target,
                                     'budget':budget,'residual_entries':residual,'kv':spec})
    return rows


def chunk_run(plan, configs, stage):
    count=plan[stage+'_chunks']
    sample={'split':'validation' if stage=='validation' else 'train',
            'pool_chunks':count if stage=='validation' else plan['screen_chunks']+plan['refine_chunks'],
            'offset':plan['screen_chunks'] if stage=='refine' else 0,'chunks':count,
            'seed':plan['seed'],'seq_len':plan['seq_len'],'scoring_prefix':plan['scoring_prefix']}
    return {'model':'hf','model_args':plan['model_args'],'batch_size':1,
            'model_shape':{'layers':plan['layers'],'head_dim':128},'sampling':sample,
            'preliminary_results':True,'configurations':configs}


def load(path):
    return json.loads(Path(path).read_text())


def measure(plan, rows, stage):
    return measure_candidates(rows, Path(plan['out']) / stage, tolerance=.03, budget_rule='absolute')


def winners(calibration, arm):
    return select_settings(calibration, include=lambda row: row['arm'] == arm)


def select_allocations(plan, calibration):
    selections = []
    for arm in ARMS:
        settings = winners(calibration, arm)
        for budget in plan['budgets']:
            selection = allocate_settings(calibration, settings, plan['layers'], ('k', 'v'),
                                          budget, trim_overshoot=True)
            selection.update(arm=arm, tag=f'{arm}_b{round(100*budget):02d}')
            selections.append(selection)
    return selections


def task_run(plan, selections):
    out=Path(plan['out'])
    configs=[configuration('dense_ceval',{'pipeline':[]},out/'tasks',CEVAL_VALID_5SHOT)]
    configs += [configuration(s['tag']+'_ceval',s['kv'],out/'tasks',CEVAL_VALID_5SHOT,
                              arm=s['arm'],global_budget=s['budget']) for s in selections]
    return {'model':'hf','model_args':plan['model_args'],'batch_size':1,'configurations':configs}


def summarize(out, run, *, require_complete=False):
    rows=[]
    for config in run['configurations']:
        path=Path(config['output_path'])
        if not path.exists():continue
        payload=load(path)
        wanted=simulation_identity(run,config,parse_kv_spec(config['kv']))
        if not identities_match(payload.get('simulation',{}),wanted):raise ValueError(f'result identity differs: {path}')
        count=payload['n-samples']['ceval-valid']['effective']
        if count!=1346:raise ValueError(f'expected full 1346-question C-Eval, found {count}')
        stats=[r for r in payload.get('gate_stats',{}).values() if 'controller_bound_hits' in r]
        rows.append({'name':config['name'],**config.get('metadata',{}),
                     'accuracy':payload['results']['ceval-valid']['acc,none'],'questions':count,
                     'actual_prefill_kv_saving':payload['storage']['targets']['kv']['compression'],
                     'max_local_target_deviation':max((r['max_final_target_deviation'] for r in stats),default=None),
                     'controller_bound_hits':sum(r['controller_bound_hits'] for r in stats)})
    dense=next((r['accuracy'] for r in rows if r['name']=='dense_ceval'),None)
    for row in rows:row['accuracy_change_points']=100*(row['accuracy']-dense) if dense is not None else None
    complete=len(rows)==len(run['configurations'])
    if require_complete and not complete:raise ValueError('accuracy run incomplete')
    comparisons=[]
    for budget in sorted({r['global_budget'] for r in rows if 'global_budget' in r}):
        pair={r['arm']:r for r in rows if r.get('global_budget')==budget}
        for left,right in (('fixed','adaptive'),('adaptive','offline'),('fixed','offline')):
            if left not in pair or right not in pair:continue
            gap=pair[right]['actual_prefill_kv_saving']-pair[left]['actual_prefill_kv_saving']
            comparisons.append({'budget':budget,'reference_arm':left,'comparison_arm':right,
                                'saving_gap_points':100*gap,'matched_within_one_point':abs(gap)<=.01,
                                'accuracy_gain_points':100*(pair[right]['accuracy']-pair[left]['accuracy'])})
    report={'status':'complete' if complete else 'preliminary','completed':len(rows),'total':len(run['configurations']),
            'dense_accuracy':dense,'rows':rows,'comparisons':comparisons,
            'scope':'full C-Eval validation 5-shot; independently calibrated methods; prefill compression only',
            'comparison':'fixed-residual, online adaptive and offline adaptive pairs; same online/offline representations; actual savings checked'}
    write_json(out/('summary.json' if complete else 'preliminary.json'),report)
    return report


def execute_tasks(out, run):
    command=[sys.executable,str(ROOT/'engine/multi_run.py'),'--run',str(out/'tasks_run.json'),
             '--skip-existing','--write-results','--results-root',str(out)]
    process=subprocess.Popen(command,cwd=ROOT);previous=-1
    try:
        while True:
            report=summarize(out,run)
            if report['completed']!=previous:
                print('[preliminary] '+json.dumps(report),flush=True);previous=report['completed']
            try:code=process.wait(timeout=20);break
            except subprocess.TimeoutExpired:pass
        if code:raise subprocess.CalledProcessError(code,command)
        print(json.dumps(summarize(out,run,require_complete=True),indent=2),flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def run_stage(plan,stage):
    out=Path(plan['out'])
    if stage in ('screen','refine','validate'):
        name='validation' if stage=='validate' else stage
        destination=out/name
        configs=[configuration('dense',{'pipeline':[]},destination,WIKITEXT_FULL)]
        rows=candidates(plan)
        if stage=='refine':
            shortlist=load(out/'shortlist.json');rows=[r for r in rows if r['name'] in shortlist]
        if stage=='validate':
            selections=load(out/'selections.json')
            configs += [configuration(s['tag'],s['kv'],destination,WIKITEXT_FULL) for s in selections]
        else:configs += [configuration(r['name'],r['kv'],destination,WIKITEXT_FULL) for r in rows]
        path=out/f'{name}_run.json';write_json(path,chunk_run(plan,configs,name));execute_chunks(path)
        if stage in ('screen','refine'):
            calibration=measure(plan,rows,name);write_json(out/('screening.json' if stage=='screen' else 'calibration.json'),calibration)
            if stage=='screen':
                chosen=[r['name'] for arm in ARMS for r in winners(calibration,arm)]
                write_json(out/'shortlist.json',chosen)
                print(f'Shortlisted {len(chosen)}/{len(rows)} candidates.',flush=True)
        else:
            validation=[]
            for selection in selections:
                payload=load(destination/f"{selection['tag']}.json")
                if payload['simulation']['kv']!=selection['kv']:raise ValueError('validation policy differs')
                validation.append({'arm':selection['arm'],'budget':selection['budget'],
                                   'actual_saving':payload['storage']['targets']['kv']['compression'],
                                   'token_ppl':perplexity(payload)})
            write_json(out/'validation.json',validation)
            try:
                require_matched_savings(validation, plan['budgets'], ARMS, tolerance=plan['matching_tolerance'])
            except ValueError as error:
                raise RuntimeError('holdout byte gate failed; inspect validation.json before full accuracy') from error
            print(json.dumps(validation,indent=2),flush=True)
    elif stage=='allocate':
        write_json(out/'selections.json',select_allocations(plan,load(out/'calibration.json')))
    elif stage=='tasks':
        # Recheck the byte gate even when resuming directly at tasks.
        validation=load(out/'validation.json')
        try:
            require_matched_savings(validation, plan['budgets'], ARMS, tolerance=plan['matching_tolerance'])
        except ValueError as error:
            raise RuntimeError('validation byte gate failed') from error
        run=task_run(plan,load(out/'selections.json'));write_json(out/'tasks_run.json',run);execute_tasks(out,run)
    else:summarize(out,load(out/'tasks_run.json'),require_complete=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--budgets',type=float,nargs='+',default=[.1,.2,.3,.4])
    parser.add_argument('--screen-chunks',type=int,default=4)
    parser.add_argument('--refine-chunks',type=int,default=12)
    parser.add_argument('--validation-chunks',type=int,default=16)
    parser.add_argument('--from',dest='start',choices=STAGES,default='screen')
    parser.add_argument('--through',choices=STAGES,default='summary')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if any(b not in (.1,.2,.3,.4) for b in args.budgets):parser.error('pair-only budgets: 10/20/30/40%, not 50%')
    if min(args.screen_chunks,args.refine_chunks,args.validation_chunks)<2 or STAGES.index(args.start)>STAGES.index(args.through):parser.error('invalid stages/chunk counts')
    out=args.out.resolve();study=plan(out,budgets=tuple(sorted(set(args.budgets))),screen=args.screen_chunks,refine=args.refine_chunks,validation=args.validation_chunks)
    path=out/'plan.json'
    if path.exists() and load(path)!=study:parser.error('different protocol exists; use fresh --out')
    write_json(path,study)
    print(f"Independent calibration: {len(candidates(study))} single-slot candidates x {args.screen_chunks} train chunks; "
          f"refine one finalist per method/slot/budget on {args.refine_chunks} disjoint train chunks. "
          f"{args.validation_chunks} validation chunks; {1+len(ARMS)*len(study['budgets'])} full C-Eval configurations.",flush=True)
    print('All methods: pairs, RoPE keys, norm restoration, no query/token weights, dense decode appends. '
          'Holdout must match savings within one percentage point before accuracy starts.',flush=True)
    if args.execute:
        for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through)+1]:
            print(f'=== {stage} ===',flush=True);run_stage(study,stage)
    else:print('Plan only; submit online_pairs_accuracy_dgx.sh.')

if __name__=='__main__':main()
