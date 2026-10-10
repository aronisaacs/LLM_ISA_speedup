#!/usr/bin/env python3
"""Small sequential pair-controller test on existing DGX vectors; no accuracy run.

Use saved rung-2 local budgets/residuals at global 20/30/40% targets. Decisions
apply across heads, with head-local representations. Compare fixed residuals,
adaptive fixed price, and adaptive feedback; all share reconstruction/accounting.
The initial 32 tokens are fitted jointly; later rows are processed in order.
"""
import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from compression_topics.spatial.algorithms import online_pairs as online
from compression_topics.spatial.scripts.group_rd_offline import load_rope
from engine.eval_runner.files import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-plan',type=Path,default=ROOT/'compression_topics/spatial/figures/group_rd_offline/plan.json')
    parser.add_argument('--selections',type=Path,default=ROOT/'compression_topics/spatial/figures/presentation_ladder_clean_50/selections.json')
    parser.add_argument('--out',type=Path,default=ROOT/'compression_topics/spatial/figures/online_pairs_startup_preserve')
    parser.add_argument('--layers',type=int,nargs='+',default=[4,12,20,28])
    parser.add_argument('--global-budgets',type=float,nargs='+',default=[.2,.3,.4])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--memory-pairs',type=float,default=10.)
    parser.add_argument('--recent-gain',type=float,default=4.)
    parser.add_argument('--cumulative-gain',type=float,default=8.)
    parser.add_argument('--max-log-step',type=float,default=.2)
    args=parser.parse_args()
    control=online.Control(memory_pairs=args.memory_pairs,recent_gain=args.recent_gain,
                           cumulative_gain=args.cumulative_gain,max_log_step=args.max_log_step)
    capture=json.loads(args.capture_plan.read_text())
    files=sorted(Path(capture['keys_dir']).glob('chunk_*.pt'))[:2]
    if len(files)!=2:parser.error('need two existing captured chunks; no capture is launched automatically')
    choices=json.loads(args.selections.read_text())
    selected={r['budget']:r for r in choices if r['rung']==2}
    if any(b not in selected for b in args.global_budgets):parser.error('missing saved rung-2 budget')
    rope=load_rope(capture['rope'])
    plan={'accounting':online.ACCOUNTING,'control':asdict(control),'capture_files':[str(f) for f in files],
          'selections':str(args.selections),'layers':args.layers,'global_budgets':args.global_budgets,
          'scope':'sampled slots at saved layer budgets; not a whole-model accuracy or global-allocation test',
          'arms':['fixed_residual','adaptive_fixed_price','adaptive_feedback'],
          'decision':'one dense/0/8/16/32-residual format across KV heads; head-local residuals; no query or importance weighting',
          'error':'mean per-token relative squared error across heads; bf16 means/residuals/restoration scales',
          'feedback':'v3: one-pair startup slack only for undercompression; overcompression bands unchanged; startup selects at/below target saving',
          'startup':'first 32 tokens price search only; subsequent decisions see current pair and controller state'}
    path=args.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:parser.error('different plan exists; use new --out')
    write_json(path,plan)
    started=time.monotonic();rows=[];skipped=[]
    for file in files:
        data=torch.load(file,map_location='cpu',weights_only=True)
        for layer in args.layers:
            for target in ('k','v'):
                source=data['keys' if target=='k' else 'values'][layer].unsqueeze(0).to(args.device)
                table=online.pair_table(source,target=target,rope_tables=rope if target=='k' else None)
                for budget in args.global_budgets:
                    assignment=next((r for r in selected[budget]['assignment'] if r['layer']==layer and r['target']==target),None)
                    if assignment is None:
                        skipped.append({'chunk':data['chunk'],'layer':layer,'target':target,'global_budget':budget,'reason':'saved allocation leaves this slot dense'})
                        continue
                    local=assignment['budget'];residual=assignment['residual_entries']
                    for arm in plan['arms']:
                        result=online.run(table,local,feedback=arm=='adaptive_feedback',
                                          fixed_residual=residual if arm=='fixed_residual' else None,control=control)
                        tag=f"chunk{data['chunk']}_L{layer:02d}_{target}_b{round(100*budget):02d}_{arm}"
                        write_json(args.out/'traces'/f'{tag}.json',result)
                        row={k:v for k,v in result.items() if k!='history'}
                        row.update(chunk=data['chunk'],layer=layer,target=target,global_budget=budget,
                                   local_budget=local,arm=arm,fixed_residual=residual if arm=='fixed_residual' else None,
                                   trace=str(args.out/'traces'/f'{tag}.json'))
                        rows.append(row)
                        print(f"[{tag}] target {local:.1%}; actual {row['measured_saving']:.2%}; error {row['mean_relative_squared_error']:.6f}; "
                              f"time in +/-3pt band {row['fraction_post_startup_within_three_points']}; price bound hits {row['price_bound_hits']}",flush=True)
                        if arm=='adaptive_feedback':
                            for checkpoint in result['early_checkpoints']:
                                print(f"  [early {checkpoint['tokens']} tokens] saving {checkpoint['cumulative_saving']:.2%}; "
                                      f"allowed saving {checkpoint['allowed_saving_min']:.2%}–{checkpoint['allowed_saving_max']:.2%}; price {checkpoint['price_used']:.6g}",flush=True)
                    write_json(args.out/'results.partial.json',{'status':'preliminary','plan':plan,'rows':rows,'skipped':skipped,
                                                               'elapsed_seconds':time.monotonic()-started})
                del table,source
    write_json(args.out/'results.json',{'status':'complete','plan':plan,'rows':rows,'skipped':skipped,
                                     'elapsed_seconds':time.monotonic()-started})
    print(f"Finished in {time.monotonic()-started:.1f}s. No model or accuracy evaluation. Results: {args.out}",flush=True)

if __name__=='__main__':main()
