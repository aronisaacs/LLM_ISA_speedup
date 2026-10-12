#!/usr/bin/env python3
"""Bounded pair diagnostic, not an accuracy sweep or a replacement runtime method."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from engine.eval_runner.files import write_json


def saved_comparison():
    root=ROOT/'compression_topics/spatial/figures'
    folders={'fixed':'online_fixed_rungs_fixed_residual','adaptive':'clean_online_ladder_pairs'}
    result={'protocol':'pair_diagnostic_v1','saved':{}}
    for name,folder in folders.items():
        p=root/folder
        selections=json.loads((p/'selections.json').read_text())
        validation=json.loads((p/'validation.json').read_text())
        summary=json.loads((p/'summary.json').read_text())
        s=next(r for r in selections if r['budget']==.4)
        result['saved'][name]={'selection':s,'validation':next(r for r in validation if r['budget']==.4),
            'accuracy':next(r for r in summary['rows'] if r.get('global_budget')==.4)}
    return result


def slot_policy(selection,layer,target):
    for step in selection['kv']['pipeline']:
        if layer in step[target+'_layers']:return step['saving'],step.get('residual_entries')
    return 0.,None


def apply_policy(x,target,rope,residual,budget,*,price=None,matched=False):
    import torch
    from compression_topics.spatial.algorithms import windows
    from compression_topics.spatial.algorithms.controller import Online,offline_price
    from compression_topics.spatial.algorithms.formats import HEADER_BITS
    table,recon=windows.build(x,target=target,max_group=2,rope_tables=rope,fixed_residual=residual)
    choices=[];prices=[]
    if matched:
        price=offline_price(table,budget)
    if price is None:
        controller=Online(budget);controller.start(table)
        for w in range(len(table)):
            c,p=controller.decide(table,w);choices.append(c);prices.append(p)
    else:
        for w in range(len(table)):choices.append(table.choose(w,price));prices.append(price)
    out=windows.restore(x,choices,recon)
    changes=[abs(math.log(b/a)) for a,b in zip(prices,prices[1:]) if a>0 and b>0]
    return out,dict(actual_saving=1-(sum(c.bits for c in choices)+HEADER_BITS)/(len(table)*table.window_bits),
        relative_reconstruction_error=float(((out.float()-x.float()).square().sum(-1)/x.float().square().sum(-1).clamp_min(1e-12)).mean()),
        initial_price=prices[0],max_price_ratio=max(prices)/max(prices[0],1e-12),
        max_update_multiplier=math.exp(max(changes,default=0)),mean_local_loss=sum(c.loss for c in choices)/len(choices))


def attention_metrics(q,k,v,changed,target,prefix,projection):
    import torch
    repeat=q.shape[1]//k.shape[1]
    def output(keys,values):
        logits=q.float()@keys.float().repeat_interleave(repeat,1).transpose(-1,-2)/q.shape[-1]**.5
        positions=torch.arange(k.shape[-2],device=k.device)
        logits.masked_fill_(positions[None,:]>torch.arange(prefix,k.shape[-2],device=k.device)[:,None],float('-inf'))
        weights=logits.softmax(-1)
        out=weights@values.float().repeat_interleave(repeat,1)
        flat=out.transpose(1,2).reshape(1,q.shape[-2],-1)
        return logits,out,torch.nn.functional.linear(flat,projection.float())
    dense_logits,dense,projected=output(k,v)
    keys=k.clone();values=v.clone()
    (keys if target=='k' else values)[...,:prefix,:]=changed
    logits,out,post=output(keys,values)
    relative=lambda a,b:float((a-b).square().sum()/b.square().sum().clamp_min(1e-12))
    finite=torch.isfinite(dense_logits)
    return dict(attention_output_relative_error=relative(out,dense),projected_output_relative_error=relative(post,projected),
        logit_mse=float((logits[finite]-dense_logits[finite]).square().mean()))


def capture_states(model,ids,layers,queries):
    import torch
    import transformers.models.llama.modeling_llama as llama
    original=llama.apply_rotary_pos_emb
    captured={};index=[0]
    def rotary(*a,**kw):
        q,k=original(*a,**kw);layer=index[0];index[0]+=1
        if layer in layers:captured[layer]=q[...,-queries:,:].detach().clone()
        return q,k
    llama.apply_rotary_pos_emb=rotary
    try:
        with torch.inference_mode():output=model(input_ids=ids.unsqueeze(0).to(model.device),use_cache=True)
        result={l:(captured[l],output.past_key_values.layers[l].keys.detach().clone(),output.past_key_values.layers[l].values.detach().clone()) for l in layers}
        return result
    finally:llama.apply_rotary_pos_emb=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'compression_topics/spatial/figures/pair_diagnostic')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    report=saved_comparison();report['limitations']=[
        'Captured dense layer states; downstream perplexity is not re-evaluated.',
        'Matched-price arm uses full captured prefix as an analysis-only reference, not a causal runtime policy.',
        'Frozen-price arms may differ in actual bytes; do not interpret them as matched comparisons.',
        'Small validation sample; cannot establish C-Eval accuracy or causal explanation of historical results.']
    allocations=[]
    for layer in range(32):
        for target in ('k','v'):
            fixed=slot_policy(report['saved']['fixed']['selection'],layer,target)
            adaptive=slot_policy(report['saved']['adaptive']['selection'],layer,target)
            allocations.append(dict(layer=layer,target=target,fixed_budget=fixed[0],fixed_residual=fixed[1],adaptive_budget=adaptive[0]))
    report['allocation_comparison']=allocations
    write_json(out/'saved_comparison.json',report)
    if not args.execute:
        print(json.dumps({n:{'holdout_ppl':r['validation']['token_ppl'],'accuracy':r['accuracy']['accuracy']} for n,r in report['saved'].items()},indent=2));return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from engine.eval_runner.text_chunks import wikitext_chunks
    from engine.kv_compress.install import _rope_tables
    from types import SimpleNamespace
    from compression_topics.spatial.algorithms import windows
    from compression_topics.spatial.algorithms.controller import offline_price
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    layers=(0,8,16,24,31);prefix=512;queries=32;seed=91
    report['diagnostic_protocol']=dict(layers=list(layers),prefix=prefix,queries=queries,train_prompts=2,validation_prompts=2,seed=seed,model='meta-llama/Llama-3.1-8B-Instruct',device='cuda',compression='pairs_only_relative_no_query_no_attention')
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(report['diagnostic_protocol']['model'])
    model=AutoModelForCausalLM.from_pretrained(report['diagnostic_protocol']['model'],dtype=torch.bfloat16).to('cuda').eval()
    rope=_rope_tables(SimpleNamespace(model=model))
    capture=lambda ids:capture_states(model,ids[:prefix+queries],layers,queries)
    train,_=wikitext_chunks(tokenizer,2,prefix+queries,seed,split='train')
    validation,chosen=wikitext_chunks(tokenizer,2,prefix+queries,seed,split='validation')
    report['validation_chunk_ids']=chosen
    training=[capture(ids) for ids in train]
    frozen={}
    slots=[r for r in allocations if r['layer'] in layers and 0<r['fixed_budget']<.49]
    for r in slots:
        l,t,b=r['layer'],r['target'],r['fixed_budget']
        x=torch.cat([capture_[l][1 if t=='k' else 2][...,:prefix,:] for capture_ in training],dim=-2)
        for label,residual in [('fixed',r['fixed_residual']),('adaptive',None)]:
            table,_=windows.build(x,target=t,max_group=2,rope_tables=rope,fixed_residual=residual)
            frozen[(l,t,label)]=offline_price(table,b)
    del training
    report['rows']=[]
    for sample,ids in enumerate(validation):
        states=capture(ids)
        for r in slots:
            l,t,b=r['layer'],r['target'],r['fixed_budget'];q,k,v=states[l];x=(k if t=='k' else v)[...,:prefix,:]
            variants=[('fixed_controller',r['fixed_residual'],None,False),('adaptive_controller',None,None,False),
                ('fixed_frozen',r['fixed_residual'],frozen[(l,t,'fixed')],False),('adaptive_frozen',None,frozen[(l,t,'adaptive')],False)]
            fixed_out,fixed_metrics=apply_policy(x,t,rope,r['fixed_residual'],b)
            variants.append(('adaptive_matched_reference',None,None,True))
            for label,residual,price,matched in variants:
                match_budget=fixed_metrics['actual_saving'] if matched else b
                changed,metrics=apply_policy(x,t,rope,residual,match_budget,price=price,matched=matched)
                metrics.update(attention_metrics(q,k,v,changed,t,prefix,model.model.layers[l].self_attn.o_proj.weight))
                metrics.update(sample=sample,layer=l,target=t,variant=label,target_saving=b,fixed_residual=r['fixed_residual'],saving_gap_vs_fixed=metrics['actual_saving']-fixed_metrics['actual_saving'])
                report['rows'].append(metrics)
            report['elapsed_seconds']=time.monotonic()-started
            write_json(out/'preliminary.json',report)
            print(f'[preliminary] prompt {sample+1}/2 layer {l} {t}; {len(report["rows"])} measurements; elapsed {report["elapsed_seconds"]:.1f}s',flush=True)
        del states
    report['status']='complete';write_json(out/'summary.json',report)
    print(f'Complete: {len(report["rows"])} measurements in {time.monotonic()-started:.1f}s',flush=True)
if __name__=='__main__':main()
