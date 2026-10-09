#!/usr/bin/env python3
"""Evaluate a calibration run on fixed random 2,048-token WikiText chunks.

One model stays loaded across candidates. Outputs token perplexity and per-chunk
mean negative log likelihood, allowing paired comparisons. No model evaluation
starts unless this runner is explicitly invoked by the study's --execute path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.kv_compress import metrics
from engine.eval_runner.text_chunks import wikitext_chunks
from engine.eval_runner.execute import load_model_if_needed
from engine.eval_runner.cache import simulation_identity, identities_match
from engine.eval_runner.files import write_json
from engine.eval_runner.workers import exit_with_parent
from engine.kv_compress import install, parse_kv_spec
from engine.multi_run import visible_cuda_devices


def worker_configurations(configurations, worker, workers):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("invalid worker index/count")
    return [config for index, config in enumerate(configurations) if index % workers == worker]


def spawn_workers(run_path, devices):
    from engine.eval_runner.workers import run_workers
    run_workers(devices, lambda worker, workers: [sys.executable, str(Path(__file__).resolve()),
        "--run", str(run_path), "--worker", str(worker), "--workers", str(workers)])


def partition(prompts, chosen, sampling):
    order = list(range(len(prompts)))
    random.Random(sampling["seed"] + 1).shuffle(order)
    start, stop = sampling["offset"], sampling["offset"] + sampling["chunks"]
    if stop > len(order):
        raise ValueError("sample partition exceeds available chunks")
    indices = order[start:stop]
    return [prompts[i] for i in indices], [chosen[i] for i in indices]


def identity(configuration, sampling, base):
    result = simulation_identity(base, {**configuration, "tasks": ["wikitext_chunks"]},
                                 parse_kv_spec(configuration["kv"]))
    # Direct token loss has no shots, template, generation or lm-eval seed settings.
    for field in ("num_fewshot", "limit", "gen_kwargs", "apply_chat_template", "evaluation_options"):
        result.pop(field, None)
    result["sampling"] = sampling
    return result


def prefill_storage_summary(snapshots):
    targets = {}
    for target in ('k', 'v'):
        dense = sum(r[target]['dense_bits'] for r in snapshots)
        stored = sum(r[target]['stored_bits'] for r in snapshots)
        targets[target] = {'dense_bits': dense, 'stored_bits': stored,
                           'compression': 1 - stored / dense if dense else 0.}
    dense = sum(r['dense_bits'] for r in targets.values())
    stored = sum(r['stored_bits'] for r in targets.values())
    targets['kv'] = {'dense_bits': dense, 'stored_bits': stored, 'compression': 1 - stored / dense if dense else 0.}
    return {'scope': 'simulated_prefill_cache_storage', 'targets': targets}


def evaluate_chunks(model, prompts, kv, chunk_ids, *, scoring_prefix=None, progress_path=None):
    if len(prompts) != len(chunk_ids) or not prompts:
        raise ValueError("nonempty prompts and matching chunk IDs required")
    if scoring_prefix is None and any(step.get('importance') == 'prefill_attention' for step in kv.get('pipeline', [])):
        raise ValueError('prefill importance requires continuation-only scoring, not full-sequence perplexity')
    metrics.reset_stats()
    uninstall = install(SimpleNamespace(model=model), parse_kv_spec(kv))
    scores = []
    prefill_storage = []
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for chunk_id, ids in zip(chunk_ids, prompts):
                batch = ids.unsqueeze(0).to(model.device)
                if scoring_prefix is None:
                    output = model(input_ids=batch, attention_mask=torch.ones_like(batch),
                                   labels=batch, use_cache=True)
                    nll, count = float(output.loss), len(ids) - 1
                else:
                    if not 1 <= scoring_prefix < len(ids):
                        raise ValueError('scoring_prefix must leave a nonempty continuation')
                    prefix, suffix = batch[:, :scoring_prefix], batch[:, scoring_prefix:]
                    before = metrics.storage_summary()
                    output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
                    # Snapshot prefill accounting before dense continuation is appended.
                    after = metrics.storage_summary()
                    if after is None:
                        if kv.get('pipeline'):
                            raise ValueError('missing prefill byte accounting')
                        config = getattr(model.config, 'text_config', model.config)
                        dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
                        bits = prefix.numel() * config.num_hidden_layers * config.num_key_value_heads * dim * 16
                        prefill_storage.append({target: {'dense_bits': bits, 'stored_bits': bits}
                                                for target in ('k', 'v')})
                    else:
                        prefill_storage.append({target: {
                            field: after['targets'][target][field] -
                            (before['targets'][target][field] if before else 0)
                            for field in ('dense_bits', 'stored_bits')}
                            for target in ('k', 'v')})
                    continuation = model(input_ids=suffix, past_key_values=output.past_key_values,
                                         attention_mask=torch.ones_like(batch), use_cache=True)
                    logits = torch.cat((output.logits[:, -1:], continuation.logits[:, :-1]), dim=1)
                    count = suffix.numel()
                    nll = float(torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                                   suffix.reshape(-1)))
                    del continuation, logits
                scores.append({"chunk": chunk_id, "nll": nll, "tokens": count})
                if progress_path is not None:
                    partial = sum(r['nll'] * r['tokens'] for r in scores) / sum(r['tokens'] for r in scores)
                    storage = prefill_storage_summary(prefill_storage) if prefill_storage else metrics.storage_summary()
                    write_json(progress_path, {'status': 'preliminary', 'completed_chunks': len(scores),
                                              'total_chunks': len(prompts), 'token_perplexity': math.exp(partial),
                                              'chunk_scores': scores, 'prefill_storage': storage,
                                              'elapsed_seconds': time.monotonic() - started})
                    print(f'[preliminary] {Path(progress_path).stem}: {len(scores)}/{len(prompts)} chunks; '
                          f'token PPL {math.exp(partial):.5f}', flush=True)
                del output
    finally:
        uninstall()
    tokens = sum(r["tokens"] for r in scores)
    nll = sum(r["nll"] * r["tokens"] for r in scores) / tokens
    storage = metrics.storage_summary()
    result = {"results": {"wikitext_chunks": {"token_perplexity,none": math.exp(nll),
                                            "mean_nll,none": nll}},
            "chunk_scores": scores, "gate_stats": metrics.pop_stats()}
    if storage is not None:
        result['storage'] = storage
    if scoring_prefix is not None:
        result['storage'] = prefill_storage_summary(prefill_storage)
        result['scoring'] = {'prefix_tokens': scoring_prefix, 'definition': 'continuation token NLL only; prefix logits excluded'}
    return result


def combine(prefix, additional):
    scores = prefix["chunk_scores"] + additional["chunk_scores"]
    if len({r["chunk"] for r in scores}) != len(scores):
        raise ValueError("extension contains duplicate chunks")
    total = sum(r["tokens"] for r in scores)
    nll = sum(r["nll"] * r["tokens"] for r in scores) / total
    stats = {}
    for slot in set(prefix["gate_stats"]) | set(additional["gate_stats"]):
        a, b = prefix["gate_stats"].get(slot, {}), additional["gate_stats"].get(slot, {})
        merged = {}
        for key in set(a) | set(b):
            if key in {"features", "kept", "group_size", "accounting"}:
                if a.get(key) != b.get(key):
                    raise ValueError("extension changed representation")
                merged[key] = a[key]
            elif key == "cosine_min":
                merged[key] = min(a.get(key, 1.), b.get(key, 1.))
            else:
                merged[key] = a.get(key, 0) + b.get(key, 0)
        stats[slot] = merged
    return {"results": {"wikitext_chunks": {"token_perplexity,none": math.exp(nll),
                                            "mean_nll,none": nll}},
            "chunk_scores": scores, "gate_stats": stats}


def main():
    exit_with_parent()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.worker is None) != (args.workers is None):
        parser.error("--worker and --workers must be supplied together")
    if args.worker is None:
        devices = visible_cuda_devices()
        if len(devices) > 1:
            print(f"Calibration workers: {len(devices)} GPUs ({','.join(devices)})", flush=True)
            spawn_workers(args.run, devices)
            return
    run = json.loads(args.run.read_text())
    sampling = run["sampling"]
    configs = run["configurations"]
    if args.worker is not None:
        configs = worker_configurations(configs, args.worker, args.workers)
    pending = []
    for configuration in configs:
        path = Path(configuration["output_path"])
        wanted = identity(configuration, sampling, run)
        try:
            stored = json.loads(path.read_text())
            if identities_match(stored.get("simulation", {}), wanted) and stored.get("chunk_scores"):
                continue
        except (OSError, ValueError):
            pass
        pending.append(configuration)
    if not pending:
        print("All sampled calibration results already finished.", flush=True)
        return
    lm = model_key = None
    for index, configuration in enumerate(pending, 1):
        previous_key = model_key
        lm, model_key, device, _ = load_model_if_needed(lm, model_key, run, configuration)
        model = lm.model.eval()
        if previous_key != model_key:
            expected = run.get("model_shape", {})
            config = getattr(model.config, "text_config", model.config)
            actual_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            if expected and (expected["layers"] != config.num_hidden_layers or expected["head_dim"] != actual_dim):
                raise ValueError("planned layer count/head dimension do not match the loaded model")
            prompts, chosen = wikitext_chunks(lm.tokenizer, sampling["pool_chunks"],
                                               sampling["seq_len"], sampling["seed"],
                                               **({'split': sampling['split']} if 'split' in sampling else {}))
            prompts, chosen = partition(prompts, chosen, sampling)
        prefix = None
        if configuration.get("reuse_path"):
            if sampling.get('scoring_prefix') is not None:
                raise ValueError('continuation runs resume finished configurations; chunk-prefix extension is unsupported')
            prefix = json.loads(Path(configuration["reuse_path"]).read_text())
            prefix_sampling = prefix["simulation"]["sampling"]
            expected_prefix = identity(configuration, prefix_sampling, run)
            if not identities_match(prefix["simulation"], expected_prefix):
                raise ValueError("extension model or configuration does not match prefix")
            for key in ("pool_chunks", "offset", "seed", "seq_len"):
                if prefix_sampling[key] != sampling[key]:
                    raise ValueError("extension sampling does not match prefix")
            if prefix_sampling.get('split', 'test') != sampling.get('split', 'test'):
                raise ValueError('extension dataset split differs from prefix')
            if prefix["simulation"]["kv"] != configuration["kv"]:
                raise ValueError("extension KV configuration does not match prefix")
            count = len(prefix["chunk_scores"])
            if [r["chunk"] for r in prefix["chunk_scores"]] != chosen[:count] or count >= len(chosen):
                raise ValueError("extension does not contain the expected chunk prefix")
            additional = evaluate_chunks(model, prompts[count:], configuration["kv"], chosen[count:],
                                         scoring_prefix=sampling.get('scoring_prefix'))
            result = combine(prefix, additional)
        else:
            result = evaluate_chunks(model, prompts, configuration["kv"], chosen,
                                     scoring_prefix=sampling.get('scoring_prefix'),
                                     progress_path=Path(configuration['output_path']).with_suffix('.partial.json')
                                     if run.get('preliminary_results') else None)
        result["simulation"] = identity(configuration, sampling, run)
        path = Path(configuration["output_path"])
        write_json(path, result)
        print(f"[worker {args.worker or 0}, {index}/{len(pending)}] {configuration['name']}: "
              f"token PPL {result['results']['wikitext_chunks']['token_perplexity,none']:.5f}", flush=True)


if __name__ == "__main__":
    main()
