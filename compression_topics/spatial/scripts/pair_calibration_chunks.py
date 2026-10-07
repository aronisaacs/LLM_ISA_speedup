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
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms import pair_gate
from compression_topics.spatial.scripts.pair_similarity_profile import wikitext_chunks, PRETRAINED
from engine.kv_compress import install, parse_kv_spec
from engine.multi_run import visible_cuda_devices


def worker_configurations(configurations, worker, workers):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("invalid worker index/count")
    return [config for index, config in enumerate(configurations) if index % workers == worker]


def spawn_workers(run_path, devices):
    processes = []
    try:
        for worker, device in enumerate(devices):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            command = [sys.executable, str(Path(__file__).resolve()), "--run", str(run_path),
                       "--worker", str(worker), "--workers", str(len(devices))]
            processes.append(subprocess.Popen(command, env=env))
        while any(process.poll() is None for process in processes):
            failed = next((process for process in processes if process.poll() not in (None, 0)), None)
            if failed is not None:
                raise subprocess.CalledProcessError(failed.returncode, failed.args)
            time.sleep(.5)
        for process in processes:
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def partition(prompts, chosen, sampling):
    order = list(range(len(prompts)))
    random.Random(sampling["seed"] + 1).shuffle(order)
    start, stop = sampling["offset"], sampling["offset"] + sampling["chunks"]
    if stop > len(order):
        raise ValueError("sample partition exceeds available chunks")
    indices = order[start:stop]
    return [prompts[i] for i in indices], [chosen[i] for i in indices]


def identity(configuration, sampling):
    return {"pretrained": PRETRAINED, "dtype": "bfloat16", "tasks": ["wikitext_chunks"],
            "sampling": sampling, "kv": configuration["kv"]}


def evaluate_chunks(model, prompts, kv, chunk_ids):
    if len(prompts) != len(chunk_ids) or not prompts:
        raise ValueError("nonempty prompts and matching chunk IDs required")
    pair_gate.reset_stats()
    uninstall = install(SimpleNamespace(model=model), parse_kv_spec(kv))
    scores = []
    try:
        with torch.inference_mode():
            for chunk_id, ids in zip(chunk_ids, prompts):
                batch = ids.unsqueeze(0).to(model.device)
                output = model(input_ids=batch, attention_mask=torch.ones_like(batch),
                               labels=batch, use_cache=True)
                scores.append({"chunk": chunk_id, "nll": float(output.loss), "tokens": len(ids) - 1})
                del output
    finally:
        uninstall()
    tokens = sum(r["tokens"] for r in scores)
    nll = sum(r["nll"] * r["tokens"] for r in scores) / tokens
    return {"results": {"wikitext_chunks": {"token_perplexity,none": math.exp(nll),
                                            "mean_nll,none": nll}},
            "chunk_scores": scores, "gate_stats": pair_gate.pop_stats()}


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
            if key in {"features", "kept"}:
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
        wanted = identity(configuration, sampling)
        try:
            stored = json.loads(path.read_text())
            if stored.get("simulation") == wanted and stored.get("chunk_scores"):
                continue
        except (OSError, ValueError):
            pass
        pending.append(configuration)
    if not pending:
        print("All sampled calibration results already finished.", flush=True)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    # Draw the whole partition once; slicing creates disjoint stage subsets.
    prompts, chosen = wikitext_chunks(tokenizer, sampling["pool_chunks"], sampling["seq_len"], sampling["seed"])
    prompts, chosen = partition(prompts, chosen, sampling)
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED, dtype=torch.bfloat16).to("cuda").eval()
    for index, configuration in enumerate(pending, 1):
        prefix = None
        if configuration.get("reuse_path"):
            prefix = json.loads(Path(configuration["reuse_path"]).read_text())
            prefix_sampling = prefix["simulation"]["sampling"]
            for key in ("pool_chunks", "offset", "seed", "seq_len"):
                if prefix_sampling[key] != sampling[key]:
                    raise ValueError("extension sampling does not match prefix")
            if prefix["simulation"]["kv"] != configuration["kv"]:
                raise ValueError("extension KV configuration does not match prefix")
            count = len(prefix["chunk_scores"])
            if [r["chunk"] for r in prefix["chunk_scores"]] != chosen[:count] or count >= len(chosen):
                raise ValueError("extension does not contain the expected chunk prefix")
            additional = evaluate_chunks(model, prompts[count:], configuration["kv"], chosen[count:])
            result = combine(prefix, additional)
        else:
            result = evaluate_chunks(model, prompts, configuration["kv"], chosen)
        result["simulation"] = identity(configuration, sampling)
        path = Path(configuration["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=1) + "\n")
        temporary.replace(path)
        print(f"[worker {args.worker or 0}, {index}/{len(pending)}] {configuration['name']}: "
              f"token PPL {result['results']['wikitext_chunks']['token_perplexity,none']:.5f}", flush=True)


if __name__ == "__main__":
    main()
