#!/usr/bin/env python3
"""Check, offline and on the CPU, that everything the KL greedy rerun needs is present.

Run it on the DGX login node before booking GPUs (a minute or two; nothing is downloaded):

  python compression_topics/kl_greedy/scripts/preflight.py
  python compression_topics/kl_greedy/scripts/preflight.py --download   # fetch what is missing first

Checks: the three models (every weight shard and the tokenizer in the Hugging Face cache), the task
datasets (all 52 C-Eval subjects, GSM8K, HumanEval), HumanEval's code_eval metric, the lm-eval task
names, the spatial query weights against each model's shape, the KL calibration sets (and, where one
is still to be built, the datasets that build it), the library versions and the GPUs.
Exit status 1 when anything needed is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_DATASETS = [("openai/gsm8k", "main", "test"), ("openai/openai_humaneval", None, "test")]
CALIBRATION_DATASETS = [  # build_calibration_set.SOURCES, loaded only when a set must be built
    ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", "validation"),
    ("allenai/ai2_arc", "ARC-Challenge", "train"),
    ("openai/gsm8k", "main", "train"),
    ("google-research-datasets/mbpp", "full", "train"),
]
LM_EVAL_TASKS = ("ceval-valid", "gsm8k", "humaneval_instruct", "humaneval_codellama")

failures: list[str] = []


def report(ok: bool, what: str, fix: str = "") -> None:
    print(f"  [{'ok' if ok else 'MISSING'}] {what}" + ("" if ok or not fix else f"\n            fix: {fix}"), flush=True)
    if not ok:
        failures.append(what)


def fetch(code: str) -> None:
    """Download in a fresh process: the hub and datasets read their offline switches at import."""
    import subprocess

    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE")}
    subprocess.run([sys.executable, "-c", code], check=False, env=env)


def offline(on: bool) -> None:
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1" if on else "0"


def ceval_subjects() -> list[str]:
    import importlib.util

    import yaml

    spec = importlib.util.find_spec("lm_eval")
    folder = Path(list(spec.submodule_search_locations)[0]) / "tasks" / "ceval"
    names = []
    for path in sorted(folder.glob("ceval-valid_*.yaml")):
        text = path.read_text()
        data = yaml.safe_load(text.replace("!function", ""))  # only the dataset name is needed
        if data and data.get("dataset_name"):
            names.append(data["dataset_name"])
    return names


MODEL_FILES = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.tiktoken", "*.txt", "*.jinja"]


def check_models(models, download: bool) -> None:
    """Only what transformers loads: config, tokenizer and the safetensors shards in the index."""
    from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
    from transformers import AutoConfig, AutoTokenizer

    print("models (Hugging Face cache " + os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub") + ")")
    for name, model in models.items():
        repo = model.pretrained
        if download:
            fetch(f"from huggingface_hub import snapshot_download; snapshot_download({repo!r}, allow_patterns={MODEL_FILES!r})")
        try:
            try:
                index = Path(hf_hub_download(repo, "model.safetensors.index.json", local_files_only=True))
                shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            except Exception:  # noqa: BLE001 - a single-file checkpoint
                shards = ["model.safetensors"]
            missing = [s for s in shards if not isinstance(try_to_load_from_cache(repo, s), str)]
            AutoConfig.from_pretrained(repo)
            AutoTokenizer.from_pretrained(repo)
            ok = bool(shards) and not missing
            report(ok, f"{name}: {repo} ({len(shards)} weight shards" +
                   (f", missing {missing}" if missing else "") + ", config, tokenizer)",
                   f"preflight.py --download (or huggingface-cli download {repo})")
        except Exception as error:  # noqa: BLE001
            report(False, f"{name}: {repo} ({type(error).__name__}: {str(error)[:120]})",
                   f"preflight.py --download (or huggingface-cli download {repo})")


def check_datasets(entries, label: str, download: bool, quiet: bool = False) -> None:
    """``quiet``: one summary line, plus a line per missing entry."""
    from datasets import load_dataset

    print(label)
    found = 0
    for path, config, split in entries:
        what = f"{path}" + (f" [{config}]" if config else "") + f" {split}"
        if download:
            fetch(f"from datasets import load_dataset; load_dataset({path!r}, {config!r}, split={split!r})")
        try:
            rows = len(load_dataset(path, config, split=split))
            found += rows > 0
            if not quiet or rows == 0:
                report(rows > 0, f"{what}: {rows} rows")
        except Exception as error:  # noqa: BLE001
            report(False, f"{what} ({type(error).__name__})",
                   f"python -c \"from datasets import load_dataset; load_dataset('{path}'"
                   + (f", '{config}'" if config else "") + f", split='{split}')\"")
    if quiet:
        print(f"  [{'ok' if found == len(entries) else 'MISSING'}] {found}/{len(entries)} present", flush=True)


def check_ceval(download: bool) -> None:
    subjects = ceval_subjects()
    check_datasets([("ceval/ceval-exam", s, split) for s in subjects for split in ("val", "dev")],
                   f"C-Eval ({len(subjects)} subjects, val and dev)", download, quiet=True)


def check_lm_eval() -> None:
    print("lm-eval tasks and metric")
    from lm_eval.tasks import TaskManager

    from engine.eval_runner.execute import LOCAL_TASKS

    known = set(TaskManager(include_path=str(LOCAL_TASKS)).all_tasks)
    for task in LM_EVAL_TASKS:
        report(task in known, f"task {task}")
    try:
        import evaluate

        evaluate.load("code_eval")
        report(True, "code_eval metric (HumanEval scoring)")
    except Exception as error:  # noqa: BLE001
        report(False, f"code_eval metric ({type(error).__name__})",
               "HF_EVALUATE_OFFLINE=0 python -c \"import evaluate; evaluate.load('code_eval')\"")


def check_query_weights(models) -> None:
    import torch
    from transformers import AutoConfig

    print("spatial query weights")
    for name, model in models.items():
        path = ROOT / model.query_weights
        if not path.exists():
            report(False, f"{name}: {model.query_weights}", "git checkout rerun-kl-greedy")
            continue
        config = AutoConfig.from_pretrained(model.pretrained)
        text = getattr(config, "text_config", config)
        expected = (text.num_hidden_layers, text.num_key_value_heads,
                    getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads)
        shape = tuple(torch.load(path, weights_only=True)["weights"].shape)
        report(shape == expected, f"{name}: {model.query_weights} {shape}, model {expected}")


def check_calibration(models) -> list[str]:
    """Models whose calibration set will be built on the first run."""
    print("KL calibration sets")
    to_build = []
    for name, model in models.items():
        folder = ROOT / "catalog" / "calibration" / model.calibration
        meta_path = folder / "manifest_metadata.json"
        if not meta_path.exists():
            print(f"  [build] {name}: {folder.relative_to(ROOT)} will be built on the first run (1 GPU)")
            to_build.append(name)
            continue
        meta = json.loads(meta_path.read_text())
        digest = hashlib.sha256((folder / "manifest.jsonl").read_bytes()).hexdigest()
        report(digest == meta["manifest_sha256"] and meta["model"] == model.pretrained,
               f"{name}: {folder.relative_to(ROOT)} ({meta['model']}, checksum)")
    return to_build


def check_environment() -> None:
    print("environment")
    import datasets
    import lm_eval
    import torch
    import transformers

    print(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
          f"lm_eval {getattr(lm_eval, '__version__', '?')}, datasets {datasets.__version__}")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401

        report(True, "transformers knows Qwen3.5")
    except ImportError:
        report(False, "transformers knows Qwen3.5", "upgrade transformers in llm_isa_dgx")
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  GPUs visible here: {gpus} (the login node may show none; the job books 4)")
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    if cache.exists():
        stat = os.statvfs(cache)
        print(f"  free space at {cache}: {stat.f_bavail * stat.f_frsize / 2**30:.0f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch missing models and datasets first")
    args = parser.parse_args()
    offline(True)
    os.chdir(ROOT)
    from compression_topics.kl_greedy.scripts.rerun import MODELS

    check_environment()
    check_models(MODELS, args.download)
    check_datasets(TASK_DATASETS, "task datasets", args.download)
    check_ceval(args.download)
    check_lm_eval()
    check_query_weights(MODELS)
    if check_calibration(MODELS):
        check_datasets(CALIBRATION_DATASETS, "calibration-set sources", args.download)
    print()
    if failures:
        print(f"{len(failures)} missing; fix them (or rerun with --download) before submitting the job.")
        sys.exit(1)
    print("All present. Submit: sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh")


if __name__ == "__main__":
    main()
