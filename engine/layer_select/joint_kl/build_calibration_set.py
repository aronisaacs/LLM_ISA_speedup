#!/usr/bin/env python3
"""Build a model's 4-domain KL calibration set (the manifest ``calibration_set.load`` reads).

Ported from the E18 bundle's ``e11_mixed4_calibration.py prepare`` (research partner's kv-sparsity
project), which made ``catalog/calibration/llama31_8b_mixed4``. Prompt building, example selection,
dense greedy answers and the manifest format are unchanged; only the CheckSparse bits are gone.

  domain     source (split)                          scored tokens
  wikitext   WikiText-2 raw (validation)             256 per window, up to 1536 tokens of context
  arc_5shot  ARC-Challenge (train), 5 shots          the last <=256 tokens after the first shot
  gsm8k      GSM8K main (train)                      the dense model's own greedy answer (<=256)
  mbpp       MBPP full (train)                       the dense model's own greedy answer (<=256)

None of these splits is used by the task evaluations (C-Eval valid, GSM8K test, HumanEval).
Instruct models use the chat template (``--prompt-format chat``; Qwen3.5 with thinking off);
base models (CodeLlama) use lm-eval's plain-text prompts and stop strings, 0-shot GSM8K and MBPP.

  python engine/layer_select/joint_kl/build_calibration_set.py --model Qwen/Qwen3.5-9B \
      --output-dir catalog/calibration/qwen35_9b_mixed4 --gsm8k-shots 5 --mbpp-shots 3 \
      --chat-template-kwargs '{"enable_thinking": false}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "E11"
FORMAT_VERSION = 1
DOMAINS = ("wikitext", "arc_5shot", "gsm8k", "mbpp")

SOURCES = {
    "wikitext": {"path": "EleutherAI/wikitext_document_level", "name": "wikitext-2-raw-v1", "split": "validation"},
    "arc_5shot": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "train"},
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "train"},
    "mbpp": {"path": "google-research-datasets/mbpp", "name": "full", "split": "train"},
}
# Splits that the downstream suite (or likely future evals) use. prepare refuses them.
FORBIDDEN_SPLITS = {
    "wikitext": {"test"},
    "arc_5shot": {"test", "validation"},
    "gsm8k": {"test"},
    "mbpp": {"test", "validation", "prompt"},
}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")
    tmp.replace(path)


def iter_jsonl(path: Path, tolerate_partial_tail: bool = False) -> Iterator[dict[str, Any]]:
    lines = Path(path).read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial_tail and number == len(lines):
                return  # interrupted write; the example is recomputed on resume
            raise ValueError(f"Invalid JSON at {path}:{number}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require_new(path: Path, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Data preparation (pure Python; tokenizer is duck-typed so tests can fake it)
# --------------------------------------------------------------------------- #
def wikitext_detokenizer(string: str) -> str:
    """lm-eval's wikitext detokenizer (lm_eval/tasks/wikitext/preprocess_wikitext.py)."""
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def encode(tokenizer: Any, text: str) -> list[int]:
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def chat_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool,
             chat_kwargs: dict[str, Any] | None = None) -> list[int]:
    """Render with the tokenizer's chat template exactly as lm-eval does (text, then encode
    without extra special tokens; the Llama-3.1 template already emits <|begin_of_text|>).
    chat_kwargs go to the template, e.g. {"enable_thinking": False} for Qwen3.5."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         **(chat_kwargs or {}))
    return encode(tokenizer, text)


def plain_ids(tokenizer: Any, text: str) -> list[int]:
    """Plain-text prompt for base models (no chat template): BOS (if any) + text."""
    bos = tokenizer.bos_token_id
    return ([int(bos)] if bos is not None else []) + encode(tokenizer, text)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def wikitext_items(tokenizer: Any, rows: list[dict[str, Any]], n: int, rng: random.Random,
                   score_tokens: int, context_tokens: int, min_context: int) -> list[dict[str, Any]]:
    bos = tokenizer.bos_token_id  # None for Qwen3.5: windows then start without BOS
    per_doc: list[list[dict[str, Any]]] = []
    for doc_index, row in enumerate(rows):
        ids = encode(tokenizer, wikitext_detokenizer(str(row["page"])))
        windows = []
        for start in range(min_context, len(ids) - score_tokens + 1, score_tokens):
            context = ids[max(0, start - context_tokens):start]
            windows.append({
                "sample_id": f"wikitext:validation:{doc_index}:{start}",
                "source_index": doc_index,
                "prompt_token_ids": ([int(bos)] if bos is not None else []) + context,
                "reference_token_ids": ids[start:start + score_tokens],
                "kind": "text",
            })
        if windows:
            rng.shuffle(windows)
            per_doc.append(windows)
    rng.shuffle(per_doc)
    # Round-robin over documents so the 128 windows come from as many documents as possible.
    chosen: list[dict[str, Any]] = []
    depth = 0
    while len(chosen) < n:
        added = False
        for windows in per_doc:
            if depth < len(windows) and len(chosen) < n:
                chosen.append(windows[depth])
                added = True
        if not added:
            raise ValueError(f"WikiText provides only {len(chosen)} windows, need {n}")
        depth += 1
    return chosen


def arc_letters(row: dict[str, Any]) -> tuple[list[str], list[str], str]:
    texts = list(row["choices"]["text"])
    labels = [str(x) for x in row["choices"]["label"]]
    letters = [chr(ord("A") + i) for i in range(len(texts))]
    gold = letters[labels.index(str(row["answerKey"]))]
    return letters, texts, gold


def arc_question(row: dict[str, Any]) -> str:
    letters, texts, _ = arc_letters(row)
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, texts))
    return f"Question: {row['question']}\n{options}\nAnswer:"


def arc_messages(shots: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for shot in shots:  # few-shot as multiturn, like C-Eval's fewshot_as_multiturn
        messages.append({"role": "user", "content": arc_question(shot)})
        messages.append({"role": "assistant", "content": arc_letters(shot)[2]})
    messages.append({"role": "user", "content": arc_question(target)})
    return messages


def gsm8k_messages(row: dict[str, Any], shots: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    # lm-eval gsm8k doc_to_text, used 0-shot with the chat template downstream. With shots (E15): lm-eval's
    # few-shot format as multiturn, each shot answered with its full train-split solution (doc_to_target).
    messages: list[dict[str, str]] = []
    for shot in shots or []:
        messages.append({"role": "user", "content": f"Question: {shot['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": f"Question: {row['question']}\nAnswer:"})
    return messages


MBPP_GEN_PREFIX = "\n```python\n"


def _lm_eval_mbpp_utils() -> Path:
    """lm-eval's mbpp/utils.py from the installed package."""
    import importlib.util

    spec = importlib.util.find_spec("lm_eval")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("lm_eval is not installed")
    return Path(list(spec.submodule_search_locations)[0]) / "tasks" / "mbpp" / "utils.py"


def mbpp_fewshot_examples() -> list[dict[str, Any]]:
    """lm-eval's fixed MBPP few-shot examples (utils.list_fewshot_samples, sampler first_n).

    Read with ast instead of importing: importing utils.py loads the code_eval metric at module level.
    The examples are MBPP task ids 2-4 (prompt split); they are context only, never scored."""
    import ast

    path = _lm_eval_mbpp_utils()
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "list_fewshot_samples":
            ret = next(n for n in node.body if isinstance(n, ast.Return))
            return ast.literal_eval(ret.value)
    raise RuntimeError(f"list_fewshot_samples not found in {path}")


# Plain-text formats for base models (prompt_format="plain"), following lm-eval's non-chat tasks.
ARC_SHOT_SEPARATOR = "\n\n"
PLAIN_STOPS = {
    "arc_5shot": ["\n\n", "Question:"],
    "gsm8k": ["Question:", "</s>", "<|im_end|>"],  # lm-eval gsm8k `until`
    "mbpp": ["[DONE]"],                             # lm-eval mbpp `until`
}


def arc_plain(shots: list[dict[str, Any]], target: dict[str, Any]) -> tuple[str, str]:
    """(full prompt, first shot block) as plain text: shots answered inline, target left open."""
    blocks = [f"{arc_question(shot)} {arc_letters(shot)[2]}" for shot in shots]
    return ARC_SHOT_SEPARATOR.join(blocks + [arc_question(target)]), blocks[0]


def gsm8k_plain(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"  # lm-eval gsm8k doc_to_text


def mbpp_plain(row: dict[str, Any]) -> str:
    tests = list(row["test_list"])[:3]  # lm-eval mbpp doc_to_text (base-model variant)
    return (f"You are an expert Python programmer, and here is your task: {row['text']} "
            "Your code should pass these tests:\n\n" + "\n".join(tests) + "\n[BEGIN]\n")


def mbpp_instruct_text(row: dict[str, Any]) -> str:
    tests = list(row["test_list"])[:3]  # lm-eval mbpp_instruct doc_to_text
    return ("You are an expert Python programmer, and here is your task:\n"
            f"{row['text']}\nYour code should pass these tests:\n" + "\n".join(tests))


def mbpp_messages(row: dict[str, Any], shots: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    # lm-eval mbpp_instruct doc_to_text; gen_prefix is appended after the assistant header. With shots (E15):
    # lm-eval's multiturn few-shot turns, assistant = gen_prefix + code + "\n```" (fewshot doc_to_target).
    messages: list[dict[str, str]] = []
    for shot in shots or []:
        messages.append({"role": "user", "content": mbpp_instruct_text(shot)})
        messages.append({"role": "assistant", "content": f"{MBPP_GEN_PREFIX}{shot['code']}\n```"})
    messages.append({"role": "user", "content": mbpp_instruct_text(row)})
    return messages


def build_generation_requests(tokenizer: Any, data: dict[str, list[dict[str, Any]]], n: int,
                              rng_seed: int, shots: int, prompt_format: str = "chat",
                              chat_kwargs: dict[str, Any] | None = None, gsm8k_shots: int = 0,
                              mbpp_shots: int = 0) -> list[dict[str, Any]]:
    """Prompts that need a dense answer (arc target, gsm8k, mbpp).

    prompt_format "chat" (E11, instruct models) renders with the chat template; "plain" (base models,
    e.g. CodeLlama) uses lm-eval's plain-text prompts and stop strings. Example selection is identical.
    gsm8k_shots / mbpp_shots (E15, chat only) add few-shot turns before the same target questions: GSM8K shots
    are sampled from the train split (never the target), MBPP shots are lm-eval's fixed examples. 0 = E11 prompts."""
    if prompt_format not in ("chat", "plain"):
        raise ValueError(prompt_format)
    chat = prompt_format == "chat"
    if (gsm8k_shots or mbpp_shots) and not chat:
        raise NotImplementedError("Few-shot GSM8K/MBPP is implemented for the chat format only")
    mbpp_pool = mbpp_fewshot_examples() if mbpp_shots else []
    if mbpp_shots > len(mbpp_pool):
        raise ValueError(f"mbpp_shots={mbpp_shots}, but lm-eval has {len(mbpp_pool)} fixed examples")
    render = lambda msgs, gen: chat_ids(tokenizer, msgs, gen, chat_kwargs)  # noqa: E731
    requests: list[dict[str, Any]] = []

    arc = data["arc_5shot"]
    rng = random.Random(f"{rng_seed}:arc_5shot")
    targets = rng.sample(range(len(arc)), n)
    for index in targets:
        pool = [j for j in range(len(arc)) if j != index]
        shot_idx = random.Random(f"{rng_seed}:arc_5shot:shots:{index}").sample(pool, shots)
        shot_rows = [arc[j] for j in shot_idx]
        if chat:
            messages = arc_messages(shot_rows, arc[index])
            prompt = render(messages, True)
            first_shot = render(messages[:2], False)
            if prompt[:len(first_shot)] != first_shot:
                # Some templates (Qwen3.5) render a final assistant turn differently from a past one;
                # fall back to the first user turn, which is always a prefix.
                first_shot = render(messages[:1], False)
        else:
            text, first_text = arc_plain(shot_rows, arc[index])
            prompt = plain_ids(tokenizer, text)
            first_shot = plain_ids(tokenizer, first_text)
            first_shot = prompt[:common_prefix_len(prompt, first_shot)]
        if prompt[:len(first_shot)] != first_shot:
            raise RuntimeError(f"First-shot rendering is not a token prefix for arc {index}")
        requests.append({
            "domain": "arc_5shot", "sample_id": f"arc_5shot:train:{arc[index]['id']}",
            "source_index": index, "shot_indices": shot_idx,
            "prompt_ids": prompt,
            "first_shot_len": len(first_shot), "first_shot_ids": first_shot,
            "gold": arc_letters(arc[index])[2], "max_new_tokens": 16,
        })

    gsm = data["gsm8k"]
    for index in random.Random(f"{rng_seed}:gsm8k").sample(range(len(gsm)), n):
        extra: dict[str, Any] = {}
        if gsm8k_shots:
            # Own RNG stream per target, so the target questions stay exactly E11's.
            pool = [j for j in range(len(gsm)) if j != index]
            shot_idx = random.Random(f"{rng_seed}:gsm8k:shots:{index}").sample(pool, gsm8k_shots)
            prompt = render(gsm8k_messages(gsm[index], [gsm[j] for j in shot_idx]), True)
            extra["shot_indices"] = shot_idx
        else:
            prompt = (render(gsm8k_messages(gsm[index]), True) if chat
                      else plain_ids(tokenizer, gsm8k_plain(gsm[index])))
        requests.append({
            "domain": "gsm8k", "sample_id": f"gsm8k:train:{index}", "source_index": index,
            "prompt_ids": prompt,
            "gold": gsm[index]["answer"].split("####")[-1].strip().replace(",", ""),
            "max_new_tokens": 512, **extra,
        })

    mbpp = data["mbpp"]
    for index in random.Random(f"{rng_seed}:mbpp").sample(range(len(mbpp)), n):
        extra = {}
        if chat:
            shot_rows = mbpp_pool[:mbpp_shots]  # sampler first_n, as lm-eval
            prompt = render(mbpp_messages(mbpp[index], shot_rows), True) + encode(tokenizer, MBPP_GEN_PREFIX)
            if mbpp_shots:
                extra["shot_task_ids"] = [int(s["task_id"]) for s in shot_rows]
        else:
            prompt = plain_ids(tokenizer, mbpp_plain(mbpp[index]))
        requests.append({
            "domain": "mbpp", "sample_id": f"mbpp:train:{mbpp[index]['task_id']}", "source_index": index,
            "prompt_ids": prompt,
            "gold": None, "max_new_tokens": 512, **extra,
        })
    return requests


def trim_generation(ids: list[int], eos_ids: set[int], pad_id: int | None) -> tuple[list[int], bool]:
    """Keep tokens up to and including the first EOS. Returns (ids, stopped_on_eos)."""
    out = []
    for token in ids:
        token = int(token)
        if token in eos_ids:
            out.append(token)
            return out, True
        if pad_id is not None and token == pad_id and token not in eos_ids:
            break
        out.append(token)
    return out, False


def last_number(text: str) -> str | None:
    found = re.findall(r"-?[\d,]*\.?\d+", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def finalize_row(request: dict[str, Any], answer_ids: list[int], answer_text: str, stopped: bool,
                 score_tokens: int) -> dict[str, Any]:
    domain = request["domain"]
    row = {"domain": domain, "sample_id": request["sample_id"], "source_index": request["source_index"],
           "kind": request.get("kind", "chat"), "dense_answer_text": answer_text, "dense_answer_tokens": len(answer_ids),
           "dense_stopped_on_eos": stopped}
    if domain == "arc_5shot":
        full = list(request["prompt_ids"]) + list(answer_ids)
        first = int(request["first_shot_len"])
        if full[:first] != list(request["first_shot_ids"]):
            raise RuntimeError(f"First-shot rendering is not a token prefix for {request['sample_id']}")
        region = full[first:]
        take = min(score_tokens, len(region))
        row["prompt_token_ids"] = full[:len(full) - take]
        row["reference_token_ids"] = full[len(full) - take:]
        row["shot_indices"] = request["shot_indices"]
        letter = re.search(r"\b([A-E])\b", answer_text)
        row["dense_correct"] = bool(letter and letter.group(1) == request["gold"])
    else:
        row["prompt_token_ids"] = list(request["prompt_ids"])
        row["reference_token_ids"] = list(answer_ids[:score_tokens])
        for key in ("shot_indices", "shot_task_ids"):  # few-shot GSM8K / MBPP (E15); absent in E11 rows
            if key in request:
                row[key] = request[key]
        if domain == "gsm8k":
            row["dense_correct"] = last_number(answer_text) == request["gold"]
    if not row["reference_token_ids"]:
        raise RuntimeError(f"Empty scored region for {request['sample_id']}")
    row["scored_tokens"] = len(row["reference_token_ids"])
    return row


def load_sources(load_dataset: Any) -> dict[str, list[dict[str, Any]]]:
    data = {}
    for domain, src in SOURCES.items():
        if src["split"] in FORBIDDEN_SPLITS[domain]:
            raise ValueError(f"{domain}: split {src['split']} is reserved for evaluation")
        dataset = load_dataset(src["path"], src["name"], split=src["split"])
        data[domain] = [dict(row) for row in dataset]
    return data


def command_prepare(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir)
    manifest_path, meta_path = out / "manifest.jsonl", out / "manifest_metadata.json"
    require_new(manifest_path, args.overwrite)
    require_new(meta_path, args.overwrite)
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    data = load_sources(load_dataset)

    chat_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    wiki = wikitext_items(tokenizer, data["wikitext"], args.examples_per_domain,
                          random.Random(f"{args.seed}:wikitext"), args.score_tokens,
                          args.wikitext_context_tokens, args.wikitext_min_context)
    requests = build_generation_requests(tokenizer, data, args.examples_per_domain, args.seed, args.arc_shots,
                                         args.prompt_format, chat_kwargs, args.gsm8k_shots, args.mbpp_shots)
    for request in requests:
        request["kind"] = args.prompt_format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), **kw).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None}
    vocab = tokenizer.get_vocab()
    for token in ("<|eot_id|>", "<|im_end|>"):  # Llama-3.1 / Qwen end-of-turn
        if token in vocab:
            eos_ids.add(int(vocab[token]))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = int(vocab["<|finetune_right_pad_id|>"]) if "<|finetune_right_pad_id|>" in vocab \
            else int(tokenizer.eos_token_id)
    stop_kw = {}

    rows: list[dict[str, Any]] = []
    for item in wiki:
        item = dict(item, domain="wikitext", scored_tokens=len(item["reference_token_ids"]))
        rows.append(item)

    started = time.time()
    by_budget = defaultdict(list)
    for request in requests:
        # chat (E11): batches mix gsm8k and mbpp as before; plain: per domain (each has its own stop strings)
        by_budget[(request["max_new_tokens"], request["domain"] if args.prompt_format == "plain" else None)
                  ].append(request)
    for (max_new, domain), group in by_budget.items():
        if args.prompt_format == "plain":
            stop_kw = {"stop_strings": PLAIN_STOPS[domain], "tokenizer": tokenizer}
        for b in range(0, len(group), args.gen_batch_size):
            batch = group[b:b + args.gen_batch_size]
            width = max(len(r["prompt_ids"]) for r in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, r in enumerate(batch):  # left padding
                ids[i, width - len(r["prompt_ids"]):] = torch.tensor(r["prompt_ids"])
                mask[i, width - len(r["prompt_ids"]):] = 1
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                     max_new_tokens=max_new, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=pad_id,
                                     eos_token_id=sorted(eos_ids), **stop_kw)
            for i, r in enumerate(batch):
                answer, stopped = trim_generation(gen[i, width:].tolist(), eos_ids, pad_id)
                if stop_kw and answer and answer[-1] == pad_id and \
                        any(st in tokenizer.decode(answer[:-1]) for st in stop_kw["stop_strings"]):
                    answer = answer[:-1]  # padding after a stop string (pad = EOS for CodeLlama), not generated
                text = tokenizer.decode(answer, skip_special_tokens=True)
                rows.append(finalize_row(r, answer, text, stopped, args.score_tokens))
            print(f"generated {min(b + args.gen_batch_size, len(group))}/{len(group)} {domain or ''} "
                  f"(max_new_tokens={max_new}, {time.time() - started:.0f}s)", flush=True)

    order = {d: i for i, d in enumerate(DOMAINS)}
    rows.sort(key=lambda r: (order[r["domain"]], r["sample_id"]))
    for position, row in enumerate(rows):
        row["position"] = position
    write_jsonl_atomic(manifest_path, rows)

    stats = {}
    for domain in DOMAINS:
        part = [r for r in rows if r["domain"] == domain]
        scored = [r["scored_tokens"] for r in part]
        prompt = [len(r["prompt_token_ids"]) for r in part]
        entry = {"examples": len(part), "scored_tokens": sum(scored),
                 "scored_tokens_min_mean_max": [min(scored), sum(scored) / len(scored), max(scored)],
                 "prompt_tokens_min_mean_max": [min(prompt), sum(prompt) / len(prompt), max(prompt)]}
        if any("dense_correct" in r for r in part):
            entry["dense_accuracy"] = sum(bool(r.get("dense_correct")) for r in part) / len(part)
        if domain != "wikitext":
            entry["dense_stopped_on_eos"] = sum(bool(r["dense_stopped_on_eos"]) for r in part) / len(part)
        if domain == "wikitext":
            entry["documents"] = len({r["source_index"] for r in part})
        if domain == "arc_5shot":
            entry["distinct_questions"] = len({r["source_index"] for r in part}
                                              | {j for r in part for j in r["shot_indices"]})
        if domain in ("gsm8k", "mbpp") and any(k in r for r in part for k in ("shot_indices", "shot_task_ids")):
            entry["shots"] = args.gsm8k_shots if domain == "gsm8k" else args.mbpp_shots
        stats[domain] = entry
        print(f"{domain}: {json.dumps(entry)}")

    write_json_atomic(meta_path, {
        "experiment": args.experiment_id, "format_version": FORMAT_VERSION, "created_unix": time.time(),
        "prompt_format": args.prompt_format, "chat_template_kwargs": chat_kwargs,
        "plain_stops": PLAIN_STOPS if args.prompt_format == "plain" else None,
        "git_commit": git_commit(),
        "model": args.model, "revision": args.revision, "dtype": args.dtype,
        "seed": args.seed, "examples_per_domain": args.examples_per_domain,
        "score_tokens": args.score_tokens, "arc_shots": args.arc_shots,
        "gsm8k_shots": args.gsm8k_shots, "mbpp_shots": args.mbpp_shots,
        "mbpp_fewshot_source": "lm_eval/tasks/mbpp/utils.py:list_fewshot_samples" if args.mbpp_shots else None,
        "wikitext_context_tokens": args.wikitext_context_tokens,
        "wikitext_min_context": args.wikitext_min_context,
        "domains": list(DOMAINS), "sources": SOURCES, "stats": stats,
        "generation": {"do_sample": False, "batch_size": args.gen_batch_size, "padding": "left",
                       "eos_ids": sorted(eos_ids)},
        "manifest_sha256": file_sha256(manifest_path),
    })
    print(f"Wrote {manifest_path} and {meta_path} in {time.time() - started:.0f}s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--examples-per-domain", type=int, default=128)
    p.add_argument("--score-tokens", type=int, default=256, help="Max scored tokens per example")
    p.add_argument("--arc-shots", type=int, default=5)
    p.add_argument("--gsm8k-shots", type=int, default=0,
                   help="Few-shot turns before each GSM8K target (chat only; Llama's set: 5)")
    p.add_argument("--mbpp-shots", type=int, default=0,
                   help="lm-eval's fixed MBPP few-shot turns before each target (chat only; Llama's set: 3)")
    p.add_argument("--wikitext-context-tokens", type=int, default=1536)
    p.add_argument("--wikitext-min-context", type=int, default=256,
                   help="Scored WikiText windows start at least this many tokens into the document")
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt-format", choices=("chat", "plain"), default="chat",
                   help="chat: instruct models; plain: base models without a chat template (CodeLlama)")
    p.add_argument("--chat-template-kwargs", default="",
                   help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": false}\' for Qwen3.5')
    p.add_argument("--experiment-id", default=EXPERIMENT_ID, help="Recorded in manifest_metadata.json")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    command_prepare(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
