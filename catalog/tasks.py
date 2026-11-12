"""lm-eval task defaults used by run files.

Keys such as name_task / file are for run output names, not passed to lm-eval
unless the run copies them into a configuration (make_configuration strips them).
"""

CEVAL_VALID_5SHOT = {
    "name_task": "ceval",
    "file": "ceval",
    "tasks": ["ceval-valid"],
    "num_fewshot": 5,
    "apply_chat_template": True,
}

GSM8K_20PCT = {
    "name_task": "gsm8k",
    "file": "gsm8k_20pct",
    "tasks": ["gsm8k"],
    "num_fewshot": 0,
    "gen_kwargs": {"max_gen_toks": 512},
    "apply_chat_template": True,
}

HUMANEVAL_INSTRUCT = {
    "name_task": "humaneval_instruct",
    "file": "humaneval_instruct",
    "tasks": ["humaneval_instruct"],
    "num_fewshot": 0,
    "env": {"HF_ALLOW_CODE_EVAL": "1"},
    "confirm_run_unsafe_code": True,
    "apply_chat_template": True,
}

# lm-eval's plain completion HumanEval, for base (non-instruct) code models.
HUMANEVAL_BASE = {
    "name_task": "humaneval_base",
    "file": "humaneval_base",
    "tasks": ["humaneval"],
    "num_fewshot": 0,
    "env": {"HF_ALLOW_CODE_EVAL": "1"},
    "confirm_run_unsafe_code": True,
}

HUMANEVAL_CODELLAMA = {
    "name_task": "humaneval",
    "file": "humaneval",
    "tasks": ["humaneval_codellama"],
    "num_fewshot": 0,
    "env": {"HF_ALLOW_CODE_EVAL": "1"},
    "confirm_run_unsafe_code": True,
}

WIKITEXT = {
    "tasks": ["wikitext"],
    "limit": 32,
    "num_fewshot": 0,
    "apply_chat_template": False,
}

WIKITEXT_FULL = {
    "tasks": ["wikitext"],
    "num_fewshot": 0,
    "apply_chat_template": False,
}

# LongBench-E, the 13 English tasks the TurboQuant paper reports (Table 1). As in LongBench's
# own pred.py, the few-shot and code tasks get no chat template and keep the BOS token; the
# other eight use the chat template. One example at a time (no padding), as LongBench does.
# No prompt is cut: lm-eval takes Llama-3.1's 131,072-token limit from the model config.
LONGBENCH_E_CATEGORIES = {
    "single_doc_qa": ("longbench_qasper_e", "longbench_multifieldqa_en_e"),
    "multi_doc_qa": ("longbench_hotpotqa_e", "longbench_2wikimqa_e"),
    "summarization": ("longbench_gov_report_e", "longbench_multi_news_e"),
    "few_shot": ("longbench_trec_e", "longbench_triviaqa_e", "longbench_samsum_e"),
    "synthetic": ("longbench_passage_count_e", "longbench_passage_retrieval_en_e"),
    "code": ("longbench_lcc_e", "longbench_repobench-p_e"),
}
_LONGBENCH_PLAIN = ("few_shot", "code")
_LLAMA31_8B_BOS = "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,add_bos_token=True"

LONGBENCH_E_CHAT = {
    "name_task": "longbench_e_chat",
    "file": "longbench_e_chat",
    "tasks": [t for c, ts in LONGBENCH_E_CATEGORIES.items() if c not in _LONGBENCH_PLAIN for t in ts],
    "num_fewshot": 0,
    "batch_size": 1,
    "apply_chat_template": True,
}

LONGBENCH_E_PLAIN = {
    "name_task": "longbench_e_plain",
    "file": "longbench_e_plain",
    "tasks": [t for c in _LONGBENCH_PLAIN for t in LONGBENCH_E_CATEGORIES[c]],
    "num_fewshot": 0,
    "batch_size": 1,
    "apply_chat_template": False,
    "model_args": _LLAMA31_8B_BOS,
}
