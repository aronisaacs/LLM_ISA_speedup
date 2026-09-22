"""lm-eval task defaults used by run files.

Keys such as name_task / file are for run output names, not passed to lm-eval
unless the run copies them into a configuration (make_configuration strips them).
"""

ARC_EASY_256 = {
    "tasks": ["arc_easy"],
    "limit": 256,
}

GSM8K_32 = {
    "tasks": ["gsm8k"],
    "limit": 32,
    "gen_kwargs": {"max_gen_toks": 128},
}

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
    "samples": "@gsm8k_samples_profile20pct.json",
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
