# Llama-3.1-8B-Instruct, 4-domain KL calibration set

512 examples, 128 per domain, used by `engine/layer_select/joint_kl` to score a compression map
by its KL divergence from the dense model. Copied unchanged from the E18 bundle (research partner's
kv-sparsity project, commit `2afc0e35`; built by `e11_mixed4_calibration.py prepare` with
`--gsm8k-shots 5 --mbpp-shots 3`). `manifest_metadata.json` holds its SHA-256, checked on load.

| Domain | Source (split) | Format | Scored tokens |
|---|---|---|---|
| `wikitext` | WikiText-2 raw (validation) | BOS + up to 1536 tokens of context | 256 per window |
| `arc_5shot` | ARC-Challenge (train) | chat template, 5 shots + target question | last 256 tokens after the first shot |
| `gsm8k` | GSM8K main (train) | chat template, 5 shots, target question | dense model's own greedy answer (up to 256) |
| `mbpp` | MBPP full (train) | chat template, lm-eval's 3 fixed shots | dense model's own greedy answer (up to 256) |

Each row stores `prompt_token_ids` and `reference_token_ids` (the scored tokens). No split used by
the task evaluations (C-Eval valid, GSM8K test, HumanEval) is touched.
