# Qwen/Qwen3.5-9B, 4-domain KL calibration set

Built by `engine/layer_select/joint_kl/build_calibration_set.py` (from `compression_topics/kl_greedy/scripts/rerun.py`) with `--prompt-format chat --gsm8k-shots 5 --mbpp-shots 3 --chat-template-kwargs {"enable_thinking": false}`. Same domains, sources, selection and format as `llama31_8b_mixed4`; the scored answers are this model's own dense greedy answers. `manifest_metadata.json` holds its SHA-256, checked on load.
