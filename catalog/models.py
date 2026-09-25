"""Hugging Face model_args strings used by run files."""

# Fixed examples per GPU. An integer never triggers lm-eval's max-batch search.
BATCH_SIZE = 8

LLAMA32_1B = "pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype=float16"
LLAMA31_8B = "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16"
QWEN35_9B = "pretrained=Qwen/Qwen3.5-9B,dtype=bfloat16,enable_thinking=False"
CODELLAMA_7B = "pretrained=meta-llama/CodeLlama-7b-hf,dtype=bfloat16"
