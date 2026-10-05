#!/usr/bin/env python3
"""Zero-groups in llama.cpp's key layout, computed on a laptop.

Group counts depend only on the K and V vectors, so no benchmark has to run.
One dense forward pass of Llama 3.1 8B over a few thousand tokens captures K
(after RoPE) and V for every layer. The model is streamed one decoder layer at
a time from the cached safetensors, so it fits in a few GB of memory. Each
sparsity level and method is then applied to the captured vectors offline.

K is reordered into llama.cpp's layout before counting (position 2j <- j,
2j + 1 <- j + d/2, the GGUF q/k permutation); V is counted as stored.
transformers-order counts are reported next to them for comparison with
vector_zero_runs.py / vector_pair_zero_runs.py.

Unlike those scripts, compression is not applied while the model runs, so
later layers see dense inputs. Each layer's K and V are sparsified on their own.

  python compression_topics/vector/scripts/llamacpp_zero_groups_local.py
  python compression_topics/vector/scripts/llamacpp_zero_groups_local.py --text ceval --tokens 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.vector.algorithms.vector_compress import _apply as scalar_sparsify  # noqa: E402
from compression_topics.vector.algorithms.vector_compress_pair import _prune_rope_pairs as pair_sparsify  # noqa: E402
from compression_topics.vector.scripts.llamacpp_zero_groups import (  # noqa: E402
    PERCENTAGES,
    random_groups,
    rle_net_saving,
    to_llamacpp_order,
    zero_group_starts,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).resolve().parents[1] / "figures" / "llamacpp_zero_groups_local.json"


class Recorder:
    """Stands in for the KV cache: keeps each layer's post-RoPE K and V."""

    def __init__(self) -> None:
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self.keys[layer_idx] = key_states.detach().to("cpu", torch.bfloat16)
        self.values[layer_idx] = value_states.detach().to("cpu", torch.bfloat16)
        return key_states, value_states


def load_text(kind: str, tokenizer, n_tokens: int, seq_len: int) -> torch.Tensor:
    from datasets import load_dataset

    if kind == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        docs = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows]
    else:
        from datasets import get_dataset_config_names

        docs = []
        for subject in get_dataset_config_names("ceval/ceval-exam"):
            for r in load_dataset("ceval/ceval-exam", subject, split="val").select(range(2)):
                docs.append(
                    f"{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n答案：{r['answer']}"
                )
    text, used = "", 0
    for doc in docs:  # tokenize only as much text as needed
        text += doc + "\n\n"
        used += 1
        if len(text) > 8 * n_tokens:
            break
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = min(n_tokens, ids.numel()) // seq_len * seq_len
    if n == 0:
        raise SystemExit("not enough text for one sequence")
    return ids[:n].reshape(-1, seq_len)


def capture(model_dir: Path, input_ids: torch.Tensor, device: str) -> Recorder:
    """Run the decoder one layer at a time, loading each layer's weights from disk."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    config = AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = "sdpa"
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            index = {name: "model.safetensors" for name in f.keys()}

    def tensor(name: str) -> torch.Tensor:
        with safe_open(model_dir / index[name], framework="pt") as f:
            return f.get_tensor(name)

    dtype = torch.float32 if device == "cpu" else torch.float16
    embed = tensor("model.embed_tokens.weight")
    hidden = embed[input_ids].to(device, dtype)
    del embed
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
    rotary = LlamaRotaryEmbedding(config).to(device)
    cos_sin = rotary(hidden, positions)
    recorder = Recorder()
    for i in range(config.num_hidden_layers):
        start = time.monotonic()
        layer = LlamaDecoderLayer(config, i)
        prefix = f"model.layers.{i}."
        state = {k[len(prefix):]: tensor(k) for k in index if k.startswith(prefix)}
        layer.load_state_dict(state)
        layer = layer.to(device, dtype).eval()
        with torch.no_grad():
            hidden = layer(hidden, position_ids=positions, past_key_values=recorder, position_embeddings=cos_sin)
        del layer, state
        if device == "mps":
            torch.mps.empty_cache()
        print(f"layer {i:2d}  {time.monotonic() - start:.1f}s", flush=True)
    return recorder


def count(tensor: torch.Tensor) -> tuple[float, float]:
    """(groups per vector, zeros per vector) along the last dimension."""
    vectors = tensor.numel() // tensor.shape[-1]
    return float(zero_group_starts(tensor).sum()) / vectors, float((tensor == 0).sum()) / vectors


def analyse(recorder: Recorder) -> dict:
    keys = list(recorder.keys.values())
    values = list(recorder.values.values())
    head_dim = keys[0].shape[-1]
    scalar_rows, pair_rows = [], []
    for pct in PERCENTAGES:
        zeroed = head_dim * pct // 100
        k_cpp = k_hf = v = 0.0
        pk_cpp = 0.0
        for k_layer, v_layer in zip(keys, values):
            k32, v32 = k_layer.float(), v_layer.float()
            ks = scalar_sparsify(k32, threshold=0.0, prune_pct=pct)
            k_cpp += count(to_llamacpp_order(ks))[0]
            k_hf += count(ks)[0]
            v += count(scalar_sparsify(v32, threshold=0.0, prune_pct=pct))[0]
            kp = to_llamacpp_order(pair_sparsify(k32, pct))
            pair_mask = (kp.reshape(*kp.shape[:-1], -1, 2) != 0).any(-1).float()
            pk_cpp += count(pair_mask)[0]
        n = len(keys)
        k_cpp, k_hf, v, pk_cpp = k_cpp / n, k_hf / n, v / n, pk_cpp / n
        scalar_rows.append(
            {
                "sparsity_pct": pct,
                "zeroed_per_vector": zeroed,
                "random_groups": random_groups(head_dim, zeroed),
                "k_llamacpp": _row(zeroed, k_cpp, head_dim),
                "v": _row(zeroed, v, head_dim),
                "k_and_v_llamacpp": _row(zeroed, (k_cpp + v) / 2, head_dim),
                "k_transformers_order": _row(zeroed, k_hf, head_dim),
                "k_and_v_transformers_order": _row(zeroed, (k_hf + v) / 2, head_dim),
            }
        )
        pairs = head_dim // 2
        zeroed_pairs = pairs * pct // 100
        pair_rows.append(
            {
                "sparsity_pct": pct,
                "zeroed_pairs_per_vector": zeroed_pairs,
                "random_groups": random_groups(pairs, zeroed_pairs),
                "groups_per_vector": pk_cpp,
                "avg_group_length_pairs": zeroed_pairs / pk_cpp if pk_cpp else None,
                "bits_freed_per_group": 32 * zeroed_pairs / pk_cpp if pk_cpp else None,
                "rle_net_saving": rle_net_saving(2 * zeroed_pairs, pk_cpp, head_dim),
            }
        )
    return {"scalar": scalar_rows, "pair": pair_rows}


def _row(zeroed: int, groups: float, head_dim: int) -> dict:
    return {
        "groups_per_vector": groups,
        "avg_group_length": zeroed / groups if groups else None,
        "bits_freed_per_group": 16 * zeroed / groups if groups else None,
        "rle_net_saving": rle_net_saving(zeroed, groups, head_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=("gsm8k", "ceval"), default="gsm8k")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from transformers import AutoTokenizer

    cached = try_to_load_from_cache(MODEL, "model.safetensors.index.json")
    if isinstance(cached, str):
        model_dir = Path(cached).parent  # use the local cache; no Hub call needed
    else:
        model_dir = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = load_text(args.text, tokenizer, args.tokens, args.seq_len)
    print(f"{input_ids.numel()} tokens as {input_ids.shape[0]} x {input_ids.shape[1]}  device {args.device}", flush=True)
    recorder = capture(model_dir, input_ids, args.device)
    result = {
        "model": MODEL,
        "text": args.text,
        "tokens": int(input_ids.numel()),
        "seq_len": args.seq_len,
        "layout": "K in llama.cpp order (adjacent RoPE pairs); V as stored",
        **analyse(recorder),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["scalar"]:
        print(
            f"scalar {row['sparsity_pct']:2d}%  K(cpp) {row['k_llamacpp']['groups_per_vector']:.2f}  "
            f"V {row['v']['groups_per_vector']:.2f}  K(tf) {row['k_transformers_order']['groups_per_vector']:.2f}  "
            f"random {row['random_groups']:.2f}",
            flush=True,
        )
    for row in result["pair"]:
        print(
            f"pair   {row['sparsity_pct']:2d}%  groups {row['groups_per_vector']:.2f}  random {row['random_groups']:.2f}",
            flush=True,
        )
    print(f"wrote  {OUT}", flush=True)


if __name__ == "__main__":
    main()
