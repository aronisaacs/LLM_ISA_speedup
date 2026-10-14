# isa_sim

The compression methods from `compression_topics/` running in llama.cpp, and
later in gem5. The PyTorch code stays the reference; everything here is checked
against it.

## Versions

| | |
|---|---|
| llama.cpp | `../llama.cpp`, master `0504396`, local branch `kv-vector-compress` |
| gem5 | `../gem5`, `stable` `f5c5a6e`, not built yet |

The llama.cpp changes are also saved as `llama_cpp/0001-vector-compressed-kv-types.patch`
(`git am` it onto `0504396`), so they are visible without the llama.cpp clone.

## Vector compression cache types

Two new ggml types, used as `-ctk vck50 -ctv vcv50` (flash attention on):

| type | for | stored per 128-wide head vector | bytes | net saving |
|---|---|---|---|---|
| `vck50` | keys | 32 of 64 RoPE pairs (fp16) + 64-bit pair mask | 136 | 46.9% |
| `vcv50` | values | 64 of 128 scalars (fp16) + 128-bit mask | 144 | 43.8% |

They are `vector_compress_pair` (keys) and `vector_compress` (values) at
`prune_pct=50`. GGUF keys keep RoPE pairs adjacent, (2i, 2i+1), so a kept pair
is one 32-bit lane. Every token is compressed, including the newest.
llama.cpp's Hadamard rotation of quantized caches is switched off for these types.

On the CPU, compression is the KV write (`from_float`, through `set_rows`), q·k
runs over the kept pairs only (`ggml_vec_dot_vck50_f16`), and values are
decompressed to fp32 before the weighted sum. All of it is scalar C for now.

## Golden check

```bash
L=../llama.cpp
clang -O2 -I$L/ggml/include isa_sim/golden/vc_roundtrip.c -L$L/build-cpu/bin \
  -lggml-base -lggml-cpu -Wl,-rpath,$PWD/$L/build-cpu/bin -o isa_sim/golden/build/vc_roundtrip
conda activate llm_isa_local
python isa_sim/golden/check_vector.py --bin isa_sim/golden/build/vc_roundtrip
```

It compresses 20,000 vectors per type with llama.cpp and with the PyTorch
functions, and compares the reconstructions and q·k. Inputs are fp16, so kept
values must match exactly; vectors that differ only by which of two equal
magnitudes was kept count as matching (`torch.topk` breaks ties in no fixed order).

## Perplexity pilot

`llama-perplexity`, Llama 3.1 8B Instruct Q4_K_M, WikiText-2 test, 8 chunks of
512 tokens, CPU (M1 Pro, 8 threads). Logs in `results/`.

| K | V | PPL |
|---|---|---|
| f16 | f16 | 8.03 ± 0.43 |
| vck50 | f16 | 11.98 ± 0.65 |
| f16 | vcv50 | 8.26 ± 0.44 |
| vck50 | vcv50 | 13.13 ± 0.70 |

`isa_sim/data/wiki.test.raw` is written from the local Hugging Face cache and is
not committed.
