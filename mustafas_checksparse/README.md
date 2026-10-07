# CheckSparse: greedy per-target and global maps (WikiText calibration only)

Model: meta-llama/Llama-3.1-8B-Instruct, bf16. Tile size: 8.

## How the maps were made
1. **Profile.** For each layer, prune only K or only V of that layer at 25, 50 and 75% and measure WikiText word perplexity. The baseline is 9.080. Results are in `code/llama31_wikitext_tile8_profile.jsonl`.
2. **Greedy allocation** (`code/greedykv_allocation.py`). Each layer's curve becomes chunks with a cost of ΔPPL per unit of sparsity. A min-heap takes the cheapest chunks until the budget is spent.
   - `--scope per-target`: K and V each average exactly the budget over their 32 layers.
   - `--scope global`: K and V share one pool of 64 units, averaging the budget overall. This ends up pruning V much more than K.
3. Run `code/reproduce_maps.sh` to regenerate all 8 maps in `layer_maps/`. They match the files used in the evals byte for byte.

## Applying a map
`code/modeling_llama.py` is a modified HF Llama. At inference, for each layer and each token, `_apply_tile_sparsity` reshapes K (after RoPE) or V to 1024 dims (8 KV heads × 128), splits it into 128 tiles of 8, and zeroes the `floor(128 × pct/100)` tiles with the smallest L1 norm. This happens before the KV-cache update.

Set these keys in the model config:
- `tile_sparsity_enabled = True`
- `tile_sparsity_tile = 8`
- `tile_sparsity_prune_pct_by_target_layer = <contents of a layer_maps JSON>`

## Other files
- `gsm8k_samples_profile20pct.json`: the 264 GSM8K test indices used for evaluation (0-shot, max_gen_toks 512).
- `LAYER_LISTS.md`: all 8 maps as readable K/V layer lists.
