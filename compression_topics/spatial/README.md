# Spatial pair compression

Adjacent tokens (2i, 2i+1) of one KV head are merged into a pair. Every 16-token
window of every head is compressed to its layer slot's saving, in a fixed-size
block. Calibration allocates different savings across the layer/K-V slots.

## Algorithm

1. **Normalize.** Each key and value vector, as stored in the cache, is divided
   by its length.
2. **Formats.** A pair is dense, or merged: the mean direction
   m = (a + b) / 2 plus 0, 16 or 32 entries of the half-difference
   d = (a - b) / 2 (PR0, PR16, PR32), with a position mask, and one bf16 scale
   per token: its length divided by the length of its rebuilt direction m ± d.
   Counts are per 128 values, so a 256-value head keeps twice as many.
3. **Reconstruction error.** Values use relative squared error. Keys use the
   query-aware error: squared error weighted by the calibrated query energy
   E[q_d^2], which also ranks the residual entries to keep.
4. **Budget per head and window.** When a window's 16th token is written, in the
   prompt or during decode, each head sorts its pairs' merge steps by error added
   per bit saved and takes them until it saves the slot's share of its dense
   bits. Each head's block is (1 - saving) x its dense window bits plus a 2-bit
   tag per pair; unused bits are padding and still count. Nothing carries over
   between windows; the newest, unfinished window stays dense.

The method is registered as `spatial` (`SpatialConfig(query_weights,
query_weights_sha256)`; keys are query-aware when the weights are given).

## Code

- `algorithms/formats.py`: pair formats and bit costs.
- `algorithms/representation.py`: normalization, reconstruction, query weights
  and the reconstruction error.
- `algorithms/windows.py`: per-head window budgets.
- `algorithms/methods.py`: configuration and the cache hook.
- `scripts/study.py`: calibration, allocation, validation and task execution.
- `scripts/ladder.py`: the official sweep.
- `scripts/pair_similarity_profile.py`: read-only pair similarity profiling.

This is an accuracy/storage simulation: tensors are reconstructed in PyTorch.
Simulated compressed byte counts do not mean smaller allocated GPU cache tensors
or a measured hardware speedup.

## Official sweep

`ladder.py --model llama|qwen|codellama` runs the method at 10/20/30/35/40%:
query calibration, single-slot screening and refinement on WikiText training
chunks, greedy allocation across slots, a holdout check that every whole-model
saving is within one point of its budget, then the tasks (C-Eval 5-shot, GSM8K,
HumanEval-instruct; CodeLlama: HumanEval with its body-indent prompt).
Output goes to `figures/official_<model>`. Plans are immutable per output
folder; use a fresh `--out` for a changed protocol.

```bash
sbatch --gres=gpu:4 compression_topics/spatial/scripts/ladder_dgx.sh --model llama
```

## Recorded results

- `figures/official_llama`: Llama 3.1 8B-Instruct, the official method. Its
  records were written while the arm was still called
  `query_pairs_per_head_padded_w16_no_rope`, and with 3-bit format tags (now 2).
- `figures/pair_similarity`: adjacent-pair similarity profile (Llama 3.1 8B,
  WikiText-2).

Earlier variants (quads and eights, shared window budgets, per-head splits,
8-token windows, keys rotated before merging) and their results were removed on
2026-10-26; they are recoverable through Git.
