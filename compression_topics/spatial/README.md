# Spatial compression

Adjacent token vectors are merged **within each KV head**, in fixed windows of
eight consecutive tokens. Each window of a layer's keys (or values) is
compressed to that layer slot's saving; calibration allocates different savings
across the layer/K-V slots.

## Algorithm

1. **Preprocessing.** Keys undo RoPE by each token's offset inside its group, so
   the members share one frame. Every vector is split into a unit direction and
   a bf16 norm.
2. **Formats.** Per head, a window is covered by dense pairs and merged groups:
   pairs with 0/16/32 residual entries, quads and eights with 0/16 entries per
   token (`max_group` 2, 4 or 8). A merged group stores the mean direction plus
   the largest entries of each deviation, with a position mask.
3. **Loss.** Values use relative squared error. Keys may use the query-aware
   loss: squared error weighted by the calibrated query energy E[q_d^2], which
   also ranks the residual entries to keep.
4. **Budget per window.** When a window's eighth token is written, in the prompt
   or during decode, the heads' formats are chosen together for the least total
   loss within (1 - saving) x the window's dense bits. Nothing carries over
   between windows; unfinished windows stay dense.

The method is registered as `spatial` (`SpatialConfig(max_group, query_aware,
query_weights, query_weights_sha256)`).

## Code

- `algorithms/formats.py`: menus and bit costs.
- `algorithms/representation.py`: RoPE alignment, reconstruction, query weights,
  and the loss.
- `algorithms/windows.py`: candidate formats and the per-window budget solver.
- `algorithms/methods.py`: configuration and the cache hook.
- `scripts/study.py`: calibration, allocation, validation and C-Eval execution.
- `engine/eval_runner/storage.py`: explicit byte scopes and the isolated reader
  for historical storage records.

This remains an accuracy/storage simulation: tensors are reconstructed in
PyTorch. Simulated compressed byte counts do not mean smaller allocated GPU
cache tensors or a measured hardware speedup.

## Storage contract

New chunk results have `storage_contract_version=2`:

- `storage` is **prefill-only** when continuation NLL is scored. It contains
  per-layer K/V `slots` and whole-model `targets`.
- `cache_storage` records the **entire cache**, including continuation tokens.
- `gate_stats` records algorithm observations over the full evaluation.
  These counters are not the source of prefix budgets in new records.

Calibration reads explicit slot bytes. Historical files without slot snapshots
are interpreted only by `calibration_slot_bytes`; the old streaming
prefill-only conversion lives there. Historical result files are never rewritten.
Chunk extension preserves storage scopes and scoring definitions. Dense baselines use the same slot schema as compression.

## Experiment protocol

Each arm calibrates its own layer policies on WikiText training chunks, using
separate screening and refinement chunks. A common measured-byte allocator
produces whole-model budgets. Validation uses separate held-out chunks before
full 1,346-question five-shot C-Eval. Completed outputs are reused only when
model, policy, and sample identities match.

A candidate that misses its local byte target is recorded in
`*_rejected_candidates.json` and excluded. Identity, sample, or accounting errors
still fail immediately. Whole-model validation remains strict and is rechecked
when resuming directly at accuracy.

## Entry points

- `ladder.py` / `ladder_dgx.sh`: pairs, hierarchy, query pairs and query
  hierarchy at 30/40/50%. Pairs cannot reach 50%; their 50% point merges every
  pair with no residual.
- `pair_similarity_profile.py`: profiling/capture utility.

Output plans are immutable. Resume with the original script and output directory
only when its plan matches; use a fresh directory for a changed protocol.
Launchers run regression checks first and print stage/preliminary progress.
Slurm resource requests may be overridden at submission. Multiple GPUs speed
parallel calibration configurations; a single accuracy configuration still runs
on one GPU.

```bash
for arm in query_hierarchy query_pairs; do
  sbatch --gres=gpu:4 --job-name=ladder_${arm} compression_topics/spatial/scripts/ladder_dgx.sh \
    --arms "$arm" --out compression_topics/spatial/figures/ladder_${arm}
done
```

## Recorded evidence and safe synchronization

Folders under `figures/` were produced by earlier versions of the code, including
the λ-controller (`clean_online_ladder_*`, `online_fixed_rungs_*`) and the first
fixed-window runs (`chunked_*`). They keep their original policy/sample
identities and cannot be resumed with the current scripts. Old algorithm
revisions remain recoverable through Git.

Do not update a DGX checkout while jobs may still launch workers from it. Finish
and save that run first, or use a separate checkout. Local cleanup never changes
the running remote checkout automatically. Results, reports, and untracked
outputs are retained unless explicitly retired.
