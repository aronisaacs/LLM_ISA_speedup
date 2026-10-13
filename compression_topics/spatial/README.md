# Spatial compression

Adjacent token vectors are merged **within each KV head**. Heads choose their
own formats; the heads of one layer's keys share a controller, and that layer's
values have a separate controller. Calibration allocates different byte budgets
across these layer/K-V slots.

## Algorithm

All current spatial variants share RoPE alignment for keys, directional
reconstruction with BF16 per-token scales, and conservative packed-format byte
accounting. Query awareness uses calibrated query energy both to select key
residual features and to score reconstruction error. Its error is absolute
weighted squared error; ordinary key/value scoring is relative squared error.
Values do not use query weights. No attention-importance weighting or shared
quad/eight masks is included in the current path.

The default menu is dense, pairs with 0/16/32 residual entries, quads with
0/16 entries per token, and eights with 0/16 entries per token. A bottom-up tree
compares larger groups against smaller ones inside each eight-token window.

- `spatial_online` fits lambda from the first 32 observed tokens, then updates
  it every eight-token window. State survives cache appends, so it supports
  compression during decode. Unfinished windows stay dense.
- `spatial_chunked` keeps no state between windows: every completed eight-token
  window, in the prompt and during decode, is compressed on its own to the
  slot's saving. Within the window's bit budget the heads' formats are chosen
  together for the least total loss (`algorithms/chunked.py`). Unfinished
  windows stay dense; there is no startup.
- `spatial_offline` fits one price over the whole observed prompt; generated
  tokens stay dense. This is a diagnostic/reference method.
- `prefill_only=True` explicitly leaves continuation/decode appends dense,
  preserving the protocol used by the recorded presentation experiments.

Fixed residuals restrict the menu through `residual_counts`, one count per
enabled group size. `merge_sizes=[4]` permits only quad merges; failed quads stay
dense without pair fallback. These are configuration choices in the same kernel,
not separate compression implementations.

## Code boundaries

- `algorithms/formats.py`: menus, bit costs, and format validation.
- `algorithms/representation.py`: RoPE alignment, reconstruction, query weights,
  and the error objective.
- `algorithms/windows.py`: per-head format choices and reconstruction.
- `algorithms/controller.py`: offline price fit and causal online feedback.
- `algorithms/chunked.py`: per-window budget solver for `spatial_chunked`.
- `algorithms/methods.py`: configuration and cache integration.
- `scripts/study.py`: common calibration, allocation, validation, and accuracy
  execution. Candidate generation remains specific to each experiment.
- `scripts/comparison.py`: common archived-reference and holdout-quality gates
  for simplification experiments.
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
- `gate_stats` records algorithm/controller observations over the full evaluation.
  These counters are not the source of prefix budgets in new records.

Calibration reads explicit slot bytes. Historical files without slot snapshots
are interpreted only by `calibration_slot_bytes`; the old streaming
prefill-only conversion lives there. Historical result files are never rewritten.
Chunk extension preserves storage scopes, scoring definitions, and maximum
controller deviations. Dense baselines use the same slot schema as compression.

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

The fixed-hierarchy and quad-only simplification studies use two screen chunks,
twelve refinement chunks, and sixteen holdout chunks. They reuse the archived
query weights and reevaluate the archived adaptive policy on the same holdout.
Their heuristic gate requires savings within one percentage point and no more
than a 2% perplexity increase. It does not guarantee accuracy equivalence. Final
comparisons report both accuracy and actual bytes, with a one-point accuracy
screening tolerance. Historical reference screening used four chunks.

## Entry points

- `ladder.py` / `ladder_dgx.sh`: pairs, hierarchy, query hierarchy, query pairs (30/40/50%);
  `--algorithm online|chunked|offline`.
- `fixed_rungs.py` / `fixed_rungs_dgx.sh`: no-residual and fixed-residual pairs
  (10/20/30/40%).
- `fixed_hierarchy.py` / `fixed_hierarchy_dgx.sh`: one residual per enabled group
  size per layer/K-V slot, targeting 40%.
- `quad_only.py` / `quad_only_dgx.sh`: dense or a fixed quad format, targeting 40%.
- `pair_diagnostic.py` / `pair_diagnostic_dgx.sh`: bounded dense-state diagnostic,
  not a full accuracy experiment or replacement runtime algorithm.
- `pair_similarity_profile.py`: profiling/capture utility.

Output plans are immutable. Resume with the original script and output directory
only when its plan matches; use a fresh directory for a changed protocol.
Launchers run regression checks first and print stage/preliminary progress.
Slurm resource requests may be overridden at submission. Multiple GPUs speed
parallel calibration configurations; a single accuracy configuration still runs
on one GPU.

## Recorded evidence and safe synchronization

`clean_online_ladder_*` and `online_fixed_rungs_*` retain the pre-cleanup
presentation results and their original policy/sample identities. New field and
method names do not retrospectively change those records. Accounting audits
verify their summaries against raw task results; this is not a re-execution of
the historical model. Old algorithm revisions remain recoverable through Git.

Do not update a DGX checkout while jobs may still launch workers from it. Finish
and save that run first, or use a separate checkout. Local cleanup never changes
the running remote checkout automatically. Results, reports, and untracked
outputs are retained unless explicitly retired.
