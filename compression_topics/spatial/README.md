# Spatial compression

The current research path uses adjacent tokens within each KV head. Keys align
RoPE before merging; values do not rotate. Layer allocation assigns separate
key/value compression targets, shared across the heads. All current evaluations
compress prefill; newly appended decode tokens remain dense.

## Method names

- **Dense baseline:** no compression.
- **Merge-only pairs:** globally select the lowest-error adjacent pairs, without
  residuals or norm restoration. This is the clean presentation study's rung 1.
- **Fixed-residual pairs:** calibrate one residual count per layer/target, then
  globally select merges. Norm restoration is included. This is clean rung 2.
- **Online adaptive pairs:** select dense or 0/8/16/32 residual features for each
  pair across heads. Fit the initial price on 32 tokens, then adjust it locally
  using byte-saving feedback.
- **Offline adaptive pairs:** the same representations and reconstruction as
  online, with a price fitted across the full prefill and discrete boundary
  repair. This is an offline reference, not an exact accuracy optimum.

The clean presentation study's historical rungs 3 and 4 mean query-weighted
residuals and fixed pairs/quads. Those numbers are not the current method ladder.

## Clean studies

### Presentation results

`scripts/presentation_ladder_study.py` and `scripts/presentation_ladder_dgx.sh`
produce the clean four-rung study. Completed data lives in
`figures/presentation_ladder_clean_50/`. Preserve its manifests, calibration,
selections and final results together so reported numbers remain reproducible.

### Current independently calibrated comparison

`scripts/online_pairs_accuracy.py` and `scripts/online_pairs_accuracy_dgx.sh`
compare dense, fixed-residual, online-adaptive and offline-adaptive pairs.
Outputs go to `figures/online_pairs_accuracy_offline/`.

Each compressed method is calibrated independently on identical WikiText train
chunks: 4 screening chunks and 12 disjoint refinement chunks. The common
allocator uses each method's own measured loss curves. A 16-chunk validation
check requires all methods to be within one percentage point of their global
10/20/30/40% targets and of each other before full 5-shot C-Eval starts.
C-Eval results also report actual savings and flag unmatched comparisons.

Submit from the DGX repository root:

```bash
mkdir -p logs
sbatch compression_topics/spatial/scripts/online_pairs_accuracy_dgx.sh
```

The study supports `--from` and `--through` for resuming stages. Plans are
immutable within an output directory; use a new `--out` for a changed protocol.
Calibration writes per-chunk preliminary results. Accuracy writes preliminary
summaries as configurations complete. The score index is local to the study.

## Implementation and retained helpers

- `algorithms/presentation_spatial.py`: clean fixed pair/quad reconstruction and
  accounting used by the presentation study and fixed-residual comparison.
- `algorithms/online_pairs.py`: common adaptive pair representations, online
  controller, offline selection and model integration.
- `algorithms/storage.py`: shared representation byte formulas.
- `scripts/online_pairs_continuation.py`: retained pilot helper used by the
  model-integration tests imported by the current launcher. Its superseded
  result files have been removed.

Older pair/group implementations remain where shared code or tests depend on
those modules. They are not the entry points for the current comparison.
Superseded exploratory ablations, controller proxies, shared-mask checks,
timing benchmarks and the old integrated adaptive study have been removed.
Recovering them, if necessary, should use Git history rather than new copies.

## Shared calibration and allocation interfaces

Studies supply candidate policies and sample schedules. The reusable workflow in
`engine/layer_select/study.py` provides:

- `measure_candidates`: matched model/protocol/chunk checks, actual byte
  measurements, loss relative to dense, uncertainty and calibration provenance.
- `select_settings`: group candidates by layer/target/budget and choose settings
  within a method family, with a study-specified tie-break rule.
- `allocate_settings`: feed a method's local loss curves to the common greedy
  allocator and assemble a whole-model policy.
- `require_matched_savings`: validate actual whole-model savings across methods
  before accuracy evaluation.

Local candidate records distinguish `target_saving` (requested compression)
from `measured_saving` (observed calibration bytes). Assignment records also have
`allocation_saving` (the allocator's assumption for the assigned policy).
Historical `budget`/`compression` fields remain for saved-result compatibility.
A budget adjustment between calibrated points is marked `directly_calibrated:
false`; its original measured savings are preserved, not relabeled as a new
measurement. Whole-model savings are measured again on validation and C-Eval.

Per-slot perplexity is whole-model perplexity with only that slot compressed.
It estimates local sensitivity; it does not predict the combined policy's
perplexity. Current allocation averages savings across equal-sized Llama slots.
Supporting unequal slot sizes requires byte-weighted allocation rather than
assuming this average is a general whole-model byte fraction.
