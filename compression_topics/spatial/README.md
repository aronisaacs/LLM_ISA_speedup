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
- **Online hierarchy:** choose dense, pairs, quads or eights within an
  eight-token window, using the short residual menu and online byte controller.

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

### Online pair/quad/eight accuracy

`scripts/hierarchy_accuracy.py` and `scripts/hierarchy_accuracy_dgx.sh` evaluate
one online hierarchy with separate token masks at total KV savings of
10/20/30/40/50/60/70/80%. Results go to `figures/hierarchy_accuracy/`.
Keys and values have separate, independently calibrated per-layer allocations.
Each slot is calibrated at seven controller targets (10/25/40/55/70/80/85%):
448 settings across 32 layers and two targets, using 4 train screening chunks
and 12 disjoint train refinement chunks. This is a target grid, not a search
over format combinations or controller hyperparameters.

The common greedy allocator uses measured calibration bytes for this method,
keeps each calibrated local policy intact, and records discrete overshoot.
The new 16-chunk validation must achieve total savings within one percentage
point of each global target before C-Eval runs. C-Eval reports its own actual
savings and flags deviations. Its dense baseline may reuse the earlier full
evaluation only after an exact model/prompt/protocol identity and question-count
check. Compressed results and calibration are never reused from other methods.

The short menu, 32-token startup and feedback parameters are frozen. There are
no shared-mask, offline, query-weighted, attention-weighted or quantized arms.
Decode appends remain dense. The launcher uses four GPUs; `--from`/`--through`
support resume, preliminary results appear during calibration and accuracy,
and quiet timing records accompany each evaluation attempt.

```bash
sbatch compression_topics/spatial/scripts/hierarchy_accuracy_dgx.sh
```

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

## Composing spatial choices

New experiments can use `algorithms/composed.py:SpatialConfig` and the registered
`spatial_composed` method. The choices are separate fields, not new algorithm
names for every combination:

- `representation`: pairs, fixed quads, or an online pair/quad/eight hierarchy.
- `decision`: fixed global ranking, online feedback, or offline price search.
- `residual_entries`: one count for fixed ranking; adaptive pairs use the
  existing short `residual_menu` of 0/8/16/32. The hierarchy uses the eight-format
  preset described below.
- `mask_layout`: separate or shared masks for the online hierarchy.
- `select_by`: deviation or calibrated query energy for residual feature ranking.
- `distortion`: relative reconstruction error or query-energy-weighted error.
- `importance`: none or attention received within the available prefill.

`representations.py` builds head-local means, residuals and norm-restored
candidates. `scoring.py` owns feature scores and distortion. `decisions.py` owns
fixed ranking, online control and offline search. `storage.py` owns byte formulas.
The composed entry point connects those components and existing cache integration.

Query feature selection and query distortion are independent; both require
checksum-verified calibration weights and apply to keys only. Configure values
in a separate pipeline step. Prefill attention importance is supported for fixed
and offline decisions; it is rejected for online decisions because later prefill
queries would influence earlier choices. All modes leave decode appends dense.
`representation='hierarchy'` adds online decisions inside eight consecutive
tokens. `max_group=2/4/8` selects nested menus. The eight-format preset is dense
pairs, pairs with 0/16/32 residual entries, quads with 0/16, and eights with 0/16.
A bottom-up tree compares a larger merge against its smaller children, without
enumerating partitions. Choices span KV heads; vectors remain head-local.
The existing pair algorithm and its 0/8/16/32 menu are unchanged.

The hierarchy currently uses ordinary reconstruction error, no attention or
query weighting, a 32-token startup, and feedback once per eight-token window.
Controller memory stays in token units. It models BF16 mean/residual/scale
storage, one signed residual mask per pair or separate/shared token masks for larger
groups, packed 3-bit terminal tags, and a 16-bit slot header. Tree costs include
unpacked tag bits; final byte accounting rounds the whole tag stream to bytes.
Tail tokens outside complete eight-token windows stay dense. This is a PyTorch
reconstruction prototype, not an encoded cache or a hardware timing benchmark.
Decode appends still stay dense. A shared mask is chosen per group and KV head
from the summed squared deviations of all tokens; their residual values and
norm scales remain separate. It saves only mask bytes, not residual values.

`scripts/hierarchy_pilot.py` reuses two existing dense 8B captures, probes five
layers' keys and values at 32/128/512/1536-token prefixes, and tests 30/40/50/60/70/80%
per-slot targets. It reports reconstruction error, exact modeled bytes, format
use and controller drift, marks infeasible targets, and flags unmatched savings.
It compares pairs once, pairs/quads with separate and shared masks, and
pairs/quads/eights with separate and shared masks. Same-format comparisons
also isolate the mask's reconstruction penalty and byte saving before control.
This diagnostic has no accuracy calculation or layer-budget allocation; each
eventual full-model variant still needs its own calibration. Submit with
`scripts/hierarchy_pilot_dgx.sh`; preliminary results and quiet timing records
are written alongside the final summary. Missing captures cause an explicit
error rather than silently launching model inference.

Existing `online_pairs` and `presentation_spatial` names remain thin compatibility
entry points for recorded studies. They use these shared components, preserve
their historical rounding/accounting conventions, and keep their old signatures
and defaults. The running study's configurations are not migrated automatically.

Example: offline pairs with query-aware residual features but ordinary distortion:

```python
from compression_topics.spatial.algorithms.composed import SpatialConfig

keys = SpatialConfig(
    decision="offline",
    representation="pairs",
    select_by="query",
    distortion="relative",
    query_weights="path/to/calibrated_query_weights.pt",
    query_weights_sha256="checksum_from_calibration",
)
policy = keys.pipeline(0.30, k_layers=[12], v_layers=[])
```

Each distinct composition still needs its own calibration; components being
reusable does not make their measured quality curves interchangeable.
