# Spatial compression

Spatial compression merges adjacent tokens within each KV head. There are two
methods:

- **`spatial_offline`:** sees the whole prompt, fits one price per layer and
  K/V target so the prompt meets its byte target, and compresses at prefill.
  Tokens generated during decode stay dense. This is a reference point.
- **`spatial_online`:** compresses the cache as it grows, at prefill and during
  decode. Each completed 8-token window is compressed in place once its last
  token is appended; the window still being filled stays dense. A price fitted
  on the first 32 tokens is nudged every window toward the byte target.

C-Eval and the WikiText chunk scores have no decode, so both methods only
compress prompt tokens there.

## Representation

Per KV head, within each 8-token window:

1. **Preprocessing.** Keys undo RoPE by each token's offset inside its group, so
   neighbours are compared in the same frame. Every token is normalized to a
   unit direction and keeps a bf16 norm.
2. **Groups.** A bottom-up tree chooses dense tokens, pairs, quads or eights
   (`max_group` 2, 4 or 8). A merged group stores the mean direction plus the
   top-r residual entries and a bitmask. Pairs choose r from {0, 16, 32};
   quads and eights from {0, 16}.
3. **Accounting.** Each group carries a 3-bit tag and each slot a 16-bit header
   (`formats.ACCOUNTING`).
4. **Price.** One price λ is shared by the heads of a layer and target. Each head
   minimises `loss + λ · bits / dense window bits` on its own.

The optional **query-aware** loss (keys only) weights each RoPE plane by its
calibrated query energy E[q²], both when choosing residual entries and when
scoring a group.

## Code

- `algorithms/formats.py`: bit costs and format menus.
- `algorithms/representation.py`: preprocessing, reconstruction and loss.
- `algorithms/windows.py`: per-head tree choices for each window at a price.
- `algorithms/controller.py`: the online price controller and the offline
  price fit.
- `algorithms/methods.py`: `SpatialConfig`, the two registered methods and
  the cache hook that runs `spatial_online` during decode.

```python
from compression_topics.spatial.algorithms.methods import SpatialConfig

kv = SpatialConfig(algorithm='online', max_group=8, query_aware=True,
                   query_weights=path, query_weights_sha256=digest).pipeline(
    0.3, k_layers=[0, 1], v_layers=[])
```

## Studies

`scripts/study.py` holds the shared protocol: screen and refine per-slot
savings on WikiText train chunks, allocate a global budget greedily across
layers, validate on WikiText validation chunks, then run 5-shot C-Eval.

- `scripts/ladder.py`: arms `pairs`, `hierarchy` and `query_hierarchy` at global
  budgets 30/40/50 percent; `--algorithm online|offline` (default online). The
  pairs 50 percent point is the minimum-storage endpoint (online only).
- `scripts/fixed_rungs.py`: `no_residual` and `fixed_residual` pair baselines at
  10 to 40 percent.
- `scripts/pair_similarity_profile.py`: pair similarity profile.

DGX launches (one job per arm):

```bash
for arm in pairs hierarchy query_hierarchy; do
  sbatch --job-name=ladder_${arm} compression_topics/spatial/scripts/ladder_dgx.sh \
    --arms "$arm" --out compression_topics/spatial/figures/ladder_${arm}
done
for arm in no_residual fixed_residual; do
  sbatch --job-name=rungs_${arm} compression_topics/spatial/scripts/fixed_rungs_dgx.sh \
    --arms "$arm" --out compression_topics/spatial/figures/fixed_rungs_${arm}
done
```

Each job runs `unit_tests.test_spatial_algorithms` and
`unit_tests.test_spatial_studies` first.

## Past results

Folders under `figures/` were produced by earlier versions of the code and are
kept as recorded results. `clean_online_ladder_{pairs,hierarchy,query_hierarchy}`
match the current online method on C-Eval (the decisions are identical on
prompts). Their plans cannot be resumed with the new scripts.

## Small fixed/adaptive pairs diagnostic

`pair_diagnostic.py --execute` compares recorded 40% allocations, then captures
five dense layers (0/8/16/24/31) on two train and two validation WikiText prompts,
each 512 prefix tokens plus 32 observed continuation queries. At each selected
K/V slot it uses the fixed arm's layer target for both residual policies and
measures reconstruction, attention logits, attention output, and output after
that layer's learned output projection. Only one K or V slot is altered at a
time in these captured-state measurements. This is not downstream perplexity or
accuracy evaluation.

Variants: fixed/adaptive with the existing causal controller; fixed/adaptive
with a static price fitted on the independent training captures; and an
analysis-only adaptive whole-prefix price reference fitted to the fixed
controller's actual bytes on each validation capture. The last is deliberately
noncausal, used to diagnose the error objective at similar bytes, not proposed
as runtime. All actual byte gaps are recorded; frozen/controller variants must
not be treated as matched if their savings differ. No query weighting or token
importance is added. Existing runtime settings are unchanged.

The one-GPU `pair_diagnostic_dgx.sh` has a 30-minute wall-time cap and writes
`preliminary.json` after each slot and `summary.json` on completion. Running the
Python script without `--execute` only compares the existing results.
