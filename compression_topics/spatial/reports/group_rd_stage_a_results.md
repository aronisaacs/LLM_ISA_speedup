# group_rd stage A — results

Commit 17a169c (`figures/group_rd_offline`). Llama 3.1 8B Instruct, 16 WikiText
calibration chunks of 2048 tokens, keys and values of all 32 layers. Offline:
no perplexity or task runs. Each configuration's distortion is compared with
another at equal stored bits (every bit counted, including mode codes).
Numbers are medians over the 32 layers; "+x%" = x% less distortion.

## 1. Per-block choice vs today's best per-layer setting

| saving | keys (query-weighted): 41 modes | keys: 8-mode default | values (squared): 41 modes | values: 8-mode default |
|---|---|---|---|---|
| 10% | −0.1% | −11.5% | −0.4% | −9.1% |
| 20% | +1.6% | −6.6% | +1.0% | −4.8% |
| 30% | +6.2% | −0.9% | +5.8% | +0.9% |
| 35% | +22.6% | +16.3% | +16.5% | +12.6% |
| 40% | +17.3% | +9.7% | +12.7% | +8.2% |
| 45% | +31.1% | +24.4% | +22.5% | +19.0% |
| 50% | +22.5% | +16.4% | +15.2% | +12.4% |
| 60% | +21.0% | +19.5% | +16.6% | +15.7% |
| 70% | +19.3% | +19.3% | +11.6% | +11.7% |

- Below 30% there is nothing to gain: the best fixed setting there is pairs
  with r = 32 in every layer, and the C-Eval runs already showed that range is
  free. Small menus lose there (see 2).
- From 35% up, per-block choice cuts distortion by 13–31% (keys) and 12–23%
  (values). The gain is consistent across layers: at 40%, every key layer
  gains 14–22%, every value layer 7–20%.
- The best fixed setting is pairs r32 at 30%, quads r32 at 40%, quads r16 at
  50% in essentially every layer, matching the C-Eval-calibrated choices.
- Gains vary non-monotonically with saving (e.g. 35% > 40% < 45%) because the
  fixed baseline is strongest just where one of its settings fits exactly.

## 2. How many modes are needed

The full menu mostly uses Q16, Q32, Q0, Q8 and P32+32 (keys and values alike);
mixed pair modes (P32+0, P0+32, P16+32 …) make up the rest.

| saving | keys: default 8 vs 41 | keys: data top-8 vs 41 | values: default 8 vs 41 | values: data top-8 vs 41 |
|---|---|---|---|---|
| 10% | −11.4% | −23.4% | −8.6% | −18.1% |
| 30% | −7.6% | −6.2% | −4.5% | −4.3% |
| 40% | −9.6% | −1.6% | −5.2% | −1.2% |
| 50% | −7.3% | −0.9% | −3.2% | −0.2% |
| 60% | −1.9% | −0.3% | −1.1% | +0.1% |

- The data-picked top-8 (keys: D, Q16, Q32, Q0, Q8, P32+32, P32+0, P32+16) is
  within 1–2% of all 41 modes at 40–60%, but bad at low savings because it was
  picked at 50% and has no half-dense mode.
- The current default (D, Pd+32, P32+32, P8+8, P0+0, Q16, Q8, Q0) is better at
  low savings but gives up 5–10% at 40–50%; it lacks Q32 and keeps P8+8/P0+0,
  which are rarely chosen.
- Top-6 ≈ top-8 at 40%+: 6 modes may be enough at those budgets.

## 3. Coarser decisions (one mode per 16 or 64 tokens)

Saving given up at equal distortion, default menu:

| saving | keys 16 | keys 64 | values 16 | values 64 |
|---|---|---|---|---|
| 30% | 1.6 pts | 2.2 pts | 1.1 pts | 1.5 pts |
| 40% | 1.2 pts | 1.6 pts | 1.0 pts | 1.3 pts |
| 50% | 1.6 pts | 2.1 pts | 1.1 pts | 1.3 pts |

One mode per 64-token page costs only about 1–2 percentage points of saving.

With per-4 decisions a 64-token page uses on average 2.5–3.7 distinct modes
(keys, query-weighted) or 1.9–2.8 (values); fewer than 2% of pages use 6+.

## 4. Index-coded residual positions

Storing positions of r ≤ 16 residuals as 7-bit indices instead of a 128-bit
mask is worth +2 to +7% less distortion at 40–70%, at no cost below.

## Recommendations

1. Replace the default menu with a data-driven one that keeps one or two low
   saving modes: e.g. D, P32+d (or Pd+32), P32+32, P32+0, Q32, Q16, Q8, Q0.
   Check it with `--stage analyze` (minutes, no GPU) before Stage B.
2. Turn on `mask='auto'`.
3. One mode per 64 tokens is a cheap hardware simplification (~1–2 pts).
4. Stage B: per-slot sensitivity weights, then C-Eval at 30/40/50% against
   pairs-only, mixed per-layer and vector compression. Stage A measures
   reconstruction error only; whether 15–30% less error closes the 40–50%
   C-Eval gap is what Stage B decides.
