# Pairs-only vs mixed pair/quad (shared pair residual) — results

Llama 3.1 8B Instruct, prefill-only reconstruction simulation, commit 859b9f1
(`figures/pair_quad_shared_sampled`). Method `group_calibrated`: per layer/K/V
slot, either pairs (shared signed residual) or quads (per-token residuals),
r ∈ {0, 4, 8, 16, 32}. Budgets include norms, residual masks, merge flags and
slot header (16-bit values). Calibration: WikiText chunks of 2048 tokens,
dense token PPL 6.7875. Final task: C-Eval valid, 5-shot, 1346 questions,
dense accuracy 54.53%.

## C-Eval (accuracy change vs dense, percentage points)

| budget (of compressed tensors) | keys: pairs | keys: mixed | K+V: pairs | K+V: mixed |
|---|---|---|---|---|
| 10% | +0.45 | +0.89 | +0.67 | +0.15 |
| 20% | +1.56 | +1.11 | +0.45 | −0.45 |
| 30% | −0.59 | −1.34 | −0.15 | −1.49 |
| 40% | **−17.31** | −2.97 | **−20.13** | −5.87 |
| 50% | (infeasible) | −8.32 | (infeasible) | −12.04 |
| 60% | (infeasible) | −17.83 | (infeasible) | −25.26 |

Keys-only budgets are a share of the keys, so the whole-KV saving is half
(keys 40% = 20% of the KV cache). Pairs cannot exceed ~49% per slot.
One C-Eval run has a binomial standard error of about ±1.4 points, so
differences under about ±2 points are noise.

## Findings

1. **Up to 30% is free.** Both policies stay within noise of dense through 30%,
   keys-only and K+V alike. The small gains at 10–20% are noise, not improvement.
2. **40% is a cliff for pairs-only.** Keys −17.3 points, K+V −20.1. To average
   40% with a 49% ceiling, almost every slot runs near its pair maximum (r = 0),
   and the minimum reconstruction cosine drops to 0.41 (keys) / 0.44 (K+V).
   Mixed at 40% loses only −3.0 (keys) / −5.9 (K+V): quads are what make 40%+
   reachable at all.
3. **Pair vs quad per slot depends on budget, very cleanly.** In calibration,
   quads beat the best pair in 6–8 of 32 key slots at 10–30%, but in 28/32 at 40%
   and 31/32 at 49%. Values are closer to a coin flip at low budgets
   (quads win 15–18 of 32 at 10–30%) and also flip to quads at ≥40%.
4. **Below 40%, mixing does not pay off on C-Eval.** At 20–30% mixed is
   0.5–1.3 points below pairs-only (within noise, but consistently on the wrong
   side). The calibration preference for quads in some slots does not
   transfer; per-slot WikiText PPL differences there are tiny (median ΔPPL
   0.001–0.003), smaller than the selection noise.
5. **At equal total KV memory, spread the budget over K and V.**
   20% of the KV cache: keys-only at 40% loses −3.0 (mixed) or −17.3 (pairs);
   K+V at 20% loses −0.45 (mixed) or gains +0.45 (pairs).
   30% of the KV cache: keys-only at 60% loses −17.8; K+V at 30% loses −0.15
   (pairs) or −1.5 (mixed).
6. **Residual choices.** Keys pairs pick r = 32 at 20–30%; at 40% only r = 8 or
   0 fit; at 49% only r = 0. Quads mostly pick r = 32 or 16 up to 50%, r = 8 at
   60%, and r = 0 at 65%+.
7. **Sensitive layers.** Key layers 1 and 14 stay dense up to 20%, and layers
   1–2 are the worst slots at high budgets (ΔPPL 0.13–0.26 at 60–74%). Value
   layers 12, 15, 16 stay dense longest; value layer 0 is the worst at 65%+
   (ΔPPL ≈ 0.46–0.50).
8. **Calibration under-predicts combined damage.** Single-slot ΔPPL at 40–49%
   is small (median 0.006–0.012), yet all slots together collapse C-Eval. Errors
   compound across layers; per-slot calibration sums are optimistic at high
   budgets.

## Implications for the next study (group_rd)

- The useful range is roughly 30–50% per tensor. Below that, pairs-only is
  already free; above it, everything degrades.
- The pair→quad switch at ~40% is exactly the decision a per-block menu can
  make inside a layer, instead of per slot.
- Compressing K and V together beats pushing keys harder; group_rd should be
  extended to values (no RoPE alignment, no query weighting) before the final
  comparison.
- A whole-model check at the target budgets is needed, since per-slot
  calibration misses the compounding seen at 40%.
