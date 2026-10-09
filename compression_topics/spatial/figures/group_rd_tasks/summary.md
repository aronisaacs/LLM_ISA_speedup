# group_rd stage B: C-Eval and WikiText

Menu `flag_d`: D, P32/d, P32+32, P16/0, Q32, Q16, Q8, Q0; a format per 4 tokens; same saving in every key and value layer.

C-Eval valid, 5-shot; dense 54.53%. Change in points; measured whole-KV saving in brackets.

| saving | plain residuals | attention-aware residuals | pair/quad per layer (previous) | pairs only (previous) |
|---|---|---|---|---|
| 30% | -2.23 (30.0%) | -0.89 (30.0%) | -1.49 | -0.15 |
| 40% | -2.67 (40.0%) | -2.45 (40.0%) | -5.87 | -20.13 |
| 50% | -13.45 (50.0%) | -9.58 (50.0%) | -12.04 | – |

WikiText word perplexity; dense 8.825.

| saving | plain residuals | attention-aware residuals |
|---|---|---|
| 30% | 9.206 (+0.381) | 9.197 (+0.371) |
| 40% | 9.675 (+0.850) | 9.669 (+0.844) |
| 50% | 11.020 (+2.194) | 11.073 (+2.247) |
