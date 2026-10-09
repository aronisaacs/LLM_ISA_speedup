# Token importance in format decisions (menu `flag_d`, a format per 4 tokens)

Decisions use attention received from even-position queries; all arms are scored by each token's
error times the attention it receives from odd-position queries (held out). Median over layers;
positive = less error. Oracle decides with the scoring weights themselves (upper bound).
"First block dense" ignores importance except that the first 4 tokens (the attention sink) stay exact.

## keys (query error)

| saving | uniform vs fixed | importance vs fixed | first block dense vs uniform | importance vs uniform | oracle vs uniform |
|---|---|---|---|---|---|
| 30% | +6.5% | +61.6% | +2.1% | +59.2% | +60.1% |
| 35% | +21.0% | +65.9% | +2.9% | +56.7% | +57.7% |
| 40% | +12.8% | +60.9% | +3.7% | +55.0% | +56.2% |
| 45% | +30.9% | +68.4% | +3.6% | +53.7% | +54.9% |
| 50% | +21.1% | +61.8% | +2.7% | +52.7% | +53.9% |
| 55% | +9.3% | +55.7% | +2.9% | +51.4% | +52.1% |
| 60% | +18.4% | +58.2% | +2.7% | +49.5% | +50.5% |
| 65% | +44.1% | +69.2% | +2.2% | +46.8% | +47.9% |
| 70% | +26.7% | +56.0% | +1.8% | +42.2% | +43.2% |

## values (squared error)

| saving | uniform vs fixed | importance vs fixed | first block dense vs uniform | importance vs uniform | oracle vs uniform |
|---|---|---|---|---|---|
| 30% | +5.9% | +63.3% | +0.2% | +61.2% | +63.1% |
| 35% | +18.4% | +66.5% | +0.3% | +59.0% | +60.5% |
| 40% | +13.3% | +62.6% | +0.2% | +57.6% | +58.8% |
| 45% | +23.1% | +66.0% | +0.2% | +56.3% | +57.4% |
| 50% | +16.6% | +61.4% | +0.3% | +54.3% | +55.9% |
| 55% | +10.7% | +57.1% | +0.4% | +52.1% | +53.4% |
| 60% | +15.3% | +57.6% | +0.5% | +49.4% | +50.5% |
| 65% | +21.1% | +58.3% | +0.7% | +46.0% | +47.4% |
| 70% | +11.5% | +46.6% | +0.8% | +39.3% | +40.4% |
