# group_rd stage A

Gain = median over layers of 1 - D(config) / D(reference) at equal stored bits
(positive: config has less distortion). Lost saving = saving a coarser decision
gives up at the distortion per-4-token decisions reach, default menu.

Default menu: D, Pd+32, P32+32, P8+8, P0+0, Q16, Q8, Q0
Top 6 menu (k): D, Q16, Q32, Q0, Q8, P32+32
Top 8 menu (k): D, Q16, Q32, Q0, Q8, P32+32, P32+0, P32+16
Top 6 menu (v): D, Q16, Q32, Q0, P32+32, Q8
Top 8 menu (v): D, Q16, Q32, Q0, P32+32, Q8, P0+32, P32+0

## keys, distortion cosine

| saving | 41 modes vs best fixed | 8-mode default vs best fixed | 8-mode default vs 41 | top 6 vs 41 | top 8 vs 41 | auto masks (default menu) | g16 lost | g64 lost |
|---|---|---|---|---|---|---|---|---|
| 5% | -1.2% | -12.9% | -11.9% | -27.9% | -27.9% | +0.0% | +1.4 pts | +1.8 pts |
| 10% | -0.6% | -9.9% | -9.6% | -20.5% | -20.5% | +0.0% | +2.0 pts | +2.7 pts |
| 15% | -0.2% | -7.9% | -7.9% | -15.3% | -15.3% | +0.0% | +2.2 pts | +3.1 pts |
| 20% | +0.4% | -5.9% | -6.3% | -11.8% | -11.7% | +0.0% | +2.2 pts | +3.1 pts |
| 25% | +1.0% | -3.8% | -4.9% | -8.1% | -8.1% | +0.0% | +1.9 pts | +2.8 pts |
| 30% | +2.8% | -1.4% | -4.3% | -5.0% | -5.0% | +0.2% | +1.3 pts | +1.9 pts |
| 35% | +19.2% | +15.1% | -5.0% | -2.4% | -2.1% | +0.7% | +0.6 pts | +0.7 pts |
| 40% | +12.6% | +7.0% | -6.1% | -1.1% | -0.6% | +1.8% | +0.9 pts | +1.2 pts |
| 45% | +25.0% | +20.6% | -5.8% | -0.9% | -0.4% | +3.0% | +1.1 pts | +1.5 pts |
| 50% | +16.3% | +12.7% | -4.2% | -0.7% | -0.3% | +3.8% | +1.1 pts | +1.5 pts |
| 55% | +7.8% | +5.6% | -2.2% | -0.5% | -0.2% | +4.8% | +1.0 pts | +1.2 pts |
| 60% | +14.6% | +13.8% | -0.8% | -0.3% | -0.1% | +6.0% | +0.8 pts | +1.1 pts |
| 65% | +21.5% | +21.5% | +0.0% | +0.0% | +0.1% | +6.7% | +0.8 pts | +1.1 pts |
| 70% | +11.7% | +11.9% | +0.2% | +0.2% | +0.2% | +5.4% | +0.6 pts | +0.8 pts |
| 75% | - | - | - | - | - | - | - | - |

Distinct modes per 64-token page (default menu, all layers, heads, chunks):

- saving 30%: mean 2.49; 1 modes 11.0%, 2 modes 37.1%, 3 modes 44.7%, 4 modes 6.6%, 5 modes 0.5%, 6 modes 0.1%, 7 modes 0.0%
- saving 50%: mean 2.88; 1 modes 1.9%, 2 modes 36.6%, 3 modes 38.0%, 4 modes 18.8%, 5 modes 4.3%, 6 modes 0.4%, 7 modes 0.0%, 8 modes 0.0%
- saving 70%: mean 2.13; 1 modes 29.7%, 2 modes 29.5%, 3 modes 38.8%, 4 modes 2.0%, 5 modes 0.0%, 6 modes 0.0%

## keys, distortion query

| saving | 41 modes vs best fixed | 8-mode default vs best fixed | 8-mode default vs 41 | top 6 vs 41 | top 8 vs 41 | auto masks (default menu) | g16 lost | g64 lost |
|---|---|---|---|---|---|---|---|---|
| 5% | -1.1% | -14.3% | -13.0% | -32.5% | -32.4% | +0.0% | +1.5 pts | +1.9 pts |
| 10% | -0.1% | -11.5% | -11.4% | -23.5% | -23.4% | +0.0% | +2.2 pts | +2.9 pts |
| 15% | +0.6% | -8.8% | -9.6% | -17.5% | -17.3% | +0.0% | +2.5 pts | +3.4 pts |
| 20% | +1.6% | -6.6% | -8.2% | -13.2% | -13.0% | +0.1% | +2.5 pts | +3.5 pts |
| 25% | +3.2% | -4.2% | -7.5% | -9.5% | -9.3% | +0.2% | +2.2 pts | +3.2 pts |
| 30% | +6.2% | -0.9% | -7.6% | -6.5% | -6.2% | +0.4% | +1.6 pts | +2.2 pts |
| 35% | +22.6% | +16.3% | -8.4% | -4.0% | -3.4% | +1.1% | +1.0 pts | +1.1 pts |
| 40% | +17.3% | +9.7% | -9.6% | -2.4% | -1.6% | +2.2% | +1.2 pts | +1.6 pts |
| 45% | +31.1% | +24.4% | -9.3% | -2.0% | -1.1% | +3.3% | +1.6 pts | +2.0 pts |
| 50% | +22.5% | +16.4% | -7.3% | -1.7% | -0.9% | +4.5% | +1.6 pts | +2.1 pts |
| 55% | +13.5% | +9.9% | -4.5% | -1.1% | -0.6% | +5.5% | +1.6 pts | +1.9 pts |
| 60% | +21.0% | +19.5% | -1.9% | -0.5% | -0.3% | +6.4% | +1.4 pts | +1.7 pts |
| 65% | +31.6% | +30.9% | -0.3% | -0.1% | -0.1% | +7.3% | +1.3 pts | +1.8 pts |
| 70% | +19.3% | +19.3% | -0.0% | -0.0% | -0.0% | +6.9% | +1.0 pts | +1.3 pts |
| 75% | - | - | - | - | - | - | - | - |

Distinct modes per 64-token page (default menu, all layers, heads, chunks):

- saving 30%: mean 2.87; 1 modes 5.4%, 2 modes 26.4%, 3 modes 48.3%, 4 modes 16.1%, 5 modes 3.4%, 6 modes 0.5%, 7 modes 0.0%, 8 modes 0.0%
- saving 50%: mean 3.65; 1 modes 1.6%, 2 modes 15.9%, 3 modes 27.7%, 4 modes 30.3%, 5 modes 19.3%, 6 modes 5.0%, 7 modes 0.2%, 8 modes 0.0%
- saving 70%: mean 2.24; 1 modes 23.1%, 2 modes 32.6%, 3 modes 42.2%, 4 modes 2.0%, 5 modes 0.2%, 6 modes 0.0%

## values, distortion cosine

| saving | 41 modes vs best fixed | 8-mode default vs best fixed | 8-mode default vs 41 | top 6 vs 41 | top 8 vs 41 | auto masks (default menu) | g16 lost | g64 lost |
|---|---|---|---|---|---|---|---|---|
| 5% | -1.2% | -11.8% | -10.6% | -25.5% | -25.5% | +0.0% | +1.2 pts | +1.6 pts |
| 10% | -0.5% | -8.9% | -8.5% | -17.7% | -17.6% | +0.0% | +1.6 pts | +2.2 pts |
| 15% | -0.1% | -6.1% | -6.1% | -12.0% | -11.9% | +0.0% | +1.7 pts | +2.2 pts |
| 20% | +0.1% | -4.2% | -4.5% | -8.3% | -8.2% | +0.0% | +1.6 pts | +2.1 pts |
| 25% | +0.5% | -2.3% | -3.1% | -5.2% | -5.0% | +0.0% | +1.2 pts | +1.7 pts |
| 30% | +1.1% | -0.9% | -2.1% | -2.7% | -2.4% | +0.1% | +0.7 pts | +1.1 pts |
| 35% | +18.5% | +16.4% | -2.7% | -0.7% | -0.5% | +0.4% | +0.3 pts | +0.4 pts |
| 40% | +11.6% | +6.1% | -5.8% | -0.7% | -0.0% | +1.5% | +0.7 pts | +0.9 pts |
| 45% | +24.8% | +20.2% | -6.1% | -0.8% | -0.0% | +2.6% | +0.8 pts | +1.0 pts |
| 50% | +14.6% | +11.0% | -4.1% | -0.5% | +0.0% | +3.4% | +0.7 pts | +0.9 pts |
| 55% | +4.3% | +2.8% | -1.5% | -0.3% | +0.1% | +4.1% | +0.5 pts | +0.7 pts |
| 60% | +11.7% | +11.5% | -0.1% | +0.0% | +0.1% | +6.0% | +0.6 pts | +0.8 pts |
| 65% | +20.7% | +20.8% | +0.2% | +0.1% | +0.1% | +7.8% | +0.6 pts | +0.8 pts |
| 70% | +10.4% | +10.5% | +0.2% | +0.2% | +0.2% | +6.2% | +0.4 pts | +0.5 pts |
| 75% | - | - | - | - | - | - | - | - |

Distinct modes per 64-token page (default menu, all layers, heads, chunks):

- saving 30%: mean 2.18; 1 modes 20.5%, 2 modes 43.5%, 3 modes 34.1%, 4 modes 1.8%, 5 modes 0.2%, 6 modes 0.0%, 7 modes 0.0%
- saving 50%: mean 2.50; 1 modes 7.5%, 2 modes 49.9%, 3 modes 30.4%, 4 modes 10.3%, 5 modes 1.9%, 6 modes 0.1%
- saving 70%: mean 2.00; 1 modes 27.4%, 2 modes 45.4%, 3 modes 27.1%, 4 modes 0.2%

## values, distortion squared

| saving | 41 modes vs best fixed | 8-mode default vs best fixed | 8-mode default vs 41 | top 6 vs 41 | top 8 vs 41 | auto masks (default menu) | g16 lost | g64 lost |
|---|---|---|---|---|---|---|---|---|
| 5% | -1.2% | -12.3% | -10.9% | -25.1% | -25.1% | +0.0% | +1.2 pts | +1.6 pts |
| 10% | -0.4% | -9.1% | -8.6% | -18.2% | -18.1% | +0.0% | +1.7 pts | +2.2 pts |
| 15% | +0.3% | -6.9% | -7.1% | -13.5% | -13.4% | +0.0% | +1.8 pts | +2.5 pts |
| 20% | +1.0% | -4.8% | -5.7% | -9.9% | -9.8% | +0.0% | +1.7 pts | +2.3 pts |
| 25% | +2.6% | -2.7% | -4.9% | -7.0% | -6.9% | +0.2% | +1.4 pts | +1.9 pts |
| 30% | +5.8% | +0.9% | -4.5% | -4.6% | -4.3% | +0.5% | +1.1 pts | +1.5 pts |
| 35% | +16.5% | +12.6% | -4.8% | -3.0% | -2.5% | +1.2% | +0.9 pts | +1.2 pts |
| 40% | +12.7% | +8.2% | -5.2% | -1.9% | -1.2% | +2.1% | +1.0 pts | +1.3 pts |
| 45% | +22.5% | +19.0% | -4.4% | -1.2% | -0.4% | +2.9% | +1.0 pts | +1.3 pts |
| 50% | +15.2% | +12.4% | -3.2% | -0.9% | -0.2% | +3.9% | +1.1 pts | +1.3 pts |
| 55% | +11.9% | +10.4% | -1.8% | -0.5% | -0.0% | +4.9% | +0.9 pts | +1.2 pts |
| 60% | +16.6% | +15.7% | -1.1% | -0.2% | +0.1% | +5.7% | +0.9 pts | +1.1 pts |
| 65% | +20.1% | +19.6% | -0.4% | -0.0% | +0.1% | +6.0% | +0.8 pts | +1.0 pts |
| 70% | +11.6% | +11.7% | +0.1% | +0.1% | +0.2% | +4.9% | +0.5 pts | +0.6 pts |
| 75% | - | - | - | - | - | - | - | - |

Distinct modes per 64-token page (default menu, all layers, heads, chunks):

- saving 30%: mean 2.48; 1 modes 10.8%, 2 modes 38.2%, 3 modes 43.8%, 4 modes 6.5%, 5 modes 0.6%, 6 modes 0.1%, 7 modes 0.0%
- saving 50%: mean 2.80; 1 modes 2.9%, 2 modes 38.3%, 3 modes 38.6%, 4 modes 16.5%, 5 modes 3.4%, 6 modes 0.3%, 7 modes 0.0%
- saving 70%: mean 1.94; 1 modes 40.0%, 2 modes 29.9%, 3 modes 26.6%, 4 modes 3.2%, 5 modes 0.3%, 6 modes 0.0%
