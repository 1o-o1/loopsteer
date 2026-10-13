# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 24.65. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p1_mid_inj": {
  "D4": {
   "n": 340,
   "clusters": 340,
   "by_rel": {
    "T1": 143,
    "T2": 197
   },
   "scanned": 350,
   "D_lt_2": 0
  }
 }
}
```

## Y2 input protocols: ifm_p1_mid_inj, D4 (K=8), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=8 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.30 [0.29, 0.32] / 0.02 |
| (a) single arg(j) | 1 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.82 [0.80, 0.83] / 0.89 |
| (b) persistent arg(j..K) | 0.5 | 0.49 [0.48, 0.50] / 0.41 | 0.48 [0.47, 0.49] / 0.39 | 0.30 [0.29, 0.32] / 0.02 |
| (b) persistent arg(j..K) | 1 | 0.99 [0.99, 1.00] / 1.00 | 0.99 [0.99, 1.00] / 0.99 | 0.82 [0.80, 0.83] / 0.89 |
| (c) state post_inj(j) | 0.5 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.39 [0.38, 0.41] / 0.16 |
| (c) state post_inj(j) | 1 | 0.00 [0.00, 0.00] / 0.00 | 0.00 [0.00, 0.01] / 0.00 | 0.86 [0.84, 0.87] / 0.91 |
| (c) state pre_inj(j) | 0.5 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.01 [0.01, 0.02] / 0.00 |
| (c) state pre_inj(j) | 1 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.04 [0.04, 0.05] / 0.00 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=8 |
|---|---|---|---|
| a | 0.01 (-0.01 / 0.00) | 0.03 (-0.00 / 0.02) | 9.11 (-3.89 / 5.22) |
| b | 10.94 (-5.34 / 5.60) | 10.93 (-5.33 / 5.60) | 9.11 (-3.89 / 5.22) |
| post | 0.01 (-0.00 / 0.01) | 0.04 (-0.00 / 0.03) | 9.53 (-4.29 / 5.24) |
| pre | 0.01 (-0.01 / -0.00) | 0.00 (-0.00 / 0.00) | 0.46 (-0.02 / 0.44) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 25.57; a j=2: 1, 25.57; a j=8: 1, 25.57; b j=1: 8, 204.56; b j=2: 7, 178.99; b j=8: 1, 25.57; post j=1: 1, 26.92; post j=2: 1, 32.43; post j=8: 1, 30.13; pre j=1: 1, 25.57; pre j=2: 1, 63.00; pre j=8: 1, 50.90

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [22.012, 16.247, 11.468, 6.775, 4.172, 2.737, 1.705, 0.971]; a j=2: [0.0, 22.014, 19.459, 12.278, 7.426, 4.659, 2.664, 1.399]; a j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.014]; b j=1: [22.012, 30.843, 37.722, 38.615, 40.93, 41.569, 37.367, 30.119]; b j=2: [0.0, 22.014, 33.536, 37.348, 40.271, 41.237, 37.178, 30.041]; b j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.014]; post j=1: [26.917, 18.336, 13.172, 7.756, 4.75, 3.084, 1.885, 1.069]; post j=2: [0.0, 32.427, 25.235, 17.105, 10.514, 6.619, 3.825, 1.967]; post j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.13]; pre j=1: [7.506, 4.875, 2.995, 1.876, 1.343, 1.063, 0.857, 0.627]; pre j=2: [0.0, 18.336, 13.172, 7.756, 4.75, 3.084, 1.885, 1.069]; pre j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.196]

(b) at j=K equals (a) at j=K bit-exactly: 680/680
