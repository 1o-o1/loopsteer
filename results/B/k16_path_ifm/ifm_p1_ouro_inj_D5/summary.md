# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 4.09. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_ouro_inj | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p1_ouro_inj": {
  "D5": {
   "n": 32,
   "clusters": 32,
   "by_rel": {
    "T2": 32
   },
   "scanned": 563,
   "D_lt_2": 2
  }
 }
}
```

## Y2 input protocols: ifm_p1_ouro_inj, D5 (K=4), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=4 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.07 [0.02, 0.12] / 0.06 | 0.04 [0.02, 0.05] / 0.03 | 0.21 [0.14, 0.28] / 0.12 |
| (a) single arg(j) | 1 | 0.15 [0.08, 0.21] / 0.12 | 0.13 [0.09, 0.17] / 0.16 | 0.52 [0.40, 0.65] / 0.28 |
| (b) persistent arg(j..K) | 0.5 | 0.57 [0.50, 0.65] / 0.38 | 0.39 [0.32, 0.47] / 0.25 | 0.21 [0.14, 0.28] / 0.12 |
| (b) persistent arg(j..K) | 1 | 1.00 [0.99, 1.01] / 0.94 | 0.85 [0.77, 0.92] / 0.59 | 0.52 [0.40, 0.65] / 0.28 |
| (c) state post_inj(j) | 0.5 | 0.06 [0.02, 0.11] / 0.06 | 0.13 [0.09, 0.18] / 0.12 | 0.33 [0.26, 0.41] / 0.25 |
| (c) state post_inj(j) | 1 | 0.15 [0.09, 0.22] / 0.09 | 0.29 [0.22, 0.37] / 0.19 | 0.58 [0.47, 0.69] / 0.38 |
| (c) state pre_inj(j) | 0.5 | 0.01 [-0.01, 0.02] / 0.00 | 0.05 [0.03, 0.07] / 0.06 | 0.04 [0.03, 0.06] / 0.06 |
| (c) state pre_inj(j) | 1 | 0.01 [-0.01, 0.02] / 0.00 | 0.13 [0.08, 0.19] / 0.06 | 0.11 [0.07, 0.14] / 0.16 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=4 |
|---|---|---|---|
| a | 0.60 (-0.29 / 0.32) | 0.58 (-0.08 / 0.50) | 2.20 (-0.86 / 1.34) |
| b | 4.25 (-2.22 / 2.04) | 3.74 (-1.81 / 1.93) | 2.20 (-0.86 / 1.34) |
| post | 0.63 (-0.28 / 0.35) | 1.24 (-0.46 / 0.78) | 2.50 (-1.07 / 1.44) |
| pre | 0.02 (0.00 / 0.02) | 0.54 (-0.23 / 0.31) | 0.45 (-0.14 / 0.31) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 1.14; a j=2: 1, 1.14; a j=4: 1, 1.14; b j=1: 4, 4.57; b j=2: 3, 3.43; b j=4: 1, 1.14; post j=1: 1, 6.71; post j=2: 1, 15.42; post j=4: 1, 14.80; pre j=1: 1, 1.14; pre j=2: 1, 93.71; pre j=4: 1, 89.02

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [6.616, 12.863, 8.531, 4.696]; a j=2: [0.0, 6.615, 9.335, 4.918]; a j=4: [0.0, 0.0, 0.0, 6.617]; b j=1: [6.616, 15.411, 16.726, 14.79]; b j=2: [0.0, 6.615, 12.404, 13.019]; b j=4: [0.0, 0.0, 0.0, 6.617]; post j=1: [6.708, 12.87, 8.526, 4.652]; post j=2: [0.0, 15.418, 14.443, 8.667]; post j=4: [0.0, 0.0, 0.0, 14.801]; pre j=1: [0.15, 0.43, 0.449, 0.5]; pre j=2: [0.0, 12.87, 8.526, 4.652]; pre j=4: [0.0, 0.0, 0.0, 12.478]

(b) at j=K equals (a) at j=K bit-exactly: 64/64
