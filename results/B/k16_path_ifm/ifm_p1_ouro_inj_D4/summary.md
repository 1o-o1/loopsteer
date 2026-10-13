# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 23.7. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_ouro_inj | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p1_ouro_inj": {
  "D4": {
   "n": 332,
   "clusters": 332,
   "by_rel": {
    "T1": 143,
    "T2": 189
   },
   "scanned": 355,
   "D_lt_2": 1
  }
 }
}
```

## Y2 input protocols: ifm_p1_ouro_inj, D4 (K=4), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=4 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.00 [0.00, 0.00] / 0.00 | 0.01 [0.00, 0.01] / 0.00 | 0.26 [0.25, 0.28] / 0.03 |
| (a) single arg(j) | 1 | 0.01 [0.01, 0.01] / 0.00 | 0.03 [0.02, 0.03] / 0.00 | 0.86 [0.85, 0.87] / 0.96 |
| (b) persistent arg(j..K) | 0.5 | 0.48 [0.46, 0.50] / 0.39 | 0.38 [0.37, 0.40] / 0.17 | 0.26 [0.25, 0.28] / 0.03 |
| (b) persistent arg(j..K) | 1 | 1.00 [1.00, 1.00] / 1.00 | 0.99 [0.99, 0.99] / 1.00 | 0.86 [0.85, 0.87] / 0.96 |
| (c) state post_inj(j) | 0.5 | 0.00 [0.00, 0.00] / 0.00 | 0.02 [0.01, 0.02] / 0.00 | 0.41 [0.40, 0.43] / 0.17 |
| (c) state post_inj(j) | 1 | 0.01 [0.01, 0.01] / 0.00 | 0.04 [0.03, 0.04] / 0.00 | 0.89 [0.88, 0.90] / 0.97 |
| (c) state pre_inj(j) | 0.5 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 |
| (c) state pre_inj(j) | 1 | 0.00 [-0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 | 0.03 [0.03, 0.04] / 0.00 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=4 |
|---|---|---|---|
| a | 0.13 (-0.01 / 0.12) | 0.32 (-0.03 / 0.29) | 10.18 (-4.53 / 5.65) |
| b | 11.75 (-5.82 / 5.93) | 11.66 (-5.74 / 5.92) | 10.18 (-4.53 / 5.65) |
| post | 0.13 (-0.01 / 0.12) | 0.46 (-0.04 / 0.42) | 10.55 (-4.85 / 5.70) |
| pre | 0.01 (-0.00 / 0.00) | 0.10 (-0.01 / 0.09) | 0.41 (-0.04 / 0.36) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 1.15; a j=2: 1, 1.15; a j=4: 1, 1.15; b j=1: 4, 4.59; b j=2: 3, 3.44; b j=4: 1, 1.15; post j=1: 1, 6.77; post j=2: 1, 12.91; post j=4: 1, 12.66; pre j=1: 1, 1.15; pre j=2: 1, 72.41; pre j=4: 1, 70.68

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [6.675, 9.9, 5.826, 2.585]; a j=2: [0.0, 6.676, 8.448, 4.258]; a j=4: [0.0, 0.0, 0.0, 6.679]; b j=1: [6.675, 12.901, 14.39, 12.657]; b j=2: [0.0, 6.676, 11.897, 11.951]; b j=4: [0.0, 0.0, 0.0, 6.679]; post j=1: [6.768, 9.911, 5.847, 2.593]; post j=2: [0.0, 12.912, 11.482, 6.53]; post j=4: [0.0, 0.0, 0.0, 12.662]; pre j=1: [0.15, 0.366, 0.418, 0.409]; pre j=2: [0.0, 9.911, 5.847, 2.593]; pre j=4: [0.0, 0.0, 0.0, 9.882]

(b) at j=K equals (a) at j=K bit-exactly: 664/664
