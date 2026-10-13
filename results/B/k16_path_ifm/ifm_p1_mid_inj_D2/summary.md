# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 43.28. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p1_mid_inj": {
  "D2": {
   "n": 581,
   "clusters": 548,
   "by_rel": {
    "op1": 289,
    "op2": 292
   },
   "scanned": 585,
   "D_lt_2": 0
  }
 }
}
```

## Y2 input protocols: ifm_p1_mid_inj, D2 (K=8), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=8 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.26 [0.25, 0.27] / 0.06 |
| (a) single arg(j) | 1 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.75 [0.73, 0.76] / 0.79 |
| (b) persistent arg(j..K) | 0.5 | 0.46 [0.45, 0.47] / 0.26 | 0.45 [0.44, 0.46] / 0.25 | 0.26 [0.25, 0.27] / 0.06 |
| (b) persistent arg(j..K) | 1 | 0.98 [0.97, 0.98] / 1.00 | 0.98 [0.97, 0.98] / 1.00 | 0.75 [0.73, 0.76] / 0.79 |
| (c) state post_inj(j) | 0.5 | 0.00 [0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.30 [0.29, 0.31] / 0.09 |
| (c) state post_inj(j) | 1 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.78 [0.77, 0.79] / 0.83 |
| (c) state pre_inj(j) | 0.5 | -0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 |
| (c) state pre_inj(j) | 1 | -0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.03 [0.02, 0.03] / 0.00 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=8 |
|---|---|---|---|
| a | 0.01 (0.00 / 0.01) | -0.00 (0.02 / 0.01) | 14.22 (-6.28 / 7.94) |
| b | 18.60 (-9.19 / 9.42) | 18.58 (-9.16 / 9.42) | 14.22 (-6.28 / 7.94) |
| post | 0.00 (0.01 / 0.02) | -0.01 (0.02 / 0.01) | 14.82 (-6.64 / 8.18) |
| pre | -0.00 (-0.00 / -0.00) | -0.01 (0.00 / -0.00) | 0.48 (-0.09 / 0.39) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 35.78; a j=2: 1, 35.78; a j=8: 1, 35.78; b j=1: 8, 286.22; b j=2: 7, 250.44; b j=8: 1, 35.78; post j=1: 1, 36.01; post j=2: 1, 42.51; post j=8: 1, 37.44; pre j=1: 1, 35.78; pre j=2: 1, 87.23; pre j=8: 1, 57.87

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [29.211, 22.09, 14.669, 10.577, 7.163, 4.765, 2.794, 1.412]; a j=2: [0.0, 29.212, 22.453, 15.981, 10.095, 6.537, 3.733, 1.82]; a j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.211]; b j=1: [29.211, 40.106, 44.644, 48.409, 49.661, 50.304, 45.015, 37.426]; b j=2: [0.0, 29.212, 40.894, 47.07, 48.899, 50.113, 44.872, 37.369]; b j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.211]; post j=1: [36.014, 25.331, 16.469, 11.568, 7.81, 5.184, 3.008, 1.508]; post j=2: [0.0, 42.511, 28.665, 19.914, 12.716, 8.237, 4.756, 2.335]; post j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 37.435]; pre j=1: [10.474, 7.442, 4.636, 3.312, 2.361, 1.729, 1.204, 0.769]; pre j=2: [0.0, 25.331, 16.469, 11.568, 7.81, 5.184, 3.008, 1.508]; pre j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.16]

(b) at j=K equals (a) at j=K bit-exactly: 1162/1162
