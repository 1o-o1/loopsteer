# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 5.08. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p1_mid_inj": {
  "D5": {
   "n": 47,
   "clusters": 47,
   "by_rel": {
    "T2": 47
   },
   "scanned": 531,
   "D_lt_2": 6
  }
 }
}
```

## Y2 input protocols: ifm_p1_mid_inj, D5 (K=8), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=8 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.00 [-0.00, 0.01] / 0.00 | 0.01 [-0.00, 0.01] / 0.00 | 0.14 [0.08, 0.19] / 0.02 |
| (a) single arg(j) | 1 | 0.01 [0.00, 0.02] / 0.00 | 0.01 [-0.00, 0.03] / 0.00 | 0.43 [0.35, 0.50] / 0.36 |
| (b) persistent arg(j..K) | 0.5 | 0.42 [0.34, 0.50] / 0.30 | 0.41 [0.32, 0.48] / 0.28 | 0.14 [0.08, 0.19] / 0.02 |
| (b) persistent arg(j..K) | 1 | 0.95 [0.93, 0.97] / 0.89 | 0.94 [0.92, 0.96] / 0.85 | 0.43 [0.35, 0.50] / 0.36 |
| (c) state post_inj(j) | 0.5 | 0.00 [-0.00, 0.01] / 0.00 | 0.01 [-0.00, 0.01] / 0.00 | 0.18 [0.12, 0.23] / 0.04 |
| (c) state post_inj(j) | 1 | 0.01 [-0.00, 0.02] / 0.00 | 0.02 [0.01, 0.03] / 0.00 | 0.46 [0.38, 0.53] / 0.36 |
| (c) state pre_inj(j) | 0.5 | -0.00 [-0.01, 0.01] / 0.00 | -0.00 [-0.01, 0.00] / 0.00 | 0.01 [-0.00, 0.01] / 0.02 |
| (c) state pre_inj(j) | 1 | -0.00 [-0.01, 0.01] / 0.00 | -0.00 [-0.01, 0.01] / 0.00 | 0.02 [0.00, 0.03] / 0.02 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=8 |
|---|---|---|---|
| a | 0.04 (-0.03 / 0.01) | 0.08 (0.00 / 0.08) | 2.39 (-0.83 / 1.57) |
| b | 5.71 (-2.85 / 2.86) | 5.66 (-2.84 / 2.83) | 2.39 (-0.83 / 1.57) |
| post | 0.04 (-0.02 / 0.01) | 0.10 (-0.01 / 0.09) | 2.47 (-0.88 / 1.59) |
| pre | 0.00 (0.00 / 0.00) | -0.00 (0.02 / 0.02) | 0.09 (0.02 / 0.12) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 29.93; a j=2: 1, 29.93; a j=8: 1, 29.93; b j=1: 8, 239.44; b j=2: 7, 209.51; b j=8: 1, 29.93; post j=1: 1, 31.02; post j=2: 1, 37.98; post j=8: 1, 36.11; pre j=1: 1, 29.93; pre j=2: 1, 78.16; pre j=8: 1, 67.81

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [25.32, 20.222, 13.894, 7.721, 4.425, 2.804, 1.696, 0.965]; a j=2: [0.0, 25.32, 27.559, 17.83, 9.828, 5.715, 3.125, 1.497]; a j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.321]; b j=1: [25.32, 36.008, 46.235, 48.122, 48.282, 49.922, 45.764, 36.106]; b j=2: [0.0, 25.32, 41.542, 47.139, 47.823, 49.732, 45.689, 36.095]; b j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.321]; post j=1: [31.017, 22.736, 15.639, 8.747, 4.973, 3.116, 1.812, 1.02]; post j=2: [0.0, 37.976, 33.722, 21.939, 12.295, 7.038, 3.748, 1.781]; post j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.112]; pre j=1: [8.732, 5.916, 3.47, 2.132, 1.438, 1.116, 0.865, 0.708]; pre j=2: [0.0, 22.736, 15.639, 8.747, 4.973, 3.116, 1.812, 1.02]; pre j=8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.167]

(b) at j=K equals (a) at j=K bit-exactly: 94/94
