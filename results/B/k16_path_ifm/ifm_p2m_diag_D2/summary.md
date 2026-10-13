# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 5.53. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p2m_diag | True | n/a | True | 0.0 |

## Items

```
{
 "ifm_p2m_diag": {
  "D2": {
   "n": 585,
   "clusters": 552,
   "by_rel": {
    "op1": 292,
    "op2": 293
   },
   "scanned": 585,
   "D_lt_2": 0
  }
 }
}
```

## Y2 input protocols: ifm_p2m_diag, D2 (K=5), e [CI] / flip; vs clean A

| protocol | a | j=1 | j=2 | j=5 |
|---|---|---|---|---|
| (a) single arg(j) | 0.5 | 0.01 [0.01, 0.01] / 0.00 | 0.02 [0.02, 0.02] / 0.00 | 0.17 [0.16, 0.17] / 0.01 |
| (a) single arg(j) | 1 | 0.03 [0.03, 0.04] / 0.00 | 0.05 [0.05, 0.05] / 0.00 | 0.43 [0.42, 0.44] / 0.23 |
| (b) persistent arg(j..K) | 0.5 | 0.51 [0.50, 0.51] / 0.45 | 0.43 [0.42, 0.44] / 0.25 | 0.17 [0.16, 0.17] / 0.01 |
| (b) persistent arg(j..K) | 1 | 1.00 [1.00, 1.00] / 1.00 | 0.97 [0.96, 0.97] / 1.00 | 0.43 [0.42, 0.44] / 0.23 |
| (c) state post_inj(j) | 0.5 | 0.01 [0.01, 0.01] / 0.00 | 0.04 [0.04, 0.04] / 0.00 | 0.47 [0.46, 0.48] / 0.35 |
| (c) state post_inj(j) | 1 | 0.03 [0.03, 0.04] / 0.00 | 0.09 [0.09, 0.10] / 0.00 | 0.95 [0.94, 0.96] / 1.00 |
| (c) state pre_inj(j) | 0.5 | 0.00 [0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 | 0.19 [0.18, 0.19] / 0.01 |
| (c) state pre_inj(j) | 1 | 0.00 [0.00, 0.00] / 0.00 | 0.03 [0.03, 0.03] / 0.00 | 0.51 [0.50, 0.52] / 0.46 |

dm (logits) and dz(A) / dz(B) at a=1:

| protocol | j=1 | j=2 | j=5 |
|---|---|---|---|
| a | 0.60 (-0.00 / 0.59) | 0.86 (-0.01 / 0.85) | 7.93 (-2.11 / 5.82) |
| b | 18.59 (-9.18 / 9.41) | 17.96 (-8.58 / 9.38) | 7.93 (-2.11 / 5.82) |
| post | 0.60 (-0.00 / 0.59) | 1.67 (-0.13 / 1.54) | 17.53 (-8.37 / 9.17) |
| pre | 0.00 (0.00 / 0.00) | 0.53 (-0.00 / 0.53) | 9.33 (-2.85 / 6.49) |

Strength at a=1 (edited calls, summed delta norm): a j=1: 1, 93.64; a j=2: 1, 93.64; a j=5: 1, 93.64; b j=1: 5, 468.21; b j=2: 4, 374.57; b j=5: 1, 93.64; post j=1: 1, 139.15; post j=2: 1, 240.98; post j=5: 1, 287.77; pre j=1: 1, 0.00; pre j=2: 1, 378.39; pre j=5: 1, 506.24

Entry delta |post_edit(i) - post_clean(i)| at the source, by call i, a=1: a j=1: [139.155, 135.44, 98.63, 61.331, 35.15]; a j=2: [0.0, 139.179, 93.502, 64.734, 39.443]; a j=5: [0.0, 0.0, 0.0, 0.0, 139.214]; b j=1: [139.155, 240.981, 271.112, 282.78, 287.766]; b j=2: [0.0, 139.179, 213.345, 256.668, 276.24]; b j=5: [0.0, 0.0, 0.0, 0.0, 139.214]; post j=1: [139.155, 135.44, 98.63, 61.331, 35.15]; post j=2: [0.0, 240.981, 165.52, 116.097, 74.9]; post j=5: [0.0, 0.0, 0.0, 0.0, 287.766]; pre j=1: [0.0, 0.0, 0.0, 0.0, 0.0]; pre j=2: [0.0, 135.44, 98.63, 61.331, 35.15]; pre j=5: [0.0, 0.0, 0.0, 0.0, 182.78]

(b) at j=K equals (a) at j=K bit-exactly: 1170/1170
