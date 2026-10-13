# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 6.25. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p2m_r5 | True | True | True | 0.0 |

## Items

```
{
 "ifm_p2m_r5": {
  "D2": {
   "n": 581,
   "clusters": 548,
   "by_rel": {
    "op1": 290,
    "op2": 291
   },
   "scanned": 585,
   "D_lt_2": 0
  }
 }
}
```

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p2m_r5, D2

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | -0.00 [-0.02, 0.01] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.01 [-0.01, 0.02] | 0.00 | 0.01 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 7.17 [6.83, 7.55] | -12.11 | -4.94 | 0.48 [0.46, 0.50] | 0.01 | 0.01 |
| KO_SRC + O1 | KO_SRC | 0.01 [0.00, 0.02] | -0.01 | 0.00 | 0.00 [-0.00, 0.00] | 0.01 | 0.01 |
| KO_SRC + PV | KO_SRC | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.01 | 0.01 |
| KO_RND0 | clean | 0.33 [0.19, 0.47] | -0.71 | -0.39 | 0.03 [0.02, 0.04] | 0.00 | 0.95 |
| KO_RND0 + O1 | KO_RND0 | 0.01 [-0.00, 0.02] | -0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_RND0 + PV | KO_RND0 | 0.01 [-0.00, 0.02] | 0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_RND1 | clean | 0.35 [0.22, 0.50] | -0.79 | -0.44 | 0.03 [0.02, 0.04] | 0.00 | 0.94 |
| KO_RND1 + O1 | KO_RND1 | 0.00 [-0.01, 0.02] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.94 |
| KO_RND1 + PV | KO_RND1 | 0.01 [0.00, 0.03] | -0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.94 |
| KO_RND2 | clean | 0.29 [0.17, 0.42] | -0.64 | -0.34 | 0.02 [0.01, 0.03] | 0.00 | 0.95 |
| KO_RND2 + O1 | KO_RND2 | 0.00 [-0.01, 0.02] | 0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_RND2 + PV | KO_RND2 | 0.01 [-0.00, 0.02] | 0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_NON | clean | 7.51 [7.18, 7.86] | -15.50 | -7.99 | 0.50 [0.49, 0.52] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | 0.00 [-0.01, 0.01] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | 0.00 [-0.01, 0.01] | -0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 7.56 [7.23, 7.90] | -17.55 | -10.00 | 0.50 [0.49, 0.52] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.06 [-0.07, -0.05] | 0.03 | -0.03 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.01 [-0.02, -0.00] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 0.20 [0.15, 0.25] | 0.00 | 0.20 | 0.01 [0.01, 0.02] | 0.00 | 1.00 |
| KO_SRC + SE | KO_SRC | 0.11 [0.07, 0.14] | -0.05 | 0.05 | 0.01 [0.00, 0.01] | 0.01 | 0.01 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [0.00, 0.00]; SRC PV 0.00 [-0.00, 0.00]; RND0 O1 0.00 [-0.00, 0.00]; RND0 PV 0.00 [-0.00, 0.00]; RND1 O1 0.00 [-0.00, 0.00]; RND1 PV 0.00 [0.00, 0.00]; RND2 O1 0.00 [-0.00, 0.00]; RND2 PV 0.00 [-0.00, 0.00]; NON O1 0.00 [-0.00, 0.00]; NON PV 0.00 [-0.00, 0.00]; ALL O1 -0.00 [-0.00, -0.00]; ALL PV 0.00 [-0.00, 0.00]

D_KO_SRC (B minus A under KO_SRC): 0.33 [0.26, 0.41]. O1 e normalised by D_KO: 0.03 [0.01, 0.04].

By relation (e): {"op1": {"O1": -0.0005, "KO_SRC+O1|vsKO": 0.0022, "KO_SRC_top1A": 0.0103, "SE+KO_SRC|vsKO": 0.0119}, "op2": {"O1": -0.0008, "KO_SRC+O1|vsKO": -0.001, "KO_SRC_top1A": 0.0137, "SE+KO_SRC|vsKO": 0.0008}}

Positive control: {"SE_e": [0.0148, 0.011, 0.0194, 581, 548], "SE_dm": [0.1988, 0.1505, 0.2504, 581, 548], "SE_given_KO_SRC_e": [0.0064, 0.0045, 0.0083, 581, 548], "SE_given_KO_SRC_dm": [0.1052, 0.0729, 0.1359, 581, 548], "dm_ratio_KO_over_noKO": 0.5292, "structural_zero_capital": "0/0", "pass": false}
