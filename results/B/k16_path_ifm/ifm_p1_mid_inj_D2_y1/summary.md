# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 46.73. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | True | True | 0.0 |

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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_mid_inj, D2

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | -0.00 [-0.01, 0.01] | -0.01 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | -0.01 [-0.02, 0.00] | -0.00 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 7.96 [7.68, 8.26] | -7.98 | -0.02 | 0.42 [0.41, 0.44] | 0.02 | 0.04 |
| KO_SRC + O1 | KO_SRC | 0.00 [-0.01, 0.01] | 0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.02 | 0.04 |
| KO_SRC + PV | KO_SRC | -0.00 [-0.01, 0.00] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.02 | 0.03 |
| KO_RND0 | clean | 0.40 [0.28, 0.56] | -0.54 | -0.14 | 0.02 [0.01, 0.03] | 0.00 | 0.96 |
| KO_RND0 + O1 | KO_RND0 | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.96 |
| KO_RND0 + PV | KO_RND0 | -0.00 [-0.01, 0.01] | -0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.96 |
| KO_RND1 | clean | 0.38 [0.26, 0.50] | -0.50 | -0.12 | 0.02 [0.01, 0.03] | 0.00 | 0.96 |
| KO_RND1 + O1 | KO_RND1 | -0.01 [-0.02, 0.01] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.96 |
| KO_RND1 + PV | KO_RND1 | -0.00 [-0.01, 0.01] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.96 |
| KO_RND2 | clean | 0.48 [0.32, 0.64] | -0.60 | -0.12 | 0.03 [0.02, 0.03] | 0.00 | 0.95 |
| KO_RND2 + O1 | KO_RND2 | 0.00 [-0.01, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_RND2 + PV | KO_RND2 | 0.00 [-0.01, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_NON | clean | 9.22 [9.02, 9.43] | -13.21 | -4.00 | 0.49 [0.48, 0.49] | 0.01 | 0.01 |
| KO_NON + O1 | KO_NON | -0.02 [-0.04, -0.00] | 0.01 | -0.00 | -0.00 [-0.00, 0.00] | 0.01 | 0.01 |
| KO_NON + PV | KO_NON | 0.00 [-0.01, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.01 | 0.01 |
| KO_ALL | clean | 9.52 [9.31, 9.73] | -18.90 | -9.39 | 0.50 [0.49, 0.51] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.00 [-0.01, -0.00] | -0.00 | -0.00 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.00 [-0.01, -0.00] | 0.00 | -0.00 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | -0.01 [-0.02, 0.01] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC + SE | KO_SRC | 0.03 [0.01, 0.05] | -0.02 | 0.02 | 0.00 [0.00, 0.00] | 0.02 | 0.03 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [-0.00, 0.00]; SRC PV 0.00 [-0.00, 0.00]; RND0 O1 0.00 [-0.00, 0.00]; RND0 PV 0.00 [-0.00, 0.00]; RND1 O1 -0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 0.00 [-0.00, 0.00]; RND2 PV 0.00 [-0.00, 0.00]; NON O1 -0.00 [-0.00, 0.00]; NON PV 0.00 [-0.00, 0.00]; ALL O1 -0.00 [-0.00, 0.00]; ALL PV 0.00 [-0.00, 0.00]

D_KO_SRC (B minus A under KO_SRC): 2.88 [2.63, 3.13]. O1 e normalised by D_KO: 0.00 [-0.00, 0.01].

By relation (e): {"op1": {"O1": -0.0004, "KO_SRC+O1|vsKO": -0.0007, "KO_SRC_top1A": 0.0484, "SE+KO_SRC|vsKO": 0.0024}, "op2": {"O1": 0.0002, "KO_SRC+O1|vsKO": 0.0011, "KO_SRC_top1A": 0.024, "SE+KO_SRC|vsKO": 0.0005}}

Positive control: {"SE_e": [-0.0003, -0.0011, 0.0004, 581, 548], "SE_dm": [-0.0051, -0.0186, 0.0075, 581, 548], "SE_given_KO_SRC_e": [0.0015, 0.0006, 0.0024, 581, 548], "SE_given_KO_SRC_dm": [0.0322, 0.0145, 0.0497, 581, 548], "dm_ratio_KO_over_noKO": -6.3137, "structural_zero_capital": "0/0", "pass": true}
