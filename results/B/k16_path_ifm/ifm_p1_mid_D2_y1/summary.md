# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 46.85. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid | True | True | n/a | 0.0 |

## Items

```
{
 "ifm_p1_mid": {
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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_mid, D2

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.01 [0.01, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 6.77 [6.52, 7.04] | -7.67 | -0.89 | 0.42 [0.41, 0.44] | 0.02 | 0.04 |
| KO_SRC + O1 | KO_SRC | 0.03 [0.03, 0.04] | -0.01 | 0.02 | 0.00 [0.00, 0.00] | 0.02 | 0.04 |
| KO_SRC + PV | KO_SRC | 0.03 [0.02, 0.04] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.01 | 0.04 |
| KO_RND0 | clean | 0.31 [0.20, 0.45] | -0.45 | -0.14 | 0.02 [0.01, 0.03] | 0.00 | 0.96 |
| KO_RND0 + O1 | KO_RND0 | 0.01 [0.00, 0.02] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND0 + PV | KO_RND0 | 0.01 [0.00, 0.02] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND1 | clean | 0.28 [0.18, 0.40] | -0.44 | -0.16 | 0.02 [0.01, 0.03] | 0.00 | 0.95 |
| KO_RND1 + O1 | KO_RND1 | 0.01 [0.00, 0.02] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.95 |
| KO_RND1 + PV | KO_RND1 | 0.01 [0.00, 0.02] | 0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.95 |
| KO_RND2 | clean | 0.31 [0.21, 0.42] | -0.46 | -0.15 | 0.02 [0.01, 0.03] | 0.00 | 0.95 |
| KO_RND2 + O1 | KO_RND2 | 0.00 [-0.00, 0.01] | 0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.95 |
| KO_RND2 + PV | KO_RND2 | 0.01 [0.00, 0.02] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.95 |
| KO_NON | clean | 7.94 [7.71, 8.16] | -12.53 | -4.58 | 0.49 [0.48, 0.50] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | 0.01 [-0.01, 0.02] | 0.01 | 0.02 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | 0.01 [-0.01, 0.02] | 0.02 | 0.03 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 8.33 [8.10, 8.58] | -13.48 | -5.14 | 0.51 [0.50, 0.53] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.01 [-0.02, 0.01] | -0.00 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.00 [-0.02, 0.01] | -0.02 | -0.02 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 16.21 [15.84, 16.55] | -8.16 | 8.05 | 0.99 [0.99, 1.00] | 1.00 | 0.00 |
| KO_SRC + SE | KO_SRC | 2.26 [2.03, 2.50] | -1.11 | 1.15 | 0.12 [0.11, 0.14] | 0.04 | 0.03 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [0.00, 0.00]; SRC PV 0.00 [0.00, 0.00]; RND0 O1 0.00 [-0.00, 0.00]; RND0 PV -0.00 [-0.00, 0.00]; RND1 O1 0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 -0.00 [-0.00, 0.00]; RND2 PV -0.00 [-0.00, 0.00]; NON O1 -0.00 [-0.00, 0.00]; NON PV -0.00 [-0.00, 0.00]; ALL O1 -0.00 [-0.00, -0.00]; ALL PV -0.00 [-0.00, -0.00]

D_KO_SRC (B minus A under KO_SRC): 2.74 [2.47, 3.02]. O1 e normalised by D_KO: 0.01 [0.01, 0.02].

By relation (e): {"op1": {"O1": 0.0008, "KO_SRC+O1|vsKO": 0.0029, "KO_SRC_top1A": 0.0548, "SE+KO_SRC|vsKO": 0.1955}, "op2": {"O1": 0.0003, "KO_SRC+O1|vsKO": 0.001, "KO_SRC_top1A": 0.0273, "SE+KO_SRC|vsKO": 0.0522}}

Positive control: {"SE_e": [0.9942, 0.9918, 0.996, 585, 552], "SE_dm": [16.2064, 15.8363, 16.5512, 585, 552], "SE_given_KO_SRC_e": [0.1237, 0.1123, 0.136, 585, 552], "SE_given_KO_SRC_dm": [2.2565, 2.0318, 2.5003, 585, 552], "dm_ratio_KO_over_noKO": 0.1392, "structural_zero_capital": "0/0", "pass": true}
