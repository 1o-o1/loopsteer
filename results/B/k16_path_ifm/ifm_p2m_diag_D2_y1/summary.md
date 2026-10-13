# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 5.83. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p2m_diag | True | True | True | 0.0 |

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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p2m_diag, D2

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | 0.03 [0.02, 0.04] | -0.00 | 0.03 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.02 [0.01, 0.03] | 0.00 | 0.03 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 8.61 [8.19, 9.05] | -14.97 | -6.36 | 0.47 [0.45, 0.49] | 0.02 | 0.02 |
| KO_SRC + O1 | KO_SRC | 0.02 [0.01, 0.03] | -0.01 | 0.02 | 0.00 [0.00, 0.00] | 0.02 | 0.02 |
| KO_SRC + PV | KO_SRC | 0.03 [0.03, 0.04] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.02 | 0.02 |
| KO_RND0 | clean | 0.85 [0.61, 1.11] | -1.49 | -0.64 | 0.04 [0.03, 0.06] | 0.00 | 0.92 |
| KO_RND0 + O1 | KO_RND0 | 0.03 [0.02, 0.05] | -0.01 | 0.03 | 0.00 [0.00, 0.00] | 0.00 | 0.92 |
| KO_RND0 + PV | KO_RND0 | 0.02 [0.01, 0.03] | 0.01 | 0.03 | 0.00 [0.00, 0.00] | 0.00 | 0.92 |
| KO_RND1 | clean | 0.73 [0.53, 0.93] | -1.42 | -0.69 | 0.04 [0.03, 0.05] | 0.01 | 0.92 |
| KO_RND1 + O1 | KO_RND1 | 0.03 [0.02, 0.05] | -0.01 | 0.02 | 0.00 [0.00, 0.00] | 0.01 | 0.92 |
| KO_RND1 + PV | KO_RND1 | 0.02 [0.01, 0.03] | 0.01 | 0.03 | 0.00 [0.00, 0.00] | 0.01 | 0.92 |
| KO_RND2 | clean | 0.79 [0.59, 1.01] | -1.44 | -0.64 | 0.04 [0.03, 0.05] | 0.00 | 0.93 |
| KO_RND2 + O1 | KO_RND2 | 0.03 [0.02, 0.04] | -0.01 | 0.02 | 0.00 [0.00, 0.00] | 0.00 | 0.93 |
| KO_RND2 + PV | KO_RND2 | 0.01 [0.00, 0.03] | 0.01 | 0.02 | 0.00 [0.00, 0.00] | 0.00 | 0.93 |
| KO_NON | clean | 9.30 [8.98, 9.65] | -20.78 | -11.48 | 0.51 [0.50, 0.52] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | 0.01 [0.00, 0.02] | -0.02 | -0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 9.38 [9.03, 9.73] | -24.32 | -14.94 | 0.51 [0.49, 0.52] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.03 [-0.04, -0.01] | 0.04 | 0.01 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.07 [-0.08, -0.05] | 0.04 | -0.03 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 0.53 [0.50, 0.57] | -0.00 | 0.53 | 0.03 [0.03, 0.03] | 0.00 | 1.00 |
| KO_SRC + SE | KO_SRC | 0.16 [0.12, 0.20] | -0.06 | 0.09 | 0.01 [0.00, 0.01] | 0.02 | 0.02 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 -0.00 [-0.00, 0.00]; SRC PV 0.00 [0.00, 0.00]; RND0 O1 0.00 [-0.00, 0.00]; RND0 PV -0.00 [-0.00, 0.00]; RND1 O1 0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 -0.00 [-0.00, 0.00]; RND2 PV -0.00 [-0.00, 0.00]; NON O1 -0.00 [-0.00, -0.00]; NON PV -0.00 [-0.00, 0.00]; ALL O1 -0.00 [-0.00, -0.00]; ALL PV -0.00 [-0.01, -0.00]

D_KO_SRC (B minus A under KO_SRC): 0.83 [0.67, 1.00]. O1 e normalised by D_KO: 0.01 [-0.00, 0.01].

By relation (e): {"op1": {"O1": 0.0017, "KO_SRC+O1|vsKO": 0.0018, "KO_SRC_top1A": 0.024, "SE+KO_SRC|vsKO": 0.0099}, "op2": {"O1": 0.0017, "KO_SRC+O1|vsKO": 0.001, "KO_SRC_top1A": 0.0102, "SE+KO_SRC|vsKO": 0.0013}}

Positive control: {"SE_e": [0.0304, 0.0284, 0.0325, 585, 552], "SE_dm": [0.5319, 0.4966, 0.5701, 585, 552], "SE_given_KO_SRC_e": [0.0056, 0.0039, 0.0075, 585, 552], "SE_given_KO_SRC_dm": [0.1567, 0.1174, 0.1986, 585, 552], "dm_ratio_KO_over_noKO": 0.2946, "structural_zero_capital": "0/0", "pass": true}
