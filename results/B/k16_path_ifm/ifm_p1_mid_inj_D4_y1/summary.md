# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 27.86. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | True | True | 0.0 |

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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_mid_inj, D4

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | 0.01 [0.00, 0.02] | -0.01 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.01 [0.00, 0.02] | -0.01 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 4.77 [4.56, 4.98] | -3.87 | 0.90 | 0.43 [0.42, 0.45] | 0.02 | 0.10 |
| KO_SRC + O1 | KO_SRC | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.02 | 0.10 |
| KO_SRC + PV | KO_SRC | 0.00 [-0.00, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.02 | 0.10 |
| KO_RND0 | clean | 0.06 [0.00, 0.13] | -0.10 | -0.04 | 0.01 [-0.00, 0.01] | 0.00 | 0.99 |
| KO_RND0 + O1 | KO_RND0 | 0.00 [-0.00, 0.01] | 0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND0 + PV | KO_RND0 | 0.00 [-0.00, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND1 | clean | 0.09 [0.03, 0.17] | -0.12 | -0.03 | 0.01 [0.00, 0.02] | 0.00 | 0.98 |
| KO_RND1 + O1 | KO_RND1 | 0.00 [-0.00, 0.01] | -0.00 | -0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND1 + PV | KO_RND1 | -0.00 [-0.01, 0.01] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND2 | clean | 0.06 [0.01, 0.12] | -0.05 | 0.01 | 0.01 [0.00, 0.01] | 0.00 | 0.99 |
| KO_RND2 + O1 | KO_RND2 | -0.00 [-0.01, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND2 + PV | KO_RND2 | -0.00 [-0.01, 0.00] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_NON | clean | 4.77 [4.58, 4.97] | -5.87 | -1.09 | 0.45 [0.43, 0.47] | 0.04 | 0.23 |
| KO_NON + O1 | KO_NON | 0.02 [-0.03, 0.08] | -0.08 | -0.05 | 0.00 [-0.00, 0.01] | 0.05 | 0.20 |
| KO_NON + PV | KO_NON | 0.06 [-0.02, 0.14] | -0.08 | -0.02 | 0.00 [-0.00, 0.01] | 0.05 | 0.21 |
| KO_ALL | clean | 5.59 [5.45, 5.74] | -12.36 | -6.76 | 0.51 [0.50, 0.52] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | 0.00 [0.00, 0.01] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | 0.01 [0.00, 0.01] | 0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC + SE | KO_SRC | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.02 | 0.10 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 -0.00 [-0.00, 0.00]; SRC PV -0.00 [-0.00, 0.00]; RND0 O1 -0.00 [-0.00, 0.00]; RND0 PV -0.00 [-0.00, 0.00]; RND1 O1 -0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 -0.00 [-0.00, 0.00]; RND2 PV -0.00 [-0.00, 0.00]; NON O1 0.00 [-0.01, 0.01]; NON PV 0.00 [-0.00, 0.01]; ALL O1 -0.00 [-0.00, 0.00]; ALL PV -0.00 [-0.00, 0.00]

D_KO_SRC (B minus A under KO_SRC): 1.46 [1.33, 1.59]. O1 e normalised by D_KO: 0.01 [0.00, 0.01].

By relation (e): {"T1": {"O1": 0.0006, "KO_SRC+O1|vsKO": -0.0, "KO_SRC_top1A": 0.1119, "SE+KO_SRC|vsKO": 0.0008}, "T2": {"O1": 0.0008, "KO_SRC+O1|vsKO": 0.0004, "KO_SRC_top1A": 0.0914, "SE+KO_SRC|vsKO": 0.0014}}

Positive control: {"SE_e": [0.0004, -0.001, 0.0021, 340, 340], "SE_dm": [0.0018, -0.0083, 0.0125, 340, 340], "SE_given_KO_SRC_e": [0.0011, 0.0001, 0.0023, 340, 340], "SE_given_KO_SRC_dm": [0.0123, 0.0015, 0.0241, 340, 340], "dm_ratio_KO_over_noKO": 6.8333, "structural_zero_capital": "0/0", "pass": false}
