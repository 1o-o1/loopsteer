# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 5.52. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid_inj | True | True | True | 0.0 |

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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_mid_inj, D5

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | -0.02 [-0.04, 0.01] | 0.00 | -0.02 | -0.00 [-0.01, 0.00] | 0.00 | 0.98 |
| PV | clean | -0.00 [-0.03, 0.02] | 0.00 | 0.00 | -0.00 [-0.01, 0.00] | 0.00 | 0.98 |
| KO_SRC | clean | 1.77 [1.29, 2.24] | -0.91 | 0.86 | 0.32 [0.24, 0.40] | 0.11 | 0.36 |
| KO_SRC + O1 | KO_SRC | 0.01 [-0.02, 0.03] | -0.01 | -0.01 | 0.00 [-0.00, 0.01] | 0.11 | 0.34 |
| KO_SRC + PV | KO_SRC | -0.00 [-0.03, 0.03] | -0.01 | -0.01 | -0.00 [-0.01, 0.01] | 0.11 | 0.36 |
| KO_RND0 | clean | 0.05 [-0.08, 0.24] | -0.17 | -0.13 | -0.00 [-0.02, 0.01] | 0.00 | 0.91 |
| KO_RND0 + O1 | KO_RND0 | -0.01 [-0.04, 0.01] | 0.01 | -0.01 | -0.00 [-0.01, 0.00] | 0.00 | 0.94 |
| KO_RND0 + PV | KO_RND0 | -0.01 [-0.04, 0.02] | 0.00 | -0.01 | 0.00 [-0.00, 0.01] | 0.00 | 0.94 |
| KO_RND1 | clean | 0.04 [-0.06, 0.18] | -0.05 | -0.02 | 0.02 [-0.01, 0.06] | 0.00 | 0.96 |
| KO_RND1 + O1 | KO_RND1 | -0.02 [-0.05, 0.01] | 0.03 | 0.01 | -0.01 [-0.01, 0.00] | 0.00 | 0.96 |
| KO_RND1 + PV | KO_RND1 | -0.00 [-0.03, 0.02] | 0.01 | 0.01 | -0.00 [-0.01, 0.00] | 0.00 | 0.96 |
| KO_RND2 | clean | -0.00 [-0.04, 0.03] | -0.06 | -0.06 | 0.00 [-0.01, 0.01] | 0.00 | 0.91 |
| KO_RND2 + O1 | KO_RND2 | -0.00 [-0.03, 0.02] | 0.01 | 0.01 | -0.00 [-0.01, 0.00] | 0.00 | 0.94 |
| KO_RND2 + PV | KO_RND2 | 0.03 [0.01, 0.05] | -0.02 | 0.02 | 0.01 [0.00, 0.01] | 0.00 | 0.89 |
| KO_NON | clean | 2.82 [2.18, 3.48] | -11.32 | -8.49 | 0.46 [0.38, 0.54] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | -0.01 [-0.04, 0.03] | -0.04 | -0.05 | -0.00 [-0.01, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | -0.04 [-0.11, 0.02] | -0.00 | -0.04 | -0.01 [-0.03, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 2.65 [1.98, 3.30] | -14.51 | -11.86 | 0.42 [0.35, 0.49] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.01 [-0.01, 0.00] | 0.00 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.00 [-0.01, 0.01] | 0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | -0.00 [-0.04, 0.03] | 0.02 | 0.02 | -0.00 [-0.01, 0.01] | 0.00 | 0.98 |
| KO_SRC + SE | KO_SRC | 0.03 [-0.01, 0.07] | -0.01 | 0.02 | 0.00 [-0.00, 0.01] | 0.13 | 0.36 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [-0.00, 0.01]; SRC PV 0.00 [-0.01, 0.01]; RND0 O1 0.00 [-0.01, 0.01]; RND0 PV 0.00 [-0.00, 0.01]; RND1 O1 -0.00 [-0.01, 0.00]; RND1 PV -0.00 [-0.01, 0.00]; RND2 O1 0.00 [-0.01, 0.01]; RND2 PV 0.01 [0.00, 0.01]; NON O1 -0.00 [-0.01, 0.01]; NON PV -0.01 [-0.03, 0.01]; ALL O1 0.00 [-0.00, 0.01]; ALL PV 0.00 [-0.00, 0.01]

D_KO_SRC (B minus A under KO_SRC): 2.22 [1.56, 2.91]. O1 e normalised by D_KO: 0.01 [-0.01, 0.02].

By relation (e): {"T2": {"O1": -0.0026, "KO_SRC+O1|vsKO": 0.0016, "KO_SRC_top1A": 0.3617, "SE+KO_SRC|vsKO": 0.0043}}

Positive control: {"SE_e": [-0.0007, -0.0092, 0.0073, 41, 41], "SE_dm": [-0.004, -0.0386, 0.0332, 47, 47], "SE_given_KO_SRC_e": [0.0043, -0.0036, 0.0118, 41, 41], "SE_given_KO_SRC_dm": [0.0279, -0.012, 0.0665, 47, 47], "dm_ratio_KO_over_noKO": -6.975, "structural_zero_capital": "0/0", "pass": true}
