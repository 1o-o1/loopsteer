# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 27.32. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_ouro_inj | True | True | True | 0.0 |

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

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_ouro_inj, D4

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 5.33 [5.12, 5.53] | -4.51 | 0.82 | 0.45 [0.44, 0.46] | 0.03 | 0.06 |
| KO_SRC + O1 | KO_SRC | 0.01 [-0.00, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.03 | 0.06 |
| KO_SRC + PV | KO_SRC | 0.00 [-0.01, 0.01] | -0.01 | -0.00 | 0.00 [-0.00, 0.00] | 0.03 | 0.06 |
| KO_RND0 | clean | 0.08 [0.03, 0.13] | -0.10 | -0.02 | 0.01 [0.00, 0.01] | 0.00 | 0.99 |
| KO_RND0 + O1 | KO_RND0 | -0.00 [-0.01, 0.01] | -0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND0 + PV | KO_RND0 | 0.01 [-0.00, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND1 | clean | 0.07 [0.01, 0.14] | -0.08 | -0.01 | 0.01 [0.00, 0.01] | 0.00 | 0.98 |
| KO_RND1 + O1 | KO_RND1 | -0.00 [-0.01, 0.01] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND1 + PV | KO_RND1 | 0.00 [-0.01, 0.01] | 0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND2 | clean | 0.08 [0.01, 0.18] | -0.06 | 0.02 | 0.01 [0.00, 0.02] | 0.01 | 0.99 |
| KO_RND2 + O1 | KO_RND2 | 0.01 [-0.00, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.01 | 0.99 |
| KO_RND2 + PV | KO_RND2 | 0.00 [-0.01, 0.01] | 0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.01 | 0.99 |
| KO_NON | clean | 5.86 [5.72, 6.00] | -13.52 | -7.66 | 0.50 [0.49, 0.51] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | 0.00 [-0.01, 0.01] | -0.02 | -0.02 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | -0.02 [-0.07, 0.02] | 0.24 | 0.22 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 5.95 [5.80, 6.08] | -14.57 | -8.62 | 0.51 [0.50, 0.51] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.00 [-0.00, 0.00] | -0.01 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.00 [-0.01, -0.00] | -0.00 | -0.00 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 0.10 [0.08, 0.12] | -0.01 | 0.09 | 0.01 [0.01, 0.01] | 0.00 | 1.00 |
| KO_SRC + SE | KO_SRC | 0.08 [0.07, 0.10] | -0.02 | 0.06 | 0.01 [0.01, 0.01] | 0.03 | 0.06 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [-0.00, 0.00]; SRC PV -0.00 [-0.00, 0.00]; RND0 O1 -0.00 [-0.00, 0.00]; RND0 PV -0.00 [-0.00, 0.00]; RND1 O1 -0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 0.00 [-0.00, 0.00]; RND2 PV -0.00 [-0.00, 0.00]; NON O1 -0.00 [-0.00, 0.00]; NON PV -0.00 [-0.01, 0.00]; ALL O1 -0.00 [-0.00, 0.00]; ALL PV -0.00 [-0.00, -0.00]

D_KO_SRC (B minus A under KO_SRC): 1.10 [1.01, 1.20]. O1 e normalised by D_KO: -0.00 [-0.01, 0.01].

By relation (e): {"T1": {"O1": -0.0002, "KO_SRC+O1|vsKO": 0.0003, "KO_SRC_top1A": 0.0839, "SE+KO_SRC|vsKO": 0.0067}, "T2": {"O1": 0.0007, "KO_SRC+O1|vsKO": 0.0005, "KO_SRC_top1A": 0.0476, "SE+KO_SRC|vsKO": 0.0076}}

Positive control: {"SE_e": [0.0086, 0.0065, 0.0106, 331, 331], "SE_dm": [0.0973, 0.0776, 0.1154, 332, 332], "SE_given_KO_SRC_e": [0.0072, 0.0056, 0.0088, 331, 331], "SE_given_KO_SRC_dm": [0.083, 0.0651, 0.1022, 332, 332], "dm_ratio_KO_over_noKO": 0.853, "structural_zero_capital": "0/0", "pass": false}
