# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 6.79. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p2m_ortho | True | True | True | 0.0 |

## Items

```
{
 "ifm_p2m_ortho": {
  "D2": {
   "n": 581,
   "clusters": 548,
   "by_rel": {
    "op1": 292,
    "op2": 289
   },
   "scanned": 585,
   "D_lt_2": 0
  }
 }
}
```

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p2m_ortho, D2

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | 0.04 [0.02, 0.05] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | 0.03 [0.02, 0.04] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 6.65 [6.33, 6.97] | -11.39 | -4.74 | 0.48 [0.46, 0.50] | 0.02 | 0.01 |
| KO_SRC + O1 | KO_SRC | 0.04 [0.03, 0.05] | -0.02 | 0.02 | 0.00 [0.00, 0.00] | 0.02 | 0.01 |
| KO_SRC + PV | KO_SRC | 0.03 [0.02, 0.04] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.02 | 0.01 |
| KO_RND0 | clean | 0.27 [0.15, 0.40] | -0.70 | -0.44 | 0.02 [0.01, 0.03] | 0.00 | 0.96 |
| KO_RND0 + O1 | KO_RND0 | 0.03 [0.02, 0.04] | -0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND0 + PV | KO_RND0 | 0.02 [0.01, 0.03] | -0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND1 | clean | 0.20 [0.10, 0.31] | -0.51 | -0.31 | 0.01 [0.01, 0.02] | 0.00 | 0.96 |
| KO_RND1 + O1 | KO_RND1 | 0.03 [0.02, 0.05] | -0.02 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND1 + PV | KO_RND1 | 0.02 [0.01, 0.03] | -0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.96 |
| KO_RND2 | clean | 0.36 [0.24, 0.47] | -0.92 | -0.56 | 0.03 [0.02, 0.04] | 0.00 | 0.93 |
| KO_RND2 + O1 | KO_RND2 | 0.03 [0.01, 0.04] | -0.01 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.93 |
| KO_RND2 + PV | KO_RND2 | 0.02 [0.01, 0.03] | -0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.93 |
| KO_NON | clean | 6.90 [6.64, 7.14] | -17.65 | -10.76 | 0.49 [0.48, 0.51] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | -0.01 [-0.02, 0.00] | -0.00 | -0.01 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | -0.03 [-0.04, -0.02] | 0.00 | -0.02 | -0.00 [-0.00, -0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 7.07 [6.76, 7.35] | -19.83 | -12.76 | 0.50 [0.49, 0.52] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | 0.02 [0.01, 0.02] | 0.00 | 0.02 | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | 0.01 [0.00, 0.02] | -0.01 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 1.23 [1.14, 1.32] | -0.08 | 1.15 | 0.09 [0.09, 0.10] | 0.00 | 0.99 |
| KO_SRC + SE | KO_SRC | 0.17 [0.13, 0.22] | -0.03 | 0.14 | 0.01 [0.01, 0.01] | 0.02 | 0.01 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [0.00, 0.00]; SRC PV 0.00 [-0.00, 0.00]; RND0 O1 -0.00 [-0.00, 0.00]; RND0 PV -0.00 [-0.00, 0.00]; RND1 O1 0.00 [-0.00, 0.00]; RND1 PV -0.00 [-0.00, 0.00]; RND2 O1 -0.00 [-0.00, 0.00]; RND2 PV -0.00 [-0.00, 0.00]; NON O1 -0.00 [-0.00, -0.00]; NON PV -0.00 [-0.01, -0.00]; ALL O1 -0.00 [-0.00, 0.00]; ALL PV -0.00 [-0.00, 0.00]

D_KO_SRC (B minus A under KO_SRC): 0.51 [0.43, 0.60]. O1 e normalised by D_KO: 0.04 [0.03, 0.05].

By relation (e): {"op1": {"O1": 0.0028, "KO_SRC+O1|vsKO": 0.0052, "KO_SRC_top1A": 0.0068, "SE+KO_SRC|vsKO": 0.015}, "op2": {"O1": 0.0016, "KO_SRC+O1|vsKO": 0.0013, "KO_SRC_top1A": 0.0208, "SE+KO_SRC|vsKO": 0.004}}

Positive control: {"SE_e": [0.0907, 0.0851, 0.097, 581, 548], "SE_dm": [1.2269, 1.1429, 1.3183, 581, 548], "SE_given_KO_SRC_e": [0.0095, 0.0069, 0.0122, 581, 548], "SE_given_KO_SRC_dm": [0.1711, 0.1296, 0.2151, 581, 548], "dm_ratio_KO_over_noKO": 0.1395, "structural_zero_capital": "0/0", "pass": true}
