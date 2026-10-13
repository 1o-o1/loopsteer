# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 26.3. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p1_mid | True | True | n/a | 0.0 |

## Items

```
{
 "ifm_p1_mid": {
  "D4": {
   "n": 333,
   "clusters": 333,
   "by_rel": {
    "T1": 143,
    "T2": 190
   },
   "scanned": 352,
   "D_lt_2": 0
  }
 }
}
```

## Y1 answer edit at the loop-1 end x knockout of last->keys in all later calls: ifm_p1_mid, D4

| row | baseline | dm [CI] | dz(A) | dz(B) | e [CI] | flip | top-1 A |
|---|---|---|---|---|---|---|---|
| O1 | clean | -0.00 [-0.01, 0.01] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| PV | clean | -0.00 [-0.01, 0.00] | 0.00 | 0.00 | -0.00 [-0.00, 0.00] | 0.00 | 1.00 |
| KO_SRC | clean | 4.74 [4.55, 4.93] | -3.93 | 0.80 | 0.45 [0.43, 0.46] | 0.02 | 0.07 |
| KO_SRC + O1 | KO_SRC | 0.00 [-0.00, 0.01] | 0.00 | 0.01 | 0.00 [-0.00, 0.00] | 0.02 | 0.06 |
| KO_SRC + PV | KO_SRC | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.02 | 0.07 |
| KO_RND0 | clean | 0.04 [0.00, 0.09] | -0.08 | -0.03 | 0.00 [-0.00, 0.01] | 0.00 | 0.99 |
| KO_RND0 + O1 | KO_RND0 | 0.00 [-0.00, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_RND0 + PV | KO_RND0 | 0.01 [-0.00, 0.02] | -0.01 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.98 |
| KO_RND1 | clean | 0.06 [0.01, 0.12] | -0.09 | -0.03 | 0.01 [0.00, 0.01] | 0.00 | 0.98 |
| KO_RND1 + O1 | KO_RND1 | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND1 + PV | KO_RND1 | -0.00 [-0.01, 0.01] | -0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND2 | clean | 0.03 [-0.00, 0.08] | -0.05 | -0.01 | 0.00 [-0.00, 0.01] | 0.00 | 0.99 |
| KO_RND2 + O1 | KO_RND2 | -0.00 [-0.01, 0.01] | -0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.98 |
| KO_RND2 + PV | KO_RND2 | 0.00 [-0.01, 0.01] | -0.00 | 0.00 | 0.00 [-0.00, 0.00] | 0.00 | 0.99 |
| KO_NON | clean | 5.25 [5.04, 5.46] | -7.40 | -2.15 | 0.48 [0.47, 0.50] | 0.00 | 0.00 |
| KO_NON + O1 | KO_NON | 0.01 [0.00, 0.02] | -0.00 | 0.01 | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| KO_NON + PV | KO_NON | 0.02 [0.01, 0.03] | -0.01 | 0.01 | 0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL | clean | 5.33 [5.12, 5.54] | -7.24 | -1.91 | 0.50 [0.48, 0.51] | 0.00 | 0.00 |
| KO_ALL + O1 | KO_ALL | -0.01 [-0.01, 0.00] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| KO_ALL + PV | KO_ALL | -0.01 [-0.02, 0.00] | 0.00 | -0.00 | -0.00 [-0.00, 0.00] | 0.00 | 0.00 |
| SE (source O1 a=1) | clean | 10.78 [10.47, 11.07] | -5.40 | 5.39 | 1.00 [1.00, 1.00] | 1.00 | 0.00 |
| KO_SRC + SE | KO_SRC | 1.32 [1.19, 1.46] | -0.62 | 0.71 | 0.11 [0.10, 0.12] | 0.09 | 0.02 |

Interaction I = dm(edit | KO) - dm(edit | no KO), in e units [CI]: SRC O1 0.00 [-0.00, 0.00]; SRC PV 0.00 [0.00, 0.00]; RND0 O1 0.00 [-0.00, 0.00]; RND0 PV 0.00 [0.00, 0.00]; RND1 O1 0.00 [-0.00, 0.00]; RND1 PV 0.00 [-0.00, 0.00]; RND2 O1 0.00 [-0.00, 0.00]; RND2 PV 0.00 [-0.00, 0.00]; NON O1 0.00 [0.00, 0.00]; NON PV 0.00 [0.00, 0.00]; ALL O1 -0.00 [-0.00, 0.00]; ALL PV -0.00 [-0.00, 0.00]

D_KO_SRC (B minus A under KO_SRC): 1.35 [1.21, 1.50]. O1 e normalised by D_KO: 0.00 [0.00, 0.01].

By relation (e): {"T1": {"O1": 0.0001, "KO_SRC+O1|vsKO": 0.0004, "KO_SRC_top1A": 0.1189, "SE+KO_SRC|vsKO": 0.163}, "T2": {"O1": -0.0004, "KO_SRC+O1|vsKO": 0.0008, "KO_SRC_top1A": 0.0263, "SE+KO_SRC|vsKO": 0.0722}}

Positive control: {"SE_e": [0.9991, 0.9978, 1.0003, 333, 333], "SE_dm": [10.7823, 10.4716, 11.0747, 333, 333], "SE_given_KO_SRC_e": [0.1112, 0.1008, 0.1217, 333, 333], "SE_given_KO_SRC_dm": [1.3221, 1.1901, 1.4647, 333, 333], "dm_ratio_KO_over_noKO": 0.1226, "structural_zero_capital": "0/0", "pass": true}
