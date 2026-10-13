# K16 PATH-IFM summary

Final readout only. Every scored row is one batch-1 forward. dm = change in z(B) - z(A) versus the stated baseline; e = dm / D only where D >= 2.0; flip = top-1 is B. CI: 95% percentile bootstrap over entity clusters (D1: A-side subject (rel, a), plan amendment A1; D2: A problem), 1000 resamples, numpy default_rng(0); [mean, lo, hi, n_items, n_clusters].

GPU minutes: 0.22. D4/D5: not run (no data at launch).

## Hook checks

| model | identity (all bit-identical) | knockout placement ok | injection placement ok | attn recompute max abs |
|---|---|---|---|---|
| ifm_p2m_ortho | None | n/a | n/a | None |

## Items

```
{
 "ifm_p2m_ortho": {
  "D5": {
   "n": 0,
   "clusters": 0,
   "by_rel": {},
   "scanned": 0,
   "D_lt_2": 0
  }
 }
}
```
