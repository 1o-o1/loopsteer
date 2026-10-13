# K17 FREEZE summary (RA-3, X4a branch freeze, ByteDance/Ouro-2.6B-Thinking)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| addition | 263 | 77 | 0.695 [0.667, 0.723] (263 / 77 cl) | 1.815 [1.793, 1.836] (263 / 77 cl) | 0.583 [0.568, 0.597] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [0.999, 1.001] (263 / 77 cl) | R1 falsified |

## Ratio of means (CPU pilot form) and strata

- addition: {"E0": 0.7039, "ans_frz_mlp": 1.8099, "ans_frz_attn": 0.5772}; mean D 16.485; strata {"op1": 129, "op2": 134}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 263, "freeze_no_edit_identity": 263, "hook_identity": 263, "pc_exact": 263, "readout_identity": 263, "source_attn_frozen_identity": 263, "source_edit_reaches_r": 263, "sz_exact": 263}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### addition

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.954 [0.940, 0.967] (263 / 77 cl) |
| E0 | 2 | 0.869 [0.854, 0.883] (263 / 77 cl) | 0.827 [0.811, 0.842] (263 / 77 cl) | 0.745 [0.721, 0.770] (263 / 77 cl) |
| E0 | 3 | 0.853 [0.832, 0.875] (263 / 77 cl) | 0.727 [0.704, 0.749] (263 / 77 cl) | 0.693 [0.669, 0.718] (263 / 77 cl) |
| E0 | 4 | 0.872 [0.847, 0.897] (263 / 77 cl) | 0.685 [0.662, 0.707] (263 / 77 cl) | 0.695 [0.667, 0.723] (263 / 77 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.954 [0.940, 0.967] (263 / 77 cl) |
| ans_frz_mlp | 2 | 1.528 [1.507, 1.549] (263 / 77 cl) | 0.446 [0.439, 0.453] (263 / 77 cl) | 1.298 [1.278, 1.317] (263 / 77 cl) |
| ans_frz_mlp | 3 | 2.196 [2.152, 2.242] (263 / 77 cl) | 0.347 [0.341, 0.352] (263 / 77 cl) | 1.711 [1.687, 1.734] (263 / 77 cl) |
| ans_frz_mlp | 4 | 2.586 [2.532, 2.644] (263 / 77 cl) | 0.353 [0.347, 0.359] (263 / 77 cl) | 1.815 [1.793, 1.836] (263 / 77 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.954 [0.940, 0.967] (263 / 77 cl) |
| ans_frz_attn | 2 | 0.879 [0.864, 0.896] (263 / 77 cl) | 0.621 [0.611, 0.631] (263 / 77 cl) | 0.703 [0.687, 0.717] (263 / 77 cl) |
| ans_frz_attn | 3 | 0.885 [0.862, 0.910] (263 / 77 cl) | 0.524 [0.510, 0.538] (263 / 77 cl) | 0.626 [0.613, 0.639] (263 / 77 cl) |
| ans_frz_attn | 4 | 0.921 [0.895, 0.948] (263 / 77 cl) | 0.490 [0.476, 0.503] (263 / 77 cl) | 0.583 [0.568, 0.597] (263 / 77 cl) |

## Checks

```
{
 "zero_edit_bit_identical": true,
 "freeze_no_edit_bit_identical": true,
 "source_attn_frozen_e_zero_bit_exact": true,
 "source_edit_reaches_r_all": true,
 "pc_exact_all": true,
 "sz_exact_all": true,
 "readout_identity_all": true,
 "n_items_facts": null,
 "n_clusters_facts": null,
 "n_items_addition": 263,
 "n_clusters_addition": 77,
 "n_items_code_reads": null,
 "n_items_code_sums": null,
 "n_clusters_code": null,
 "n_items_twostep": null,
 "n_clusters_twostep": null,
 "e_attn_facts_lower_bound_ge_0.5": null,
 "e_mlp_facts_upper_bound_le_0.15": null,
 "gpu_minutes": 17.35,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
