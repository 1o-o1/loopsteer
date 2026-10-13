# K17 FREEZE summary (RA-3, X4a branch freeze, ifm_p1_ouro)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| addition | 263 | 77 | 0.003 [0.002, 0.004] (263 / 77 cl) | 0.001 [0.001, 0.002] (263 / 77 cl) | 0.004 [0.003, 0.006] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.001] (263 / 77 cl) | R1 supported |

## Ratio of means (CPU pilot form) and strata

- addition: {"E0": 0.0032, "ans_frz_mlp": 0.0013, "ans_frz_attn": 0.0046}; mean D 18.594; strata {"op1": 129, "op2": 134}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 263, "freeze_no_edit_identity": 263, "hook_identity": 263, "pc_exact": 263, "readout_identity": 263, "source_attn_frozen_identity": 263, "source_edit_reaches_r": 263, "sz_exact": 263}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### addition

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.006 [0.004, 0.007] (263 / 77 cl) |
| E0 | 2 | 0.215 [0.205, 0.225] (263 / 77 cl) | 0.312 [0.292, 0.333] (263 / 77 cl) | 0.005 [0.003, 0.007] (263 / 77 cl) |
| E0 | 3 | 0.047 [0.044, 0.049] (263 / 77 cl) | 0.122 [0.095, 0.150] (263 / 77 cl) | 0.002 [0.001, 0.003] (263 / 77 cl) |
| E0 | 4 | 0.038 [0.036, 0.040] (263 / 77 cl) | 0.089 [0.065, 0.112] (263 / 77 cl) | 0.003 [0.002, 0.004] (263 / 77 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.006 [0.004, 0.007] (263 / 77 cl) |
| ans_frz_mlp | 2 | 0.348 [0.336, 0.360] (263 / 77 cl) | 0.264 [0.255, 0.273] (263 / 77 cl) | 0.002 [0.001, 0.004] (263 / 77 cl) |
| ans_frz_mlp | 3 | 0.101 [0.096, 0.108] (263 / 77 cl) | 0.041 [0.036, 0.046] (263 / 77 cl) | 0.002 [0.001, 0.003] (263 / 77 cl) |
| ans_frz_mlp | 4 | 0.045 [0.043, 0.048] (263 / 77 cl) | 0.007 [0.003, 0.011] (263 / 77 cl) | 0.001 [0.001, 0.002] (263 / 77 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.006 [0.004, 0.007] (263 / 77 cl) |
| ans_frz_attn | 2 | 0.176 [0.170, 0.182] (263 / 77 cl) | 0.314 [0.303, 0.324] (263 / 77 cl) | 0.005 [0.004, 0.006] (263 / 77 cl) |
| ans_frz_attn | 3 | 0.046 [0.044, 0.049] (263 / 77 cl) | 0.133 [0.110, 0.156] (263 / 77 cl) | 0.003 [0.002, 0.004] (263 / 77 cl) |
| ans_frz_attn | 4 | 0.042 [0.039, 0.044] (263 / 77 cl) | 0.131 [0.110, 0.152] (263 / 77 cl) | 0.004 [0.003, 0.006] (263 / 77 cl) |

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
 "gpu_minutes": 10.77,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
