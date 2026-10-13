# K17 FREEZE summary (RA-3, X4a branch freeze, ifm_p1_ouro)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 293 | 74 | -0.000 [-0.002, 0.001] dyadic [-0.002, 0.001] (293 / 74 cl) | 0.016 [0.014, 0.018] dyadic [0.011, 0.021] (293 / 74 cl) | 0.001 [-0.000, 0.001] dyadic [-0.001, 0.002] (293 / 74 cl) | 1.000 [1.000, 1.000] (293 / 74 cl) | 1.001 [1.000, 1.002] (293 / 74 cl) | R1 supported |

## Ratio of means (CPU pilot form) and strata

- facts: {"E0": -0.0005, "ans_frz_mlp": 0.0161, "ans_frz_attn": 0.0004}; mean D 23.409; strata {"capital": 220, "city": 73}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 293, "freeze_no_edit_identity": 293, "hook_identity": 293, "pc_exact": 293, "readout_identity": 293, "source_attn_frozen_identity": 293, "source_edit_reaches_r": 293, "sz_exact": 293}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (293 / 74 cl) | 1.000 [1.000, 1.000] (293 / 74 cl) | 0.035 [0.032, 0.038] (293 / 74 cl) |
| E0 | 2 | 0.540 [0.532, 0.548] (293 / 74 cl) | 0.640 [0.629, 0.652] (293 / 74 cl) | 0.003 [0.001, 0.004] (293 / 74 cl) |
| E0 | 3 | 0.212 [0.205, 0.220] (293 / 74 cl) | 0.333 [0.310, 0.356] (293 / 74 cl) | 0.000 [-0.001, 0.001] (293 / 74 cl) |
| E0 | 4 | 0.037 [0.035, 0.040] (293 / 74 cl) | 0.096 [0.074, 0.117] (293 / 74 cl) | -0.000 [-0.002, 0.001] (293 / 74 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (293 / 74 cl) | 1.000 [1.000, 1.000] (293 / 74 cl) | 0.035 [0.032, 0.038] (293 / 74 cl) |
| ans_frz_mlp | 2 | 0.924 [0.893, 0.953] (293 / 74 cl) | 0.402 [0.388, 0.417] (293 / 74 cl) | 0.025 [0.022, 0.028] (293 / 74 cl) |
| ans_frz_mlp | 3 | 0.649 [0.616, 0.682] (293 / 74 cl) | 0.277 [0.264, 0.291] (293 / 74 cl) | 0.020 [0.018, 0.023] (293 / 74 cl) |
| ans_frz_mlp | 4 | 0.081 [0.074, 0.088] (293 / 74 cl) | 0.079 [0.071, 0.087] (293 / 74 cl) | 0.016 [0.014, 0.018] (293 / 74 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (293 / 74 cl) | 1.000 [1.000, 1.000] (293 / 74 cl) | 0.035 [0.032, 0.038] (293 / 74 cl) |
| ans_frz_attn | 2 | 0.586 [0.576, 0.597] (293 / 74 cl) | 0.560 [0.551, 0.569] (293 / 74 cl) | 0.004 [0.003, 0.005] (293 / 74 cl) |
| ans_frz_attn | 3 | 0.246 [0.239, 0.253] (293 / 74 cl) | 0.267 [0.251, 0.283] (293 / 74 cl) | 0.001 [0.001, 0.002] (293 / 74 cl) |
| ans_frz_attn | 4 | 0.038 [0.036, 0.040] (293 / 74 cl) | 0.080 [0.062, 0.098] (293 / 74 cl) | 0.001 [-0.000, 0.001] (293 / 74 cl) |

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
 "n_items_facts": 293,
 "n_clusters_facts": 74,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "n_items_code_reads": null,
 "n_items_code_sums": null,
 "n_clusters_code": null,
 "n_items_twostep": null,
 "n_clusters_twostep": null,
 "e_attn_facts_lower_bound_ge_0.5": false,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 12.12,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
