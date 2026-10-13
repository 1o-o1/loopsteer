# K17 FREEZE summary (RA-3, X4a branch freeze, ifm_p1_ouro)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 3 | 2 | -0.002 [-0.003, 0.000] dyadic [-0.006, 0.000] (3 / 2 cl) | 0.017 [0.011, 0.028] dyadic [0.009, 0.028] (3 / 2 cl) | 0.001 [-0.001, 0.006] dyadic [-0.006, 0.006] (3 / 2 cl) | 1.000 [1.000, 1.000] (3 / 2 cl) | 1.000 [0.998, 1.003] (3 / 2 cl) | R1 supported |

## Ratio of means (CPU pilot form) and strata

- facts: {"E0": -0.0023, "ans_frz_mlp": 0.017, "ans_frz_attn": 0.0011}; mean D 18.354; strata {"capital": 3}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 3, "freeze_no_edit_identity": 3, "hook_identity": 3, "pc_exact": 3, "readout_identity": 3, "source_attn_frozen_identity": 3, "source_edit_reaches_r": 3, "sz_exact": 3}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (3 / 2 cl) | 1.000 [1.000, 1.000] (3 / 2 cl) | 0.043 [0.040, 0.048] (3 / 2 cl) |
| E0 | 2 | 0.590 [0.577, 0.615] (3 / 2 cl) | 0.580 [0.537, 0.602] (3 / 2 cl) | 0.002 [-0.000, 0.003] (3 / 2 cl) |
| E0 | 3 | 0.253 [0.245, 0.269] (3 / 2 cl) | 0.282 [0.278, 0.284] (3 / 2 cl) | -0.002 [-0.003, 0.000] (3 / 2 cl) |
| E0 | 4 | 0.030 [0.029, 0.033] (3 / 2 cl) | -0.063 [-0.098, 0.006] (3 / 2 cl) | -0.002 [-0.003, 0.000] (3 / 2 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (3 / 2 cl) | 1.000 [1.000, 1.000] (3 / 2 cl) | 0.043 [0.040, 0.048] (3 / 2 cl) |
| ans_frz_mlp | 2 | 1.031 [0.928, 1.082] (3 / 2 cl) | 0.364 [0.351, 0.371] (3 / 2 cl) | 0.032 [0.028, 0.042] (3 / 2 cl) |
| ans_frz_mlp | 3 | 0.668 [0.636, 0.684] (3 / 2 cl) | 0.251 [0.239, 0.273] (3 / 2 cl) | 0.023 [0.017, 0.033] (3 / 2 cl) |
| ans_frz_mlp | 4 | 0.053 [0.052, 0.054] (3 / 2 cl) | 0.074 [0.069, 0.084] (3 / 2 cl) | 0.017 [0.011, 0.028] (3 / 2 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (3 / 2 cl) | 1.000 [1.000, 1.000] (3 / 2 cl) | 0.043 [0.040, 0.048] (3 / 2 cl) |
| ans_frz_attn | 2 | 0.652 [0.596, 0.764] (3 / 2 cl) | 0.482 [0.475, 0.496] (3 / 2 cl) | 0.003 [-0.003, 0.006] (3 / 2 cl) |
| ans_frz_attn | 3 | 0.309 [0.276, 0.374] (3 / 2 cl) | 0.320 [0.249, 0.355] (3 / 2 cl) | 0.003 [0.002, 0.005] (3 / 2 cl) |
| ans_frz_attn | 4 | 0.033 [0.031, 0.038] (3 / 2 cl) | 0.064 [0.034, 0.079] (3 / 2 cl) | 0.001 [-0.001, 0.006] (3 / 2 cl) |

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
 "n_items_facts": 3,
 "n_clusters_facts": 2,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "n_items_code_reads": null,
 "n_items_code_sums": null,
 "n_clusters_code": null,
 "n_items_twostep": null,
 "n_clusters_twostep": null,
 "e_attn_facts_lower_bound_ge_0.5": false,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 0.37,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
