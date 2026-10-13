# K17 FREEZE summary (RA-3, X4a branch freeze, ByteDance/Ouro-2.6B-Thinking)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 266 | 67 | 0.002 [0.001, 0.003] dyadic [0.000, 0.004] (266 / 67 cl) | 1.297 [1.237, 1.359] dyadic [1.197, 1.405] (266 / 67 cl) | 0.006 [0.004, 0.008] dyadic [0.003, 0.009] (266 / 67 cl) | 1.000 [1.000, 1.000] (266 / 67 cl) | 1.000 [0.999, 1.000] (266 / 67 cl) | R1 falsified |

## Ratio of means (CPU pilot form) and strata

- facts: {"E0": 0.0018, "ans_frz_mlp": 1.2645, "ans_frz_attn": 0.0059}; mean D 20.363; strata {"capital": 230, "city": 36}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 266, "freeze_no_edit_identity": 266, "hook_identity": 266, "pc_exact": 266, "readout_identity": 266, "source_attn_frozen_identity": 266, "source_edit_reaches_r": 266, "sz_exact": 266}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (266 / 67 cl) | 1.000 [1.000, 1.000] (266 / 67 cl) | 0.205 [0.182, 0.231] (266 / 67 cl) |
| E0 | 2 | 0.285 [0.268, 0.302] (266 / 67 cl) | 0.260 [0.221, 0.299] (266 / 67 cl) | 0.008 [-0.001, 0.018] (266 / 67 cl) |
| E0 | 3 | 0.109 [0.103, 0.115] (266 / 67 cl) | 0.075 [0.056, 0.094] (266 / 67 cl) | 0.001 [-0.001, 0.003] (266 / 67 cl) |
| E0 | 4 | 0.057 [0.053, 0.062] (266 / 67 cl) | 0.055 [0.035, 0.076] (266 / 67 cl) | 0.002 [0.001, 0.003] (266 / 67 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (266 / 67 cl) | 1.000 [1.000, 1.000] (266 / 67 cl) | 0.205 [0.182, 0.231] (266 / 67 cl) |
| ans_frz_mlp | 2 | 0.756 [0.721, 0.797] (266 / 67 cl) | 0.453 [0.440, 0.466] (266 / 67 cl) | 0.464 [0.408, 0.529] (266 / 67 cl) |
| ans_frz_mlp | 3 | 1.319 [1.265, 1.381] (266 / 67 cl) | 0.408 [0.398, 0.419] (266 / 67 cl) | 0.934 [0.870, 1.004] (266 / 67 cl) |
| ans_frz_mlp | 4 | 1.937 [1.883, 1.995] (266 / 67 cl) | 0.434 [0.425, 0.445] (266 / 67 cl) | 1.297 [1.237, 1.359] (266 / 67 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (266 / 67 cl) | 1.000 [1.000, 1.000] (266 / 67 cl) | 0.205 [0.182, 0.231] (266 / 67 cl) |
| ans_frz_attn | 2 | 0.287 [0.269, 0.306] (266 / 67 cl) | 0.221 [0.205, 0.238] (266 / 67 cl) | 0.027 [0.021, 0.033] (266 / 67 cl) |
| ans_frz_attn | 3 | 0.150 [0.138, 0.163] (266 / 67 cl) | 0.092 [0.075, 0.111] (266 / 67 cl) | 0.006 [0.003, 0.010] (266 / 67 cl) |
| ans_frz_attn | 4 | 0.084 [0.076, 0.093] (266 / 67 cl) | 0.065 [0.047, 0.084] (266 / 67 cl) | 0.006 [0.004, 0.008] (266 / 67 cl) |

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
 "n_items_facts": 266,
 "n_clusters_facts": 67,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "n_items_code_reads": null,
 "n_items_code_sums": null,
 "n_clusters_code": null,
 "n_items_twostep": null,
 "n_clusters_twostep": null,
 "e_attn_facts_lower_bound_ge_0.5": true,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 17.25,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
