# K17 FREEZE summary (RA-3, X4a branch freeze, ByteDance/Ouro-1.4B-Thinking)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 267 | 71 | 0.004 [0.002, 0.006] dyadic [0.001, 0.006] (267 / 71 cl) | 1.454 [1.399, 1.507] dyadic [1.366, 1.540] (267 / 71 cl) | 0.008 [0.004, 0.012] dyadic [0.003, 0.015] (267 / 71 cl) | 1.000 [1.000, 1.000] (267 / 71 cl) | 1.000 [0.999, 1.001] (267 / 71 cl) | R1 falsified |

## Ratio of means (CPU pilot form) and strata

- facts: {"E0": 0.0034, "ans_frz_mlp": 1.4271, "ans_frz_attn": 0.0071}; mean D 19.207; strata {"capital": 214, "city": 53}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 267, "freeze_no_edit_identity": 267, "hook_identity": 267, "pc_exact": 267, "readout_identity": 267, "source_attn_frozen_identity": 267, "source_edit_reaches_r": 267, "sz_exact": 267}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (267 / 71 cl) | 1.000 [1.000, 1.000] (267 / 71 cl) | 0.336 [0.312, 0.363] (267 / 71 cl) |
| E0 | 2 | 0.313 [0.297, 0.328] (267 / 71 cl) | 0.339 [0.306, 0.372] (267 / 71 cl) | 0.029 [0.021, 0.038] (267 / 71 cl) |
| E0 | 3 | 0.138 [0.128, 0.149] (267 / 71 cl) | 0.100 [0.067, 0.134] (267 / 71 cl) | 0.004 [-0.001, 0.008] (267 / 71 cl) |
| E0 | 4 | 0.077 [0.071, 0.083] (267 / 71 cl) | 0.099 [0.073, 0.125] (267 / 71 cl) | 0.004 [0.002, 0.006] (267 / 71 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (267 / 71 cl) | 1.000 [1.000, 1.000] (267 / 71 cl) | 0.336 [0.312, 0.363] (267 / 71 cl) |
| ans_frz_mlp | 2 | 0.967 [0.931, 1.003] (267 / 71 cl) | 0.603 [0.592, 0.612] (267 / 71 cl) | 0.724 [0.675, 0.777] (267 / 71 cl) |
| ans_frz_mlp | 3 | 1.503 [1.447, 1.560] (267 / 71 cl) | 0.530 [0.522, 0.538] (267 / 71 cl) | 1.257 [1.199, 1.314] (267 / 71 cl) |
| ans_frz_mlp | 4 | 2.016 [1.962, 2.072] (267 / 71 cl) | 0.507 [0.497, 0.516] (267 / 71 cl) | 1.454 [1.399, 1.507] (267 / 71 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (267 / 71 cl) | 1.000 [1.000, 1.000] (267 / 71 cl) | 0.336 [0.312, 0.363] (267 / 71 cl) |
| ans_frz_attn | 2 | 0.349 [0.331, 0.370] (267 / 71 cl) | 0.309 [0.288, 0.331] (267 / 71 cl) | 0.057 [0.048, 0.068] (267 / 71 cl) |
| ans_frz_attn | 3 | 0.205 [0.186, 0.227] (267 / 71 cl) | 0.139 [0.116, 0.163] (267 / 71 cl) | 0.018 [0.011, 0.025] (267 / 71 cl) |
| ans_frz_attn | 4 | 0.144 [0.128, 0.162] (267 / 71 cl) | 0.069 [0.050, 0.089] (267 / 71 cl) | 0.008 [0.004, 0.012] (267 / 71 cl) |

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
 "n_items_facts": 267,
 "n_clusters_facts": 71,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "n_items_code_reads": null,
 "n_items_code_sums": null,
 "n_clusters_code": null,
 "n_items_twostep": null,
 "n_clusters_twostep": null,
 "e_attn_facts_lower_bound_ge_0.5": true,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 9.2,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
