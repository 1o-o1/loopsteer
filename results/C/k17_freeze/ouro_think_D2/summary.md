# K17 FREEZE summary (RA-3, X4a branch freeze, ByteDance/Ouro-1.4B-Thinking)

Plan sha256 de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424. Cells: mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| addition | 263 | 77 | 0.750 [0.721, 0.778] (263 / 77 cl) | 1.597 [1.576, 1.619] (263 / 77 cl) | 0.761 [0.748, 0.775] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.998 [0.996, 0.999] (263 / 77 cl) | R1 falsified |

## Ratio of means (CPU pilot form) and strata

- addition: {"E0": 0.7452, "ans_frz_mlp": 1.589, "ans_frz_attn": 0.7587}; mean D 15.034; strata {"op1": 129, "op2": 134}; clean top-1 A 1.000 B 1.000; identity counts {"ans_loop1_state_is_donor": 263, "freeze_no_edit_identity": 263, "hook_identity": 263, "pc_exact": 263, "readout_identity": 263, "source_attn_frozen_identity": 263, "source_edit_reaches_r": 263, "sz_exact": 263}

## G5 split per loop end: norm ratio ||delta|| / ||H_B - H_A||, cosine to H_B - H_A, e_j = dm_j / D

### addition

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.088 [1.067, 1.108] (263 / 77 cl) |
| E0 | 2 | 0.922 [0.904, 0.939] (263 / 77 cl) | 0.869 [0.848, 0.890] (263 / 77 cl) | 0.867 [0.833, 0.899] (263 / 77 cl) |
| E0 | 3 | 0.918 [0.898, 0.938] (263 / 77 cl) | 0.852 [0.826, 0.877] (263 / 77 cl) | 0.817 [0.784, 0.849] (263 / 77 cl) |
| E0 | 4 | 0.898 [0.878, 0.917] (263 / 77 cl) | 0.850 [0.824, 0.874] (263 / 77 cl) | 0.750 [0.721, 0.778] (263 / 77 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.088 [1.067, 1.108] (263 / 77 cl) |
| ans_frz_mlp | 2 | 2.034 [1.983, 2.085] (263 / 77 cl) | 0.318 [0.307, 0.329] (263 / 77 cl) | 1.421 [1.394, 1.448] (263 / 77 cl) |
| ans_frz_mlp | 3 | 2.753 [2.700, 2.805] (263 / 77 cl) | 0.259 [0.253, 0.265] (263 / 77 cl) | 1.593 [1.569, 1.618] (263 / 77 cl) |
| ans_frz_mlp | 4 | 3.154 [3.097, 3.213] (263 / 77 cl) | 0.251 [0.244, 0.257] (263 / 77 cl) | 1.597 [1.576, 1.619] (263 / 77 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.088 [1.067, 1.108] (263 / 77 cl) |
| ans_frz_attn | 2 | 1.206 [1.174, 1.238] (263 / 77 cl) | 0.519 [0.509, 0.529] (263 / 77 cl) | 0.850 [0.834, 0.865] (263 / 77 cl) |
| ans_frz_attn | 3 | 1.348 [1.318, 1.379] (263 / 77 cl) | 0.484 [0.471, 0.498] (263 / 77 cl) | 0.799 [0.784, 0.814] (263 / 77 cl) |
| ans_frz_attn | 4 | 1.516 [1.482, 1.552] (263 / 77 cl) | 0.429 [0.417, 0.442] (263 / 77 cl) | 0.761 [0.748, 0.775] (263 / 77 cl) |

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
 "gpu_minutes": 8.81,
 "gpu_minutes_le_90": true,
 "outputs_hashed": true,
 "plan_sha256": "de416161128018f699ac9a8314b5c725ff5841be54a79ed8631e63d63d2d3424"
}
```
