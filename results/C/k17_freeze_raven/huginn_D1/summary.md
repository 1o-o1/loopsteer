# K17 FREEZE RAVEN summary (RA-19 item 3, X4a branch freeze, huginn_0125, K = 32)

Plan sha256 ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e. Mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 66 | 29 | -0.000 [-0.001, 0.000] dyadic [-0.002, 0.001] (66 / 29 cl) | -0.001 [-0.001, -0.000] dyadic [-0.001, 0.000] (66 / 29 cl) | -0.000 [-0.001, 0.000] dyadic [-0.001, 0.001] (66 / 29 cl) | 1.000 [1.000, 1.000] (66 / 29 cl) | -0.001 [-0.001, 0.000] (66 / 29 cl) | R1 supported |

## Ratio of means, strata, identities

- facts: {"E0": -0.0003, "ans_frz_mlp": -0.0004, "ans_frz_attn": -0.0003}; mean D 19.600 (66 items with D >= 2); strata {"capital": 5, "city_mid": 61}; identity counts {"ans_loop1_state_is_donor": 66, "freeze_no_edit_identity": 66, "hook_identity": 66, "pc_exact": 66, "readout_identity": 66, "source_attn_frozen_identity": 66, "source_edit_reaches_r": 66, "sz_exact": 66}

## G5 split at selected loop ends

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (66 / 29 cl) | 1.000 [1.000, 1.000] (66 / 29 cl) | 0.087 [0.071, 0.102] (66 / 29 cl) |
| E0 | 2 | 0.341 [0.324, 0.360] (66 / 29 cl) | 0.567 [0.528, 0.605] (66 / 29 cl) | 0.039 [0.030, 0.048] (66 / 29 cl) |
| E0 | 4 | 0.131 [0.119, 0.144] (66 / 29 cl) | 0.222 [0.198, 0.245] (66 / 29 cl) | 0.002 [0.001, 0.004] (66 / 29 cl) |
| E0 | 8 | 0.055 [0.051, 0.060] (66 / 29 cl) | 0.053 [0.037, 0.071] (66 / 29 cl) | -0.002 [-0.002, -0.001] (66 / 29 cl) |
| E0 | 16 | 0.034 [0.032, 0.035] (66 / 29 cl) | 0.032 [0.016, 0.053] (66 / 29 cl) | 0.000 [-0.000, 0.001] (66 / 29 cl) |
| E0 | 24 | 0.033 [0.031, 0.034] (66 / 29 cl) | 0.006 [-0.006, 0.019] (66 / 29 cl) | -0.001 [-0.001, 0.000] (66 / 29 cl) |
| E0 | 31 | 0.032 [0.031, 0.034] (66 / 29 cl) | 0.004 [-0.008, 0.015] (66 / 29 cl) | 0.001 [-0.000, 0.001] (66 / 29 cl) |
| E0 | 32 | 0.032 [0.031, 0.034] (66 / 29 cl) | 0.016 [0.002, 0.030] (66 / 29 cl) | -0.000 [-0.001, 0.000] (66 / 29 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (66 / 29 cl) | 1.000 [1.000, 1.000] (66 / 29 cl) | 0.087 [0.071, 0.102] (66 / 29 cl) |
| ans_frz_mlp | 2 | 0.277 [0.261, 0.295] (66 / 29 cl) | 0.363 [0.341, 0.386] (66 / 29 cl) | 0.039 [0.031, 0.046] (66 / 29 cl) |
| ans_frz_mlp | 4 | 0.076 [0.072, 0.080] (66 / 29 cl) | 0.033 [0.026, 0.039] (66 / 29 cl) | -0.003 [-0.004, -0.001] (66 / 29 cl) |
| ans_frz_mlp | 8 | 0.026 [0.025, 0.028] (66 / 29 cl) | 0.006 [0.001, 0.012] (66 / 29 cl) | -0.000 [-0.001, 0.000] (66 / 29 cl) |
| ans_frz_mlp | 16 | 0.021 [0.020, 0.021] (66 / 29 cl) | 0.009 [0.004, 0.015] (66 / 29 cl) | 0.000 [-0.000, 0.001] (66 / 29 cl) |
| ans_frz_mlp | 24 | 0.020 [0.019, 0.021] (66 / 29 cl) | 0.004 [-0.002, 0.010] (66 / 29 cl) | 0.000 [-0.000, 0.001] (66 / 29 cl) |
| ans_frz_mlp | 31 | 0.020 [0.019, 0.021] (66 / 29 cl) | 0.010 [0.006, 0.013] (66 / 29 cl) | 0.000 [-0.000, 0.001] (66 / 29 cl) |
| ans_frz_mlp | 32 | 0.020 [0.019, 0.021] (66 / 29 cl) | 0.008 [0.004, 0.012] (66 / 29 cl) | -0.001 [-0.001, -0.000] (66 / 29 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (66 / 29 cl) | 1.000 [1.000, 1.000] (66 / 29 cl) | 0.087 [0.071, 0.102] (66 / 29 cl) |
| ans_frz_attn | 2 | 0.324 [0.306, 0.346] (66 / 29 cl) | 0.579 [0.541, 0.615] (66 / 29 cl) | 0.040 [0.032, 0.047] (66 / 29 cl) |
| ans_frz_attn | 4 | 0.117 [0.105, 0.130] (66 / 29 cl) | 0.263 [0.232, 0.293] (66 / 29 cl) | 0.004 [0.003, 0.006] (66 / 29 cl) |
| ans_frz_attn | 8 | 0.047 [0.042, 0.052] (66 / 29 cl) | 0.044 [0.029, 0.061] (66 / 29 cl) | -0.001 [-0.001, 0.000] (66 / 29 cl) |
| ans_frz_attn | 16 | 0.028 [0.027, 0.029] (66 / 29 cl) | 0.018 [0.009, 0.027] (66 / 29 cl) | -0.000 [-0.001, 0.000] (66 / 29 cl) |
| ans_frz_attn | 24 | 0.027 [0.026, 0.028] (66 / 29 cl) | 0.011 [0.003, 0.019] (66 / 29 cl) | -0.000 [-0.001, 0.000] (66 / 29 cl) |
| ans_frz_attn | 31 | 0.027 [0.026, 0.028] (66 / 29 cl) | 0.010 [0.004, 0.016] (66 / 29 cl) | 0.000 [-0.000, 0.001] (66 / 29 cl) |
| ans_frz_attn | 32 | 0.027 [0.026, 0.028] (66 / 29 cl) | 0.015 [0.010, 0.020] (66 / 29 cl) | -0.001 [-0.001, -0.000] (66 / 29 cl) |

## Checks

```
{
 "model": "huginn_0125",
 "zero_edit_bit_identical": true,
 "freeze_no_edit_bit_identical": true,
 "source_attn_frozen_e_zero_bit_exact": true,
 "source_edit_reaches_r_all": true,
 "pc_exact_all": true,
 "sz_exact_all": true,
 "ans_loop1_state_is_donor_all": true,
 "readout_identity_all": true,
 "freeze_rows_loopK_readout_equals_final_n": {
  "ans_frz_mlp": 0,
  "ans_frz_attn": 0
 },
 "n_items_all": 66,
 "n_items_facts": 66,
 "n_clusters_facts": 29,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "verdict_facts": "R1 supported",
 "verdict_addition": null,
 "e_attn_facts_lower_bound_ge_0.5": false,
 "e_attn_facts_upper_bound_le_0.15": true,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 13.6,
 "outputs_hashed": true,
 "plan_sha256": "ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e"
}
```
