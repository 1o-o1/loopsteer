# K17 FREEZE RAVEN summary (RA-19 item 3, X4a branch freeze, huginn_0125, K = 32)

Plan sha256 ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e. Mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| addition | 242 | 74 | 0.000 [-0.001, 0.001] (242 / 74 cl) | -0.000 [-0.001, 0.000] (242 / 74 cl) | 0.001 [-0.000, 0.001] (242 / 74 cl) | 1.000 [1.000, 1.000] (242 / 74 cl) | -0.000 [-0.001, 0.001] (242 / 74 cl) | R1 supported |

## Ratio of means, strata, identities

- addition: {"E0": 0.0002, "ans_frz_mlp": -0.0002, "ans_frz_attn": 0.0006}; mean D 11.001 (242 items with D >= 2); strata {"op1": 128, "op2": 114}; identity counts {"ans_loop1_state_is_donor": 242, "freeze_no_edit_identity": 242, "hook_identity": 242, "pc_exact": 242, "readout_identity": 242, "source_attn_frozen_identity": 242, "source_edit_reaches_r": 242, "sz_exact": 242}

## G5 split at selected loop ends

### addition

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (242 / 74 cl) | 1.000 [1.000, 1.000] (242 / 74 cl) | 0.007 [0.005, 0.009] (242 / 74 cl) |
| E0 | 2 | 0.457 [0.442, 0.471] (242 / 74 cl) | 0.584 [0.565, 0.601] (242 / 74 cl) | 0.004 [0.003, 0.005] (242 / 74 cl) |
| E0 | 4 | 0.191 [0.184, 0.198] (242 / 74 cl) | 0.277 [0.264, 0.289] (242 / 74 cl) | 0.001 [-0.001, 0.002] (242 / 74 cl) |
| E0 | 8 | 0.048 [0.047, 0.050] (242 / 74 cl) | 0.065 [0.057, 0.072] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| E0 | 16 | 0.030 [0.029, 0.031] (242 / 74 cl) | 0.018 [0.013, 0.022] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| E0 | 24 | 0.029 [0.029, 0.030] (242 / 74 cl) | 0.013 [0.009, 0.017] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| E0 | 31 | 0.029 [0.028, 0.030] (242 / 74 cl) | 0.016 [0.012, 0.020] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| E0 | 32 | 0.029 [0.028, 0.030] (242 / 74 cl) | 0.014 [0.010, 0.017] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (242 / 74 cl) | 1.000 [1.000, 1.000] (242 / 74 cl) | 0.007 [0.005, 0.009] (242 / 74 cl) |
| ans_frz_mlp | 2 | 0.388 [0.376, 0.401] (242 / 74 cl) | 0.433 [0.419, 0.445] (242 / 74 cl) | 0.001 [-0.000, 0.002] (242 / 74 cl) |
| ans_frz_mlp | 4 | 0.123 [0.119, 0.128] (242 / 74 cl) | 0.067 [0.063, 0.070] (242 / 74 cl) | -0.000 [-0.001, 0.000] (242 / 74 cl) |
| ans_frz_mlp | 8 | 0.028 [0.027, 0.029] (242 / 74 cl) | 0.010 [0.007, 0.013] (242 / 74 cl) | 0.001 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_mlp | 16 | 0.019 [0.019, 0.020] (242 / 74 cl) | 0.008 [0.006, 0.010] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_mlp | 24 | 0.019 [0.019, 0.020] (242 / 74 cl) | 0.009 [0.006, 0.011] (242 / 74 cl) | -0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_mlp | 31 | 0.019 [0.019, 0.020] (242 / 74 cl) | 0.008 [0.006, 0.010] (242 / 74 cl) | 0.000 [-0.000, 0.001] (242 / 74 cl) |
| ans_frz_mlp | 32 | 0.019 [0.019, 0.020] (242 / 74 cl) | 0.009 [0.007, 0.011] (242 / 74 cl) | -0.001 [-0.002, 0.000] (242 / 74 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (242 / 74 cl) | 1.000 [1.000, 1.000] (242 / 74 cl) | 0.007 [0.005, 0.009] (242 / 74 cl) |
| ans_frz_attn | 2 | 0.428 [0.414, 0.442] (242 / 74 cl) | 0.542 [0.525, 0.558] (242 / 74 cl) | 0.002 [0.001, 0.003] (242 / 74 cl) |
| ans_frz_attn | 4 | 0.161 [0.155, 0.167] (242 / 74 cl) | 0.225 [0.215, 0.233] (242 / 74 cl) | -0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_attn | 8 | 0.036 [0.035, 0.037] (242 / 74 cl) | 0.035 [0.031, 0.039] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_attn | 16 | 0.024 [0.023, 0.024] (242 / 74 cl) | 0.013 [0.010, 0.016] (242 / 74 cl) | -0.000 [-0.001, 0.001] (242 / 74 cl) |
| ans_frz_attn | 24 | 0.024 [0.023, 0.025] (242 / 74 cl) | 0.011 [0.008, 0.014] (242 / 74 cl) | 0.001 [-0.000, 0.001] (242 / 74 cl) |
| ans_frz_attn | 31 | 0.024 [0.023, 0.024] (242 / 74 cl) | 0.011 [0.009, 0.014] (242 / 74 cl) | 0.001 [0.001, 0.002] (242 / 74 cl) |
| ans_frz_attn | 32 | 0.024 [0.023, 0.025] (242 / 74 cl) | 0.011 [0.008, 0.014] (242 / 74 cl) | 0.000 [-0.001, 0.001] (242 / 74 cl) |

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
 "n_items_all": 242,
 "n_items_facts": null,
 "n_clusters_facts": null,
 "n_items_addition": 242,
 "n_clusters_addition": 74,
 "verdict_facts": null,
 "verdict_addition": "R1 supported",
 "e_attn_facts_lower_bound_ge_0.5": null,
 "e_attn_facts_upper_bound_le_0.15": null,
 "e_mlp_facts_upper_bound_le_0.15": null,
 "gpu_minutes": 27.47,
 "outputs_hashed": true,
 "plan_sha256": "ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e"
}
```
