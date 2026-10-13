# K17 FREEZE RAVEN summary (RA-19 item 3, X4a branch freeze, mcleish_llama32_r32, K = 32)

Plan sha256 ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e. Mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| addition | 263 | 77 | -0.000 [-0.001, 0.000] (263 / 77 cl) | 0.000 [-0.000, 0.000] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) | R1 supported |

## Ratio of means, strata, identities

- addition: {"E0": -0.0002, "ans_frz_mlp": 0.0001, "ans_frz_attn": 0.0003}; mean D 16.404 (263 items with D >= 2); strata {"op1": 129, "op2": 134}; identity counts {"ans_loop1_state_is_donor": 263, "freeze_no_edit_identity": 263, "hook_identity": 263, "pc_exact": 263, "readout_identity": 263, "source_attn_frozen_identity": 263, "source_edit_reaches_r": 263, "sz_exact": 263}

## G5 split at selected loop ends

### addition

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.708 [0.688, 0.728] (263 / 77 cl) |
| E0 | 2 | 0.221 [0.213, 0.229] (263 / 77 cl) | 0.201 [0.176, 0.227] (263 / 77 cl) | 0.002 [-0.002, 0.006] (263 / 77 cl) |
| E0 | 4 | 0.049 [0.046, 0.052] (263 / 77 cl) | 0.035 [0.013, 0.057] (263 / 77 cl) | -0.000 [-0.001, 0.001] (263 / 77 cl) |
| E0 | 8 | 0.020 [0.019, 0.020] (263 / 77 cl) | 0.020 [0.008, 0.031] (263 / 77 cl) | 0.001 [0.000, 0.001] (263 / 77 cl) |
| E0 | 16 | 0.019 [0.019, 0.020] (263 / 77 cl) | 0.019 [0.009, 0.029] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| E0 | 24 | 0.019 [0.019, 0.020] (263 / 77 cl) | 0.027 [0.015, 0.039] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| E0 | 31 | 0.020 [0.019, 0.020] (263 / 77 cl) | 0.012 [-0.001, 0.026] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) |
| E0 | 32 | 0.019 [0.019, 0.020] (263 / 77 cl) | 0.017 [0.006, 0.028] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.708 [0.688, 0.728] (263 / 77 cl) |
| ans_frz_mlp | 2 | 0.395 [0.388, 0.402] (263 / 77 cl) | 0.112 [0.108, 0.117] (263 / 77 cl) | 0.025 [0.020, 0.029] (263 / 77 cl) |
| ans_frz_mlp | 4 | 0.078 [0.077, 0.080] (263 / 77 cl) | 0.003 [-0.001, 0.006] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_mlp | 8 | 0.018 [0.018, 0.019] (263 / 77 cl) | 0.003 [-0.000, 0.007] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) |
| ans_frz_mlp | 16 | 0.018 [0.017, 0.018] (263 / 77 cl) | 0.006 [0.002, 0.010] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_mlp | 24 | 0.018 [0.017, 0.018] (263 / 77 cl) | 0.006 [0.003, 0.009] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_mlp | 31 | 0.018 [0.017, 0.018] (263 / 77 cl) | 0.002 [-0.002, 0.006] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_mlp | 32 | 0.018 [0.017, 0.018] (263 / 77 cl) | 0.002 [-0.001, 0.006] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (263 / 77 cl) | 1.000 [1.000, 1.000] (263 / 77 cl) | 0.708 [0.688, 0.728] (263 / 77 cl) |
| ans_frz_attn | 2 | 0.196 [0.190, 0.201] (263 / 77 cl) | 0.163 [0.152, 0.174] (263 / 77 cl) | 0.001 [-0.002, 0.003] (263 / 77 cl) |
| ans_frz_attn | 4 | 0.034 [0.033, 0.036] (263 / 77 cl) | 0.007 [-0.004, 0.018] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) |
| ans_frz_attn | 8 | 0.015 [0.014, 0.015] (263 / 77 cl) | 0.015 [0.009, 0.022] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) |
| ans_frz_attn | 16 | 0.014 [0.014, 0.015] (263 / 77 cl) | 0.009 [0.004, 0.015] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_attn | 24 | 0.014 [0.014, 0.015] (263 / 77 cl) | 0.016 [0.010, 0.022] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_attn | 31 | 0.014 [0.014, 0.015] (263 / 77 cl) | 0.005 [-0.002, 0.011] (263 / 77 cl) | -0.000 [-0.001, 0.000] (263 / 77 cl) |
| ans_frz_attn | 32 | 0.015 [0.014, 0.015] (263 / 77 cl) | 0.006 [0.000, 0.012] (263 / 77 cl) | 0.000 [-0.000, 0.001] (263 / 77 cl) |

## Checks

```
{
 "model": "mcleish_llama32_r32",
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
 "n_items_all": 263,
 "n_items_facts": null,
 "n_clusters_facts": null,
 "n_items_addition": 263,
 "n_clusters_addition": 77,
 "verdict_facts": null,
 "verdict_addition": "R1 supported",
 "e_attn_facts_lower_bound_ge_0.5": null,
 "e_attn_facts_upper_bound_le_0.15": null,
 "e_mlp_facts_upper_bound_le_0.15": null,
 "gpu_minutes": 25.23,
 "outputs_hashed": true,
 "plan_sha256": "ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e"
}
```
