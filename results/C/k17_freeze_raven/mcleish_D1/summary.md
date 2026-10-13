# K17 FREEZE RAVEN summary (RA-19 item 3, X4a branch freeze, mcleish_llama32_r32, K = 32)

Plan sha256 ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e. Mean of per-item e = dm / D (items with D >= 2.0) [90% percentile cluster bootstrap, 10000 resamples, seed 0] (items with e defined / clusters).

| cell | n items | clusters | E0 e | e_attn (MLP frozen) | e_mlp (attention frozen) | pc e | src e | verdict |
|---|---|---|---|---|---|---|---|---|
| facts | 321 | 66 | -0.000 [-0.000, 0.000] dyadic [-0.001, 0.000] (321 / 66 cl) | 0.000 [-0.000, 0.000] dyadic [-0.000, 0.000] (321 / 66 cl) | -0.000 [-0.000, 0.000] dyadic [-0.000, 0.000] (321 / 66 cl) | 1.000 [1.000, 1.000] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) | R1 supported |

## Ratio of means, strata, identities

- facts: {"E0": -0.0001, "ans_frz_mlp": 0.0, "ans_frz_attn": -0.0001}; mean D 20.886 (321 items with D >= 2); strata {"capital": 133, "city": 65, "city_mid": 123}; identity counts {"ans_loop1_state_is_donor": 321, "freeze_no_edit_identity": 321, "hook_identity": 321, "pc_exact": 321, "readout_identity": 321, "source_attn_frozen_identity": 321, "source_edit_reaches_r": 321, "sz_exact": 321}

## G5 split at selected loop ends

### facts

| row | loop | norm ratio | cos | e_j |
|---|---|---|---|---|
| E0 | 1 | 1.000 [1.000, 1.000] (321 / 66 cl) | 1.000 [1.000, 1.000] (321 / 66 cl) | 0.309 [0.280, 0.337] (321 / 66 cl) |
| E0 | 2 | 0.313 [0.294, 0.332] (321 / 66 cl) | 0.415 [0.389, 0.441] (321 / 66 cl) | 0.011 [0.005, 0.017] (321 / 66 cl) |
| E0 | 4 | 0.081 [0.070, 0.093] (321 / 66 cl) | 0.102 [0.077, 0.126] (321 / 66 cl) | -0.004 [-0.006, -0.002] (321 / 66 cl) |
| E0 | 8 | 0.019 [0.016, 0.022] (321 / 66 cl) | 0.021 [0.007, 0.036] (321 / 66 cl) | -0.000 [-0.001, 0.000] (321 / 66 cl) |
| E0 | 16 | 0.013 [0.013, 0.014] (321 / 66 cl) | 0.017 [0.005, 0.029] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| E0 | 24 | 0.013 [0.013, 0.013] (321 / 66 cl) | 0.006 [-0.007, 0.019] (321 / 66 cl) | 0.000 [-0.000, 0.000] (321 / 66 cl) |
| E0 | 31 | 0.013 [0.013, 0.014] (321 / 66 cl) | -0.001 [-0.014, 0.011] (321 / 66 cl) | -0.000 [-0.001, 0.000] (321 / 66 cl) |
| E0 | 32 | 0.013 [0.013, 0.014] (321 / 66 cl) | -0.003 [-0.012, 0.007] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 1 | 1.000 [1.000, 1.000] (321 / 66 cl) | 1.000 [1.000, 1.000] (321 / 66 cl) | 0.309 [0.280, 0.337] (321 / 66 cl) |
| ans_frz_mlp | 2 | 0.346 [0.339, 0.352] (321 / 66 cl) | 0.159 [0.153, 0.164] (321 / 66 cl) | 0.016 [0.011, 0.021] (321 / 66 cl) |
| ans_frz_mlp | 4 | 0.072 [0.070, 0.075] (321 / 66 cl) | 0.018 [0.013, 0.024] (321 / 66 cl) | -0.001 [-0.001, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 8 | 0.013 [0.013, 0.014] (321 / 66 cl) | 0.009 [0.005, 0.012] (321 / 66 cl) | 0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 16 | 0.012 [0.011, 0.012] (321 / 66 cl) | 0.005 [0.002, 0.008] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 24 | 0.012 [0.011, 0.012] (321 / 66 cl) | 0.003 [-0.000, 0.006] (321 / 66 cl) | 0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 31 | 0.012 [0.011, 0.012] (321 / 66 cl) | 0.001 [-0.002, 0.005] (321 / 66 cl) | 0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_mlp | 32 | 0.012 [0.011, 0.012] (321 / 66 cl) | 0.004 [0.001, 0.007] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_attn | 1 | 1.000 [1.000, 1.000] (321 / 66 cl) | 1.000 [1.000, 1.000] (321 / 66 cl) | 0.309 [0.280, 0.337] (321 / 66 cl) |
| ans_frz_attn | 2 | 0.282 [0.271, 0.293] (321 / 66 cl) | 0.447 [0.428, 0.465] (321 / 66 cl) | 0.013 [0.007, 0.019] (321 / 66 cl) |
| ans_frz_attn | 4 | 0.057 [0.053, 0.063] (321 / 66 cl) | 0.123 [0.103, 0.144] (321 / 66 cl) | -0.002 [-0.003, -0.001] (321 / 66 cl) |
| ans_frz_attn | 8 | 0.012 [0.011, 0.012] (321 / 66 cl) | 0.022 [0.011, 0.032] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_attn | 16 | 0.011 [0.010, 0.011] (321 / 66 cl) | 0.004 [-0.007, 0.015] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_attn | 24 | 0.011 [0.010, 0.011] (321 / 66 cl) | 0.012 [0.001, 0.022] (321 / 66 cl) | 0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_attn | 31 | 0.011 [0.010, 0.011] (321 / 66 cl) | 0.009 [-0.002, 0.021] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |
| ans_frz_attn | 32 | 0.010 [0.010, 0.011] (321 / 66 cl) | 0.002 [-0.010, 0.013] (321 / 66 cl) | -0.000 [-0.000, 0.000] (321 / 66 cl) |

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
 "n_items_all": 321,
 "n_items_facts": 321,
 "n_clusters_facts": 66,
 "n_items_addition": null,
 "n_clusters_addition": null,
 "verdict_facts": "R1 supported",
 "verdict_addition": null,
 "e_attn_facts_lower_bound_ge_0.5": false,
 "e_attn_facts_upper_bound_le_0.15": true,
 "e_mlp_facts_upper_bound_le_0.15": true,
 "gpu_minutes": 21.49,
 "outputs_hashed": true,
 "plan_sha256": "ee8c583f5a9652485f2508f5ac15b14266387b168d63726864f9849aba5e936e"
}
```
