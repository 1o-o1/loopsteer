# RA-30 pilot/gsm8k

Items 35 (split 10). Pool: Table 2 single-card a1 scored list. Neuron arms carry only the last layer's MLP part of the loop-1 state difference.

| arm | n | CA | 90% CI | A | break | carry B | CA (floor and clean end in A) |
|---|---|---|---|---|---|---|---|
| base | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| text | 35 | 0.943 | [0.886, 1.000] | 0.029 | 0.029 | 0.943 | 0.943 |
| floor | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| cv1 | 35 | 0.857 | [0.743, 0.943] | 0.086 | 0.057 | 0.886 | 0.857 |
| cv2 | 35 | 0.600 | [0.457, 0.743] | 0.371 | 0.029 | 0.600 | 0.600 |
| cv3 | 35 | 0.400 | [0.257, 0.543] | 0.571 | 0.029 | 0.400 | 0.400 |
| kv1 | 35 | 0.914 | [0.829, 0.971] | 0.057 | 0.029 | 0.914 | 0.914 |
| cvP | 35 | 0.886 | [0.800, 0.971] | 0.086 | 0.029 | 0.886 | 0.886 |
| rnd1 | 35 | 0.000 | [0.000, 0.000] | 0.914 | 0.086 | 0.029 | 0.000 |
| rnd3 | 35 | 0.000 | [0.000, 0.000] | 0.971 | 0.029 | 0.000 | 0.000 |
| span1 | 35 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 0.971 | 1.000 |
| clean | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_full | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_t1 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_t5 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_t20 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_r5 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| nc_ds5 | 25 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_full | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_t1 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_t5 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_t20 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_r5 | 35 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| ns_ds5 | 25 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |

CA side by side (cv1, kv1, text): {"cv1": [0.8571, 0.7429, 0.9429, 35], "kv1": [0.9143, 0.8286, 0.9714, 35], "text": [0.9429, 0.8857, 1.0, 35]}
P3 CA(cv1)/CA(text): [0.9091, 0.8182, 0.9714, 35]; CA(kv1)/CA(text): [0.9697, 0.8571, 1.0938, 35]
P4 cv1-cv2: [0.2571, 0.1429, 0.3714, 35]; cv2-cv3: [0.2, 0.0571, 0.3429, 35]
P5 base CA: [0.0, 0.0, 0.0, 35]
Identity cvP = cv1 token-identical share: 0.857
Neuron: {"share_nc_full": [0.0, 0.0, 0.0, 35], "share_nc_t1": [0.0, 0.0, 0.0, 35], "share_nc_t5": [0.0, 0.0, 0.0, 35], "share_nc_t20": [0.0, 0.0, 0.0, 35], "share_nc_r5": [0.0, 0.0, 0.0, 35], "share_nc_ds5": [0.0, 0.0, 0.0, 25], "gap_nc_t5_r5": [0.0, 0.0, 0.0, 35], "share_ns_full": [0.0, 0.0, 0.0, 35], "share_ns_t1": [0.0, 0.0, 0.0, 35], "share_ns_t5": [0.0, 0.0, 0.0, 35], "share_ns_t20": [0.0, 0.0, 0.0, 35], "share_ns_r5": [0.0, 0.0, 0.0, 35], "share_ns_ds5": [0.0, 0.0, 0.0, 25], "gap_ns_t5_r5": [0.0, 0.0, 0.0, 35]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"div": 12, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"div": 16, "identical": false, "margin": 0.0, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"div": 36, "identical": false, "margin": 0.0, "pass": true}, {"div": 33, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "drops": {"ambiguous_operand": 1, "ambiguous_value": 1, "chain_arithmetic_error": 1, "clean_wrong": 1, "eligible": 35, "inconsistent": 3, "no_changed_step": 17, "no_equation": 8, "no_intermediate": 2, "one_step": 3, "token_count": 18}, "eligible_rate": 0.3889, "gate_80": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 819, "rule": "cap = min(512 - prefix, 2 x clean continuation after the prefix + 64) if under 1% of parsed rows pass it", "share": 0.0, "use_cap_in_full": true}, "n_lead_same_hist": {"0": 24, "1": 11}, "n_pool": 395, "norm_ratio_prerounding": [{"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}], "projected_eligible": 154, "projected_full_card_hours": 2.12, "r1_rule_kept_info": 29, "sec_clean_mean": 8.99, "sec_per_arm_item": 1.11, "sec_per_item_mean": 26.65, "split_n": 10}
