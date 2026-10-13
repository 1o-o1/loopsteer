# RA-30 E6/ra30/ouro_2_6b_base/pilot/gsm8k

Items 20 (split 0). Pool: Table 2 single-card a1 scored list. Neuron arms carry only the last layer's MLP part of the loop-1 state difference.

| arm | n | CA | 90% CI | A | break | carry B | CA (floor and clean end in A) |
|---|---|---|---|---|---|---|---|
| base | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| text | 20 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 1.000 | 1.000 |
| floor | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| cv1 | 20 | 0.850 | [0.700, 0.950] | 0.150 | 0.000 | 0.850 | 0.850 |
| cv2 | 20 | 0.550 | [0.350, 0.750] | 0.400 | 0.050 | 0.550 | 0.550 |
| cv3 | 20 | 0.400 | [0.200, 0.600] | 0.550 | 0.050 | 0.400 | 0.400 |
| kv1 | 20 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 1.000 | 1.000 |
| cvP | 20 | 0.850 | [0.700, 0.950] | 0.150 | 0.000 | 0.850 | 0.850 |
| rnd1 | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |
| rnd3 | 20 | 0.000 | [0.000, 0.000] | 0.950 | 0.050 | 0.000 | 0.000 |
| span1 | 20 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 1.000 | 1.000 |
| clean | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.000 |

CA side by side (cv1, kv1, text): {"cv1": [0.85, 0.7, 0.95, 20], "kv1": [1.0, 1.0, 1.0, 20], "text": [1.0, 1.0, 1.0, 20]}
P3 CA(cv1)/CA(text): [0.85, 0.7, 0.95, 20]; CA(kv1)/CA(text): [1.0, 1.0, 1.0, 20]
P4 cv1-cv2: [0.3, 0.15, 0.45, 20]; cv2-cv3: [0.15, 0.05, 0.3, 20]
P5 base CA: [0.0, 0.0, 0.0, 20]
Identity cvP = cv1 token-identical share: 0.950
Neuron: {}

Checks: {"G2b": [{"div": 8, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"div": 13, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"div": 8, "identical": false, "margin": 0.125, "pass": true}], "drops": {"ambiguous_operand": 1, "chain_arithmetic_error": 2, "eligible": 20, "inconsistent": 2, "no_changed_step": 10, "no_equation": 1, "token_count": 7}, "eligible_rate": 0.4651, "gate_80": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 240, "rule": "cap = min(512 - prefix, 2 x clean continuation after the prefix + 64) if under 1% of parsed rows pass it", "share": 0.0, "use_cap_in_full": true}, "n_lead_same_hist": {"0": 15, "1": 5}, "n_pool": 425, "norm_ratio_prerounding": [{"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}], "projected_eligible": 198, "projected_full_card_hours": 5.25, "r1_rule_kept_info": 17, "sec_clean_mean": 23.39, "sec_per_arm_item": 1.891, "sec_per_item_mean": 45.39, "split_n": 0}
