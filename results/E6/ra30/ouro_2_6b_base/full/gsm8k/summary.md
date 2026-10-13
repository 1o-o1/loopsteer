# RA-30 E6/ra30/ouro_2_6b_base/full/gsm8k

Items 173 (split 0). Pool: Table 2 single-card a1 scored list. Neuron arms carry only the last layer's MLP part of the loop-1 state difference.

| arm | n | CA | 90% CI | A | break | carry B | CA (floor and clean end in A) |
|---|---|---|---|---|---|---|---|
| base | 173 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.052 | 0.000 |
| text | 173 | 0.936 | [0.907, 0.965] | 0.029 | 0.035 | 0.960 | 0.936 |
| floor | 173 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.041 | 0.000 |
| cv1 | 173 | 0.803 | [0.751, 0.850] | 0.156 | 0.041 | 0.832 | 0.803 |
| cv2 | 173 | 0.514 | [0.451, 0.578] | 0.428 | 0.058 | 0.561 | 0.514 |
| cv3 | 173 | 0.324 | [0.266, 0.382] | 0.584 | 0.092 | 0.370 | 0.324 |
| kv1 | 173 | 0.948 | [0.919, 0.977] | 0.023 | 0.029 | 0.971 | 0.948 |
| cvP | 173 | 0.803 | [0.751, 0.850] | 0.162 | 0.035 | 0.832 | 0.803 |
| rnd1 | 173 | 0.000 | [0.000, 0.000] | 0.954 | 0.046 | 0.052 | 0.000 |
| rnd3 | 173 | 0.000 | [0.000, 0.000] | 0.983 | 0.017 | 0.041 | 0.000 |
| span1 | 173 | 0.988 | [0.971, 1.000] | 0.000 | 0.012 | 0.960 | 0.988 |
| clean | 173 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.052 | 0.000 |

CA side by side (cv1, kv1, text): {"cv1": [0.8035, 0.7514, 0.8497, 173], "kv1": [0.948, 0.9191, 0.9769, 173], "text": [0.9364, 0.9075, 0.9653, 173]}
P3 CA(cv1)/CA(text): [0.858, 0.8121, 0.9012, 173]; CA(kv1)/CA(text): [1.0123, 1.0, 1.0272, 173]
P4 cv1-cv2: [0.289, 0.2312, 0.3468, 173]; cv2-cv3: [0.1908, 0.1329, 0.2486, 173]
P5 base CA: [0.0, 0.0, 0.0, 173]
Identity cvP = cv1 token-identical share: 0.873
Neuron: {}
Neuron hook check (neuron arms not run in the mains; their pilot CA 0.000 is not a result): nc_full: no effect of the last-layer MLP part (|delta a| 5.17364, logits equal False, max |dz| 0.1875); ns_full: no effect of the last-layer MLP part (|delta a| 2.954675, logits equal False, max |dz| 0.1875)

Checks: {"G2b": [{"div": 8, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"div": 13, "identical": false, "margin": 0.125, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "drops": {"ambiguous_operand": 3, "ambiguous_value": 2, "chain_arithmetic_error": 8, "clean_wrong": 4, "derived_B_format": 2, "eligible": 173, "inconsistent": 18, "no_changed_step": 94, "no_equation": 5, "no_intermediate": 13, "one_step": 23, "token_count": 80}, "eligible_rate": 0.4071, "gate_80": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 2074, "rule": "cap = min(512 - prefix, 2 x clean continuation after the prefix + 64) if under 1% of parsed rows pass it", "share": 0.0, "use_cap_in_full": true}, "n_lead_same_hist": {"0": 125, "1": 46, "2": 2}, "n_pool": 425, "norm_ratio_prerounding": [{"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}, {"rnd1": 1.0, "rnd3": 1.0}], "projected_eligible": 173, "projected_full_card_hours": 4.87, "r1_rule_kept_info": 131, "sec_clean_mean": 23.67, "sec_per_arm_item": 1.802, "sec_per_item_mean": 43.25, "split_n": 0}
