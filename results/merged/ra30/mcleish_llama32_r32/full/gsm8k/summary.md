# RA-30 McLeish r32 full

Breadth cell (RA D1: the 80-item gate is waived). Items n = 75, merged from 3 shards. Pool: the steer_gen_raven McLeish GSM8K scored list (237). K = 32.

| arm | n | CA | 90% CI | A | break | carry B |
|---|---|---|---|---|---|---|
| base | 75 | 0.000 | [0.000, 0.000] | 0.973 | 0.027 | 0.053 |
| text | 75 | 0.947 | [0.907, 0.987] | 0.027 | 0.027 | 0.973 |
| floor | 75 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.040 |
| cv1 | 75 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.040 |
| kv1 | 75 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.040 |
| rnd1 | 75 | 0.000 | [0.000, 0.000] | 0.987 | 0.013 | 0.040 |
| cvP | 75 | 0.893 | [0.840, 0.947] | 0.067 | 0.040 | 0.933 |
| kvP | 75 | 0.013 | [0.000, 0.040] | 0.933 | 0.053 | 0.053 |
| eP | 75 | 0.933 | [0.880, 0.973] | 0.027 | 0.040 | 0.973 |
| span1 | 75 | 0.000 | [0.000, 0.000] | 0.987 | 0.013 | 0.053 |
| clean | 75 | 0.000 | [0.000, 0.000] | 0.973 | 0.027 | 0.040 |

Predictions (about 0: CA <= 0.05 and 90% upper bound <= 0.10; most: CA(cvP)/CA(text) >= 0.67 and 90% lower bound >= 0.50):
{"thresholds": {"about0": {"point": 0.05, "hi90": 0.1}, "most": {"ratio": 0.67, "lo90": 0.5}}, "cv1_about0": {"CA": [0.0, 0.0, 0.0, 75], "holds": true}, "kv1_about0": {"CA": [0.0, 0.0, 0.0, 75], "holds": true}, "cvP_most": {"ratio": [0.9437, 0.8767, 1.0, 75], "holds": true}, "kvP_over_text": [0.0141, 0.0, 0.0423, 75], "eP_over_text": [0.9859, 0.9444, 1.0282, 75], "cvP_over_eP": [0.9571, 0.9028, 1.0, 75], "base_CA": [0.0, 0.0, 0.0, 75]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "breadth_cell": "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval", "drops": {"ambiguous_value": 1, "chain_arithmetic_error": 7, "eligible": 75, "expr_parse": 1, "inconsistent": 10, "last_result_not_answer": 1, "no_changed_step": 49, "no_equation": 21, "no_intermediate": 5, "one_step": 29, "token_count": 36, "value_is_operand": 2}, "eligible_rate": 0.3165, "gate_80": "waived (RA D1: the full run needs n >= 20 projected eligible)", "gate_n20": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 822, "share": 0.0, "use_cap_in_full": true}, "n_pool": 237, "norm_ratio_prerounding": [{"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}], "projected_eligible": 75, "projected_full_card_hours": 1.6, "r1_rule_kept_info": 59, "sec_clean_mean": 14.86, "sec_per_arm_item": 2.691, "sec_per_item_mean": 29.61, "shards_merged": ["shard_0_of_3", "shard_1_of_3", "shard_2_of_3"], "shards_rule": {"projected_full_card_hours": 1.6, "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)", "shards": 3}}
