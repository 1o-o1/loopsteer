# RA-30 McLeish r32 pilot

Breadth cell (RA D1: the 80-item gate is waived). Items n = 20. Pool: the steer_gen_raven McLeish GSM8K scored list (237). K = 32.

| arm | n | CA | 90% CI | A | break | carry B |
|---|---|---|---|---|---|---|
| base | 20 | 0.000 | [0.000, 0.000] | 0.900 | 0.100 | 0.100 |
| text | 20 | 0.950 | [0.850, 1.000] | 0.000 | 0.050 | 1.000 |
| floor | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.050 |
| cv1 | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.050 |
| kv1 | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.050 |
| rnd1 | 20 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.050 |
| cvP | 20 | 0.900 | [0.800, 1.000] | 0.050 | 0.050 | 0.950 |
| kvP | 20 | 0.000 | [0.000, 0.000] | 0.900 | 0.100 | 0.050 |
| eP | 20 | 0.950 | [0.850, 1.000] | 0.000 | 0.050 | 1.000 |
| span1 | 20 | 0.000 | [0.000, 0.000] | 0.950 | 0.050 | 0.100 |
| clean | 20 | 0.000 | [0.000, 0.000] | 0.950 | 0.050 | 0.050 |

Predictions (about 0: CA <= 0.05 and 90% upper bound <= 0.10; most: CA(cvP)/CA(text) >= 0.67 and 90% lower bound >= 0.50):
{"thresholds": {"about0": {"point": 0.05, "hi90": 0.1}, "most": {"ratio": 0.67, "lo90": 0.5}}, "cv1_about0": {"CA": [0.0, 0.0, 0.0, 20], "holds": true}, "kv1_about0": {"CA": [0.0, 0.0, 0.0, 20], "holds": true}, "cvP_most": {"ratio": [0.9474, 0.8, 1.1111, 20], "holds": true}, "kvP_over_text": [0.0, 0.0, 0.0, 20], "eP_over_text": [1.0, 0.8947, 1.1176, 20], "cvP_over_eP": [0.9474, 0.85, 1.0, 20], "base_CA": [0.0, 0.0, 0.0, 20]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "breadth_cell": "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval", "drops": {"ambiguous_value": 1, "chain_arithmetic_error": 3, "eligible": 20, "expr_parse": 1, "inconsistent": 2, "no_changed_step": 16, "no_equation": 5, "no_intermediate": 1, "one_step": 10, "token_count": 10, "value_is_operand": 1}, "eligible_rate": 0.2857, "gate_80": "waived (RA D1: the full run needs n >= 20 projected eligible)", "gate_n20": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 217, "share": 0.0, "use_cap_in_full": true}, "n_pool": 237, "norm_ratio_prerounding": [{"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}], "projected_eligible": 68, "projected_full_card_hours": 1.65, "r1_rule_kept_info": 15, "sec_clean_mean": 14.33, "sec_per_arm_item": 3.431, "sec_per_item_mean": 37.75, "shards_merged": [], "shards_rule": {"projected_full_card_hours": 1.65, "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)", "shards": 3}}
