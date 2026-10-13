# RA-30 McLeish r32 gsm8k

Breadth cell (RA D1: the 80-item gate is waived). Items n = 29. Pool: the steer_gen_raven McLeish GSM8K scored list (237). K = 32.

| arm | n | CA | 90% CI | A | break | carry B |
|---|---|---|---|---|---|---|
| base | 29 | 0.000 | [0.000, 0.000] | 0.966 | 0.035 | 0.103 |
| text | 29 | 0.966 | [0.897, 1.000] | 0.000 | 0.035 | 1.000 |
| floor | 29 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.069 |
| cv1 | 29 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.069 |
| kv1 | 29 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.069 |
| rnd1 | 29 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.069 |
| cvP | 29 | 0.931 | [0.862, 1.000] | 0.035 | 0.035 | 0.966 |
| kvP | 29 | 0.000 | [0.000, 0.000] | 0.966 | 0.035 | 0.069 |
| eP | 29 | 0.966 | [0.897, 1.000] | 0.000 | 0.035 | 1.000 |
| span1 | 29 | 0.000 | [0.000, 0.000] | 0.966 | 0.035 | 0.103 |
| clean | 29 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.069 |

Predictions (about 0: CA <= 0.05 and 90% upper bound <= 0.10; most: CA(cvP)/CA(text) >= 0.67 and 90% lower bound >= 0.50):
{"thresholds": {"about0": {"point": 0.05, "hi90": 0.1}, "most": {"ratio": 0.67, "lo90": 0.5}}, "cv1_about0": {"CA": [0.0, 0.0, 0.0, 29], "holds": true}, "kv1_about0": {"CA": [0.0, 0.0, 0.0, 29], "holds": true}, "cvP_most": {"ratio": [0.9643, 0.8621, 1.0741, 29], "holds": true}, "kvP_over_text": [0.0, 0.0, 0.0, 29], "eP_over_text": [1.0, 0.9286, 1.0769, 29], "cvP_over_eP": [0.9643, 0.8929, 1.0, 29], "base_CA": [0.0, 0.0, 0.0, 29]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "breadth_cell": "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval", "drops": {"ambiguous_value": 1, "chain_arithmetic_error": 3, "eligible": 29, "expr_parse": 1, "inconsistent": 4, "no_changed_step": 17, "no_equation": 6, "no_intermediate": 2, "one_step": 10, "token_count": 6}, "eligible_rate": 0.3671, "gate_80": "waived (RA D1: the full run needs n >= 20 projected eligible)", "gate_n20": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 319, "share": 0.0, "use_cap_in_full": true}, "n_pool": 237, "norm_ratio_prerounding": [{"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}], "projected_eligible": 87, "projected_full_card_hours": 1.74, "r1_rule_kept_info": 19, "sec_clean_mean": 15.6, "sec_per_arm_item": 2.692, "sec_per_item_mean": 29.61, "shards_merged": [], "shards_rule": {"projected_full_card_hours": 1.74, "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)", "shards": 3}}
