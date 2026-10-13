# RA-30 McLeish r32 gsm8k

Breadth cell (RA D1: the 80-item gate is waived). Items n = 24. Pool: the steer_gen_raven McLeish GSM8K scored list (237). K = 32.

| arm | n | CA | 90% CI | A | break | carry B |
|---|---|---|---|---|---|---|
| base | 24 | 0.000 | [0.000, 0.000] | 0.958 | 0.042 | 0.042 |
| text | 24 | 0.875 | [0.750, 0.958] | 0.083 | 0.042 | 0.917 |
| floor | 24 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.042 |
| cv1 | 24 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.042 |
| kv1 | 24 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.042 |
| rnd1 | 24 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.042 |
| cvP | 24 | 0.875 | [0.750, 0.958] | 0.042 | 0.083 | 0.917 |
| kvP | 24 | 0.000 | [0.000, 0.000] | 0.917 | 0.083 | 0.042 |
| eP | 24 | 0.875 | [0.750, 0.958] | 0.042 | 0.083 | 0.917 |
| span1 | 24 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.042 |
| clean | 24 | 0.000 | [0.000, 0.000] | 0.958 | 0.042 | 0.042 |

Predictions (about 0: CA <= 0.05 and 90% upper bound <= 0.10; most: CA(cvP)/CA(text) >= 0.67 and 90% lower bound >= 0.50):
{"thresholds": {"about0": {"point": 0.05, "hi90": 0.1}, "most": {"ratio": 0.67, "lo90": 0.5}}, "cv1_about0": {"CA": [0.0, 0.0, 0.0, 24], "holds": true}, "kv1_about0": {"CA": [0.0, 0.0, 0.0, 24], "holds": true}, "cvP_most": {"ratio": [1.0, 0.9, 1.1111, 24], "holds": true}, "kvP_over_text": [0.0, 0.0, 0.0, 24], "eP_over_text": [1.0, 1.0, 1.0, 24], "cvP_over_eP": [1.0, 0.9, 1.1111, 24], "base_CA": [0.0, 0.0, 0.0, 24]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "breadth_cell": "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval", "drops": {"chain_arithmetic_error": 4, "eligible": 24, "inconsistent": 2, "last_result_not_answer": 1, "no_changed_step": 15, "no_equation": 7, "no_intermediate": 1, "one_step": 9, "token_count": 16}, "eligible_rate": 0.3038, "gate_80": "waived (RA D1: the full run needs n >= 20 projected eligible)", "gate_n20": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 261, "share": 0.0, "use_cap_in_full": true}, "n_pool": 237, "norm_ratio_prerounding": [{"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}], "projected_eligible": 72, "projected_full_card_hours": 1.64, "r1_rule_kept_info": 18, "sec_clean_mean": 15.08, "sec_per_arm_item": 2.929, "sec_per_item_mean": 32.22, "shards_merged": [], "shards_rule": {"projected_full_card_hours": 1.64, "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)", "shards": 3}}
