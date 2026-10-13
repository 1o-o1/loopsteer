# RA-30 McLeish r32 gsm8k

Breadth cell (RA D1: the 80-item gate is waived). Items n = 22. Pool: the steer_gen_raven McLeish GSM8K scored list (237). K = 32.

| arm | n | CA | 90% CI | A | break | carry B |
|---|---|---|---|---|---|---|
| base | 22 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 |
| text | 22 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 1.000 |
| floor | 22 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 |
| cv1 | 22 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 |
| kv1 | 22 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 |
| rnd1 | 22 | 0.000 | [0.000, 0.000] | 0.955 | 0.045 | 0.000 |
| cvP | 22 | 0.864 | [0.727, 0.955] | 0.136 | 0.000 | 0.909 |
| kvP | 22 | 0.045 | [0.000, 0.136] | 0.909 | 0.045 | 0.045 |
| eP | 22 | 0.955 | [0.864, 1.000] | 0.045 | 0.000 | 1.000 |
| span1 | 22 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | 0.000 |
| clean | 22 | 0.000 | [0.000, 0.000] | 0.955 | 0.045 | 0.000 |

Predictions (about 0: CA <= 0.05 and 90% upper bound <= 0.10; most: CA(cvP)/CA(text) >= 0.67 and 90% lower bound >= 0.50):
{"thresholds": {"about0": {"point": 0.05, "hi90": 0.1}, "most": {"ratio": 0.67, "lo90": 0.5}}, "cv1_about0": {"CA": [0.0, 0.0, 0.0, 22], "holds": true}, "kv1_about0": {"CA": [0.0, 0.0, 0.0, 22], "holds": true}, "cvP_most": {"ratio": [0.8636, 0.7273, 0.9545, 22], "holds": true}, "kvP_over_text": [0.0455, 0.0, 0.1364, 22], "eP_over_text": [0.9545, 0.8636, 1.0, 22], "cvP_over_eP": [0.9048, 0.7895, 1.0, 22], "base_CA": [0.0, 0.0, 0.0, 22]}

Checks: {"G2b": [{"identical": true, "pass": true}, {"identical": true, "pass": true}, {"identical": true, "pass": true}], "breadth_cell": "McLeish r32: the 80-item gate is waived (RA D1); n is printed with every interval", "drops": {"eligible": 22, "inconsistent": 4, "no_changed_step": 17, "no_equation": 8, "no_intermediate": 2, "one_step": 10, "token_count": 14, "value_is_operand": 2}, "eligible_rate": 0.2785, "gate_80": "waived (RA D1: the full run needs n >= 20 projected eligible)", "gate_n20": true, "horizon_cap_rule": {"rows_marker_after_cap": 0, "rows_with_answer": 242, "share": 0.0, "use_cap_in_full": true}, "n_pool": 237, "norm_ratio_prerounding": [{"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}, {"rnd1": 1.0}], "projected_eligible": 66, "projected_full_card_hours": 1.4, "r1_rule_kept_info": 22, "sec_clean_mean": 13.89, "sec_per_arm_item": 2.431, "sec_per_item_mean": 26.74, "shards_merged": [], "shards_rule": {"projected_full_card_hours": 1.4, "rule": "3 if <= 16.2 card-hours, else ceil(projected / 5.4)", "shards": 3}}
