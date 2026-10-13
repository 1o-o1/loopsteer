# RA-29 think14 csqa

Items: 1215 built, 1215 tried, 807 kept; drops {"screen_A": 318, "screen_B": 73, "gap_below_2": 17}; new gold letter of the kept items {"a": 159, "b": 170, "c": 165, "d": 159, "e": 154}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.001 | [0.000, 0.002] | 0.000 |
| blk_L1 | 1.000 | [0.999, 1.000] | 0.996 |
| lt_L2 | 0.003 | [0.002, 0.005] | 0.000 |
| lt_L3 | 0.214 | [0.206, 0.222] | 0.029 |
| lt_L4first | 0.214 | [0.206, 0.222] | 0.029 |
| lt_L4q1 | 0.215 | [0.207, 0.223] | 0.029 |
| lt_L4mid | 0.426 | [0.416, 0.436] | 0.336 |
| lt_L4q3 | 0.765 | [0.760, 0.770] | 0.949 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.990 | [0.988, 0.993] | 0.991 |
| blk_L3 | 0.743 | [0.734, 0.752] | 0.916 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.001 | [0.000, 0.002] | 0.000 |
| rnd_lt_L4 | 0.002 | [-0.005, 0.008] | 0.007 |
| rnd_blk_L1 | 0.390 | [0.375, 0.405] | 0.160 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998006, 1.009432], "rnd_lt_L4": [0.999437, 1.000771], "rnd_blk_L1": [0.999884, 1.000151]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 159 | 0.000 | 1.000 | 0.353 | 0.358 | 0.588 | 0.818 |
| b | 170 | 0.001 | 1.000 | 0.257 | 0.261 | 0.416 | 0.777 |
| c | 165 | 0.002 | 1.000 | 0.152 | 0.148 | 0.278 | 0.732 |
| d | 159 | 0.002 | 0.999 | 0.159 | 0.158 | 0.345 | 0.759 |
| e | 154 | 0.000 | 0.998 | 0.145 | 0.147 | 0.513 | 0.740 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 1.96, "seconds_per_kept_item": 2.767, "seconds_per_arm_and_item": {"lt_L1": 0.1233, "blk_L1": 0.1311, "lt_L2": 0.1268, "lt_L3": 0.1267, "lt_L4first": 0.1268, "lt_L4q1": 0.1267, "lt_L4mid": 0.1269, "lt_L4q3": 0.1278, "lt_L4": 0.1268, "blk_L2": 0.1307, "blk_L3": 0.1319, "blk_L4": 0.1301, "persist_lt_L1to4": 0.1281, "rnd_lt_L1": 0.1272, "rnd_lt_L4": 0.1265, "rnd_blk_L1": 0.1309, "pc": 0.1288}, "full_card_hours_at_this_rate": 0.662}
