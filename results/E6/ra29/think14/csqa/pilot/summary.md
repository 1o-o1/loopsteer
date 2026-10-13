# RA-29 think14 csqa

Items: 1215 built, 50 tried, 30 kept; drops {"screen_A": 16, "screen_B": 3, "gap_below_2": 1}; new gold letter of the kept items {"a": 5, "b": 6, "c": 7, "d": 6, "e": 6}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.002 | [-0.002, 0.006] | 0.000 |
| blk_L1 | 0.994 | [0.989, 0.999] | 1.000 |
| lt_L2 | 0.009 | [0.005, 0.012] | 0.000 |
| lt_L3 | 0.205 | [0.173, 0.238] | 0.033 |
| lt_L4first | 0.203 | [0.172, 0.236] | 0.033 |
| lt_L4q1 | 0.206 | [0.174, 0.241] | 0.033 |
| lt_L4mid | 0.425 | [0.375, 0.474] | 0.233 |
| lt_L4q3 | 0.766 | [0.741, 0.791] | 0.967 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.972 | [0.940, 0.996] | 1.000 |
| blk_L3 | 0.740 | [0.675, 0.795] | 0.900 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.003 | [-0.000, 0.006] | 0.000 |
| rnd_lt_L4 | 0.022 | [-0.002, 0.050] | 0.033 |
| rnd_blk_L1 | 0.412 | [0.319, 0.515] | 0.133 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998809, 1.005135], "rnd_lt_L4": [0.999791, 1.000284], "rnd_blk_L1": [0.999937, 1.000111]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 5 | -0.004 | 1.009 | 0.298 | 0.298 | 0.540 | 0.812 |
| b | 6 | -0.001 | 0.992 | 0.285 | 0.298 | 0.474 | 0.775 |
| c | 7 | 0.008 | 0.998 | 0.166 | 0.163 | 0.251 | 0.700 |
| d | 6 | 0.004 | 0.989 | 0.117 | 0.115 | 0.441 | 0.829 |
| e | 6 | 0.003 | 0.983 | 0.178 | 0.179 | 0.466 | 0.735 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 2.063, "seconds_per_kept_item": 3.193, "seconds_per_arm_and_item": {"lt_L1": 0.135, "blk_L1": 0.1343, "lt_L2": 0.1302, "lt_L3": 0.1301, "lt_L4first": 0.1289, "lt_L4q1": 0.1294, "lt_L4mid": 0.1303, "lt_L4q3": 0.1313, "lt_L4": 0.1304, "blk_L2": 0.1378, "blk_L3": 0.1361, "blk_L4": 0.1347, "persist_lt_L1to4": 0.1294, "rnd_lt_L1": 0.1316, "rnd_lt_L4": 0.1282, "rnd_blk_L1": 0.133, "pc": 0.1296}, "full_card_hours_at_this_rate": 0.696}
