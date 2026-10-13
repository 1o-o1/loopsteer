# RA-29 think14 arc

Items: 1172 built, 40 tried, 30 kept; drops {"screen_A": 7, "screen_B": 2, "gap_below_2": 1}; new gold letter of the kept items {"a": 8, "b": 7, "c": 7, "d": 8, "e": 0}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.005 | [0.001, 0.011] | 0.000 |
| blk_L1 | 0.998 | [0.992, 1.003] | 1.000 |
| lt_L2 | 0.007 | [-0.000, 0.014] | 0.000 |
| lt_L3 | 0.257 | [0.227, 0.285] | 0.000 |
| lt_L4first | 0.256 | [0.225, 0.285] | 0.000 |
| lt_L4q1 | 0.265 | [0.234, 0.294] | 0.000 |
| lt_L4mid | 0.515 | [0.471, 0.559] | 0.533 |
| lt_L4q3 | 0.816 | [0.793, 0.839] | 0.967 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.992 | [0.982, 1.002] | 1.000 |
| blk_L3 | 0.735 | [0.702, 0.769] | 0.900 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.001 | [-0.005, 0.007] | 0.000 |
| rnd_lt_L4 | 0.007 | [-0.029, 0.041] | 0.000 |
| rnd_blk_L1 | 0.421 | [0.360, 0.485] | 0.267 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998971, 1.003789], "rnd_lt_L4": [0.999501, 1.000378], "rnd_blk_L1": [0.999915, 1.000023]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 8 | 0.008 | 0.998 | 0.312 | 0.332 | 0.664 | 0.853 |
| b | 7 | 0.007 | 1.009 | 0.285 | 0.281 | 0.417 | 0.765 |
| c | 7 | 0.009 | 0.983 | 0.219 | 0.225 | 0.420 | 0.808 |
| d | 8 | -0.002 | 1.000 | 0.210 | 0.217 | 0.535 | 0.831 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": true, "P1_lt_L1_below_0.2": true, "P1_holds": true, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 2.391, "seconds_per_kept_item": 3.061, "seconds_per_arm_and_item": {"lt_L1": 0.1245, "blk_L1": 0.1344, "lt_L2": 0.1332, "lt_L3": 0.1295, "lt_L4first": 0.1286, "lt_L4q1": 0.1284, "lt_L4mid": 0.13, "lt_L4q3": 0.1285, "lt_L4": 0.1291, "blk_L2": 0.1339, "blk_L3": 0.1337, "blk_L4": 0.1326, "persist_lt_L1to4": 0.13, "rnd_lt_L1": 0.1291, "rnd_lt_L4": 0.1316, "rnd_blk_L1": 0.1338, "pc": 0.1292}, "full_card_hours_at_this_rate": 0.778}
