# RA-29 ouro14 hellaswag

Items: 2000 built, 70 tried, 30 kept; drops {"screen_A": 25, "screen_B": 11, "gap_below_2": 4}; new gold letter of the kept items {"a": 8, "b": 10, "c": 8, "d": 4, "e": 0}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.000 | [-0.005, 0.004] | 0.000 |
| blk_L1 | 0.999 | [0.992, 1.005] | 1.000 |
| lt_L2 | 0.032 | [0.021, 0.043] | 0.000 |
| lt_L3 | 0.250 | [0.218, 0.282] | 0.033 |
| lt_L4first | 0.247 | [0.217, 0.277] | 0.033 |
| lt_L4q1 | 0.251 | [0.219, 0.283] | 0.033 |
| lt_L4mid | 0.479 | [0.434, 0.525] | 0.333 |
| lt_L4q3 | 0.809 | [0.776, 0.842] | 0.967 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.955 | [0.939, 0.969] | 0.967 |
| blk_L3 | 0.742 | [0.719, 0.765] | 0.833 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.005 | [-0.002, 0.012] | 0.000 |
| rnd_lt_L4 | -0.006 | [-0.035, 0.021] | 0.000 |
| rnd_blk_L1 | 0.517 | [0.450, 0.587] | 0.300 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998907, 1.002189], "rnd_lt_L4": [0.99974, 1.000313], "rnd_blk_L1": [0.999983, 1.000027]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 8 | 0.007 | 0.994 | 0.275 | 0.281 | 0.498 | 0.773 |
| b | 10 | -0.006 | 1.010 | 0.218 | 0.222 | 0.406 | 0.822 |
| c | 8 | 0.001 | 0.989 | 0.296 | 0.299 | 0.496 | 0.841 |
| d | 4 | -0.003 | 1.000 | 0.185 | 0.165 | 0.592 | 0.786 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 1.632, "seconds_per_kept_item": 3.296, "seconds_per_arm_and_item": {"lt_L1": 0.1309, "blk_L1": 0.1381, "lt_L2": 0.1416, "lt_L3": 0.1338, "lt_L4first": 0.1338, "lt_L4q1": 0.1334, "lt_L4mid": 0.1347, "lt_L4q3": 0.1362, "lt_L4": 0.1362, "blk_L2": 0.143, "blk_L3": 0.1411, "blk_L4": 0.143, "persist_lt_L1to4": 0.1364, "rnd_lt_L1": 0.133, "rnd_lt_L4": 0.1332, "rnd_blk_L1": 0.1405, "pc": 0.1343}, "full_card_hours_at_this_rate": 0.906}
