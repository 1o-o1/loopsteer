# RA-29 think26 arc

Items: 1172 built, 36 tried, 30 kept; drops {"screen_B": 2, "screen_A": 3, "gap_below_2": 1}; new gold letter of the kept items {"a": 8, "b": 8, "c": 7, "d": 7, "e": 0}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.001 | [-0.002, 0.004] | 0.000 |
| blk_L1 | 0.996 | [0.992, 0.999] | 1.000 |
| lt_L2 | 0.043 | [0.030, 0.056] | 0.000 |
| lt_L3 | 0.211 | [0.187, 0.236] | 0.033 |
| lt_L4first | 0.212 | [0.188, 0.237] | 0.033 |
| lt_L4q1 | 0.302 | [0.262, 0.345] | 0.100 |
| lt_L4mid | 0.485 | [0.438, 0.533] | 0.400 |
| lt_L4q3 | 0.567 | [0.516, 0.618] | 0.533 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.956 | [0.938, 0.975] | 1.000 |
| blk_L3 | 0.799 | [0.771, 0.826] | 0.900 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.005 | [0.000, 0.009] | 0.000 |
| rnd_lt_L4 | -0.029 | [-0.068, 0.010] | 0.000 |
| rnd_blk_L1 | 0.468 | [0.429, 0.509] | 0.233 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.99936, 1.00435], "rnd_lt_L4": [0.99985, 1.000342], "rnd_blk_L1": [0.99996, 1.00006]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 8 | -0.002 | 1.003 | 0.275 | 0.365 | 0.559 | 0.658 |
| b | 8 | -0.000 | 0.988 | 0.179 | 0.279 | 0.496 | 0.579 |
| c | 7 | 0.004 | 0.993 | 0.201 | 0.283 | 0.438 | 0.469 |
| d | 7 | 0.003 | 1.000 | 0.185 | 0.277 | 0.437 | 0.547 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 4.877, "seconds_per_kept_item": 5.711, "seconds_per_arm_and_item": {"lt_L1": 0.2448, "blk_L1": 0.2524, "lt_L2": 0.2438, "lt_L3": 0.2466, "lt_L4first": 0.2451, "lt_L4q1": 0.2481, "lt_L4mid": 0.2452, "lt_L4q3": 0.2468, "lt_L4": 0.2477, "blk_L2": 0.26, "blk_L3": 0.2532, "blk_L4": 0.2558, "persist_lt_L1to4": 0.2483, "rnd_lt_L1": 0.2492, "rnd_lt_L4": 0.2508, "rnd_blk_L1": 0.257, "pc": 0.2461}, "full_card_hours_at_this_rate": 1.588}
