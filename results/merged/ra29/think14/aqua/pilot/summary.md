# RA-29 think14 aqua

Items: 250 built, 120 tried, 20 kept; drops {"screen_A": 74, "gap_below_2": 3, "screen_B": 23}; new gold letter of the kept items {"a": 5, "b": 2, "c": 6, "d": 6, "e": 1}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.002 | [-0.009, 0.004] | 0.000 |
| blk_L1 | 1.002 | [0.996, 1.009] | 1.000 |
| lt_L2 | 0.012 | [0.002, 0.021] | 0.000 |
| lt_L3 | 0.184 | [0.139, 0.236] | 0.100 |
| lt_L4first | 0.183 | [0.141, 0.232] | 0.100 |
| lt_L4q1 | 0.180 | [0.135, 0.231] | 0.100 |
| lt_L4mid | 0.537 | [0.466, 0.608] | 0.350 |
| lt_L4q3 | 0.882 | [0.845, 0.920] | 0.950 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.997 | [0.988, 1.007] | 1.000 |
| blk_L3 | 0.755 | [0.698, 0.810] | 0.800 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | -0.002 | [-0.009, 0.004] | 0.000 |
| rnd_lt_L4 | -0.057 | [-0.098, -0.014] | 0.050 |
| rnd_blk_L1 | 0.358 | [0.268, 0.445] | 0.100 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [1.000226, 1.007423], "rnd_lt_L4": [0.999866, 1.000332], "rnd_blk_L1": [0.999977, 1.000116]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 5 | 0.000 | 1.001 | 0.298 | 0.307 | 0.680 | 0.932 |
| b | 2 | 0.009 | 1.031 | 0.221 | 0.212 | 0.289 | 0.761 |
| c | 6 | -0.008 | 1.001 | 0.137 | 0.135 | 0.434 | 0.857 |
| d | 6 | -0.002 | 0.995 | 0.126 | 0.112 | 0.552 | 0.893 |
| e | 1 | 0.000 | 0.995 | 0.170 | 0.160 | 0.851 | 0.957 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": true, "P1_lt_L1_below_0.2": true, "P1_holds": true, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 0.851, "seconds_per_kept_item": 3.236, "seconds_per_arm_and_item": {"lt_L1": 0.127, "blk_L1": 0.1434, "lt_L2": 0.1331, "lt_L3": 0.1317, "lt_L4first": 0.1294, "lt_L4q1": 0.1334, "lt_L4mid": 0.1308, "lt_L4q3": 0.1305, "lt_L4": 0.131, "blk_L2": 0.1349, "blk_L3": 0.1349, "blk_L4": 0.1356, "persist_lt_L1to4": 0.1321, "rnd_lt_L1": 0.1322, "rnd_lt_L4": 0.1311, "rnd_blk_L1": 0.1452, "pc": 0.1429}, "full_card_hours_at_this_rate": 0.059}
