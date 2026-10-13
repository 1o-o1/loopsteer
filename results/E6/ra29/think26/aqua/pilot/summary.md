# RA-29 think26 aqua

Items: 250 built, 105 tried, 30 kept; drops {"screen_A": 50, "gap_below_2": 9, "screen_B": 16}; new gold letter of the kept items {"a": 6, "b": 1, "c": 11, "d": 8, "e": 4}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.001 | [-0.004, 0.003] | 0.000 |
| blk_L1 | 1.004 | [0.999, 1.009] | 1.000 |
| lt_L2 | 0.056 | [0.042, 0.070] | 0.000 |
| lt_L3 | 0.207 | [0.173, 0.243] | 0.067 |
| lt_L4first | 0.210 | [0.176, 0.246] | 0.067 |
| lt_L4q1 | 0.299 | [0.261, 0.340] | 0.067 |
| lt_L4mid | 0.471 | [0.425, 0.519] | 0.400 |
| lt_L4q3 | 0.648 | [0.595, 0.705] | 0.667 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.956 | [0.943, 0.969] | 1.000 |
| blk_L3 | 0.808 | [0.781, 0.833] | 0.900 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.003 | [-0.005, 0.010] | 0.000 |
| rnd_lt_L4 | -0.036 | [-0.074, 0.000] | 0.000 |
| rnd_blk_L1 | 0.381 | [0.296, 0.458] | 0.133 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998984, 1.005082], "rnd_lt_L4": [0.99947, 1.000305], "rnd_blk_L1": [0.999963, 1.000133]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 6 | -0.013 | 1.003 | 0.339 | 0.460 | 0.602 | 0.748 |
| b | 1 | -0.016 | 1.008 | 0.115 | 0.131 | 0.328 | 0.410 |
| c | 11 | -0.000 | 1.000 | 0.209 | 0.278 | 0.447 | 0.585 |
| d | 8 | 0.003 | 1.000 | 0.152 | 0.215 | 0.382 | 0.586 |
| e | 4 | 0.014 | 1.019 | 0.138 | 0.327 | 0.555 | 0.855 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 2.156, "seconds_per_kept_item": 5.74, "seconds_per_arm_and_item": {"lt_L1": 0.2435, "blk_L1": 0.2597, "lt_L2": 0.2562, "lt_L3": 0.248, "lt_L4first": 0.2497, "lt_L4q1": 0.2489, "lt_L4mid": 0.2487, "lt_L4q3": 0.2486, "lt_L4": 0.2498, "blk_L2": 0.261, "blk_L3": 0.2604, "blk_L4": 0.2597, "persist_lt_L1to4": 0.2518, "rnd_lt_L1": 0.2507, "rnd_lt_L4": 0.2512, "rnd_blk_L1": 0.2688, "pc": 0.2517}, "full_card_hours_at_this_rate": 0.15}
