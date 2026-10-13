# RA-29 think26 csqa

Items: 1215 built, 46 tried, 30 kept; drops {"screen_A": 12, "gap_below_2": 1, "screen_B": 3}; new gold letter of the kept items {"a": 4, "b": 7, "c": 7, "d": 8, "e": 4}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.004 | [-0.009, 0.000] | 0.000 |
| blk_L1 | 1.011 | [1.000, 1.024] | 1.000 |
| lt_L2 | 0.028 | [0.018, 0.039] | 0.000 |
| lt_L3 | 0.209 | [0.173, 0.245] | 0.033 |
| lt_L4first | 0.209 | [0.175, 0.244] | 0.033 |
| lt_L4q1 | 0.312 | [0.270, 0.354] | 0.100 |
| lt_L4mid | 0.517 | [0.461, 0.569] | 0.367 |
| lt_L4q3 | 0.588 | [0.537, 0.637] | 0.533 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.968 | [0.950, 0.988] | 0.967 |
| blk_L3 | 0.780 | [0.747, 0.812] | 0.967 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | -0.002 | [-0.008, 0.005] | 0.000 |
| rnd_lt_L4 | 0.005 | [-0.027, 0.036] | 0.000 |
| rnd_blk_L1 | 0.458 | [0.386, 0.537] | 0.133 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.999083, 1.006781], "rnd_lt_L4": [0.999727, 1.000364], "rnd_blk_L1": [0.999937, 1.000057]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 4 | -0.004 | 0.993 | 0.368 | 0.461 | 0.615 | 0.652 |
| b | 7 | -0.001 | 0.996 | 0.224 | 0.317 | 0.539 | 0.602 |
| c | 7 | -0.005 | 1.007 | 0.162 | 0.241 | 0.470 | 0.520 |
| d | 8 | -0.005 | 1.004 | 0.169 | 0.327 | 0.584 | 0.654 |
| e | 4 | -0.007 | 1.075 | 0.188 | 0.252 | 0.329 | 0.489 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": true, "P1_lt_L1_below_0.2": true, "P1_holds": true, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 3.965, "seconds_per_kept_item": 5.7, "seconds_per_arm_and_item": {"lt_L1": 0.2403, "blk_L1": 0.253, "lt_L2": 0.2447, "lt_L3": 0.245, "lt_L4first": 0.247, "lt_L4q1": 0.2486, "lt_L4mid": 0.2492, "lt_L4q3": 0.2499, "lt_L4": 0.2479, "blk_L2": 0.2575, "blk_L3": 0.2674, "blk_L4": 0.2553, "persist_lt_L1to4": 0.2495, "rnd_lt_L1": 0.2488, "rnd_lt_L4": 0.2471, "rnd_blk_L1": 0.2521, "pc": 0.2466}, "full_card_hours_at_this_rate": 1.338}
