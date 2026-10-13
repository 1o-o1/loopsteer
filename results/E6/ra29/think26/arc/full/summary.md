# RA-29 think26 arc

Items: 1172 built, 1172 tried, 1021 kept; drops {"screen_B": 34, "screen_A": 102, "gap_below_2": 15}; new gold letter of the kept items {"a": 250, "b": 263, "c": 257, "d": 249, "e": 2}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.002 | [0.001, 0.003] | 0.000 |
| blk_L1 | 0.998 | [0.998, 0.999] | 1.000 |
| lt_L2 | 0.049 | [0.046, 0.051] | 0.001 |
| lt_L3 | 0.196 | [0.192, 0.201] | 0.025 |
| lt_L4first | 0.196 | [0.192, 0.201] | 0.027 |
| lt_L4q1 | 0.298 | [0.291, 0.305] | 0.118 |
| lt_L4mid | 0.483 | [0.475, 0.492] | 0.417 |
| lt_L4q3 | 0.569 | [0.561, 0.578] | 0.637 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.949 | [0.945, 0.952] | 0.993 |
| blk_L3 | 0.788 | [0.782, 0.794] | 0.933 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.005 | [0.003, 0.007] | 0.000 |
| rnd_lt_L4 | -0.000 | [-0.005, 0.005] | 0.001 |
| rnd_blk_L1 | 0.449 | [0.440, 0.458] | 0.252 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.99739, 1.007961], "rnd_lt_L4": [0.999403, 1.000642], "rnd_blk_L1": [0.999901, 1.000113]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 250 | 0.002 | 0.998 | 0.267 | 0.349 | 0.521 | 0.630 |
| b | 263 | -0.000 | 0.998 | 0.175 | 0.276 | 0.469 | 0.539 |
| c | 257 | 0.002 | 0.995 | 0.178 | 0.265 | 0.468 | 0.537 |
| d | 249 | 0.004 | 1.002 | 0.166 | 0.301 | 0.475 | 0.573 |
| e | 2 | -0.004 | 0.996 | 0.182 | 0.440 | 0.556 | 0.656 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 4.569, "seconds_per_kept_item": 5.141, "seconds_per_arm_and_item": {"lt_L1": 0.2385, "blk_L1": 0.2452, "lt_L2": 0.2412, "lt_L3": 0.2417, "lt_L4first": 0.241, "lt_L4q1": 0.2415, "lt_L4mid": 0.2424, "lt_L4q3": 0.2417, "lt_L4": 0.2418, "blk_L2": 0.2446, "blk_L3": 0.2453, "blk_L4": 0.2443, "persist_lt_L1to4": 0.243, "rnd_lt_L1": 0.2425, "rnd_lt_L4": 0.2411, "rnd_blk_L1": 0.2447, "pc": 0.242}, "full_card_hours_at_this_rate": 1.488}
