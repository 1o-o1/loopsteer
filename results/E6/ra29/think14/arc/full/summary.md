# RA-29 think14 arc

Items: 1172 built, 1172 tried, 935 kept; drops {"screen_A": 149, "screen_B": 62, "gap_below_2": 26}; new gold letter of the kept items {"a": 223, "b": 237, "c": 239, "d": 233, "e": 3}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.001 | [-0.000, 0.002] | 0.000 |
| blk_L1 | 1.000 | [0.999, 1.001] | 0.999 |
| lt_L2 | 0.006 | [0.004, 0.007] | 0.001 |
| lt_L3 | 0.245 | [0.240, 0.251] | 0.074 |
| lt_L4first | 0.245 | [0.239, 0.251] | 0.070 |
| lt_L4q1 | 0.245 | [0.240, 0.251] | 0.068 |
| lt_L4mid | 0.490 | [0.481, 0.498] | 0.448 |
| lt_L4q3 | 0.807 | [0.803, 0.812] | 0.982 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.988 | [0.986, 0.991] | 0.995 |
| blk_L3 | 0.709 | [0.702, 0.716] | 0.865 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.001 | [-0.000, 0.002] | 0.001 |
| rnd_lt_L4 | -0.004 | [-0.010, 0.001] | 0.003 |
| rnd_blk_L1 | 0.406 | [0.394, 0.419] | 0.202 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.997017, 1.00671], "rnd_lt_L4": [0.99945, 1.000581], "rnd_blk_L1": [0.999841, 1.000131]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 223 | 0.001 | 1.000 | 0.305 | 0.312 | 0.597 | 0.852 |
| b | 237 | 0.001 | 0.999 | 0.236 | 0.235 | 0.418 | 0.761 |
| c | 239 | -0.000 | 0.999 | 0.247 | 0.243 | 0.425 | 0.801 |
| d | 233 | 0.001 | 1.002 | 0.199 | 0.197 | 0.525 | 0.819 |
| e | 3 | -0.009 | 0.980 | 0.056 | 0.062 | 0.573 | 0.785 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 2.238, "seconds_per_kept_item": 2.714, "seconds_per_arm_and_item": {"lt_L1": 0.1209, "blk_L1": 0.1286, "lt_L2": 0.1247, "lt_L3": 0.1246, "lt_L4first": 0.1246, "lt_L4q1": 0.1245, "lt_L4mid": 0.1244, "lt_L4q3": 0.1245, "lt_L4": 0.1243, "blk_L2": 0.1282, "blk_L3": 0.1283, "blk_L4": 0.1281, "persist_lt_L1to4": 0.126, "rnd_lt_L1": 0.1243, "rnd_lt_L4": 0.1243, "rnd_blk_L1": 0.1289, "pc": 0.125}, "full_card_hours_at_this_rate": 0.728}
