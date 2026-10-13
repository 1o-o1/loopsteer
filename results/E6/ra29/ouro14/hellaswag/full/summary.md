# RA-29 ouro14 hellaswag

Items: 2000 built, 2000 tried, 934 kept; drops {"screen_A": 649, "screen_B": 296, "gap_below_2": 121}; new gold letter of the kept items {"a": 251, "b": 252, "c": 236, "d": 195, "e": 0}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.002 | [0.001, 0.003] | 0.000 |
| blk_L1 | 0.996 | [0.995, 0.998] | 0.994 |
| lt_L2 | 0.037 | [0.034, 0.039] | 0.004 |
| lt_L3 | 0.239 | [0.234, 0.245] | 0.075 |
| lt_L4first | 0.239 | [0.234, 0.244] | 0.073 |
| lt_L4q1 | 0.242 | [0.236, 0.248] | 0.075 |
| lt_L4mid | 0.497 | [0.490, 0.505] | 0.445 |
| lt_L4q3 | 0.799 | [0.794, 0.804] | 0.938 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.951 | [0.947, 0.955] | 0.965 |
| blk_L3 | 0.738 | [0.732, 0.744] | 0.875 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.000 | [-0.001, 0.002] | 0.000 |
| rnd_lt_L4 | 0.004 | [-0.002, 0.009] | 0.007 |
| rnd_blk_L1 | 0.488 | [0.474, 0.501] | 0.267 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998751, 1.003509], "rnd_lt_L4": [0.999267, 1.000808], "rnd_blk_L1": [0.999959, 1.000047]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 251 | -0.000 | 0.997 | 0.239 | 0.248 | 0.508 | 0.778 |
| b | 252 | 0.003 | 0.995 | 0.237 | 0.239 | 0.452 | 0.802 |
| c | 236 | 0.001 | 1.001 | 0.277 | 0.283 | 0.473 | 0.818 |
| d | 195 | 0.003 | 0.990 | 0.195 | 0.188 | 0.572 | 0.798 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 1.525, "seconds_per_kept_item": 2.84, "seconds_per_arm_and_item": {"lt_L1": 0.1269, "blk_L1": 0.1358, "lt_L2": 0.1301, "lt_L3": 0.1303, "lt_L4first": 0.1307, "lt_L4q1": 0.1301, "lt_L4mid": 0.1301, "lt_L4q3": 0.1303, "lt_L4": 0.1302, "blk_L2": 0.134, "blk_L3": 0.1344, "blk_L4": 0.1337, "persist_lt_L1to4": 0.1315, "rnd_lt_L1": 0.1301, "rnd_lt_L4": 0.1294, "rnd_blk_L1": 0.1355, "pc": 0.1306}, "full_card_hours_at_this_rate": 0.847}
