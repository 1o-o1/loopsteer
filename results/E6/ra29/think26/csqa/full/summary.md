# RA-29 think26 csqa

Items: 1215 built, 1215 tried, 857 kept; drops {"screen_A": 252, "gap_below_2": 21, "screen_B": 85}; new gold letter of the kept items {"a": 179, "b": 173, "c": 171, "d": 167, "e": 167}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.001 | [0.000, 0.002] | 0.000 |
| blk_L1 | 1.000 | [0.999, 1.002] | 0.999 |
| lt_L2 | 0.044 | [0.042, 0.047] | 0.001 |
| lt_L3 | 0.209 | [0.204, 0.214] | 0.023 |
| lt_L4first | 0.208 | [0.203, 0.213] | 0.020 |
| lt_L4q1 | 0.298 | [0.292, 0.305] | 0.100 |
| lt_L4mid | 0.466 | [0.457, 0.474] | 0.344 |
| lt_L4q3 | 0.547 | [0.539, 0.556] | 0.511 |
| lt_L4 | 1.000 | [1.000, 1.000] | 1.000 |
| blk_L2 | 0.952 | [0.949, 0.956] | 0.988 |
| blk_L3 | 0.772 | [0.766, 0.779] | 0.909 |
| blk_L4 | 0.000 | [0.000, 0.000] | 0.000 |
| persist_lt_L1to4 | 1.000 | [1.000, 1.000] | 1.000 |
| rnd_lt_L1 | 0.001 | [-0.000, 0.002] | 0.000 |
| rnd_lt_L4 | -0.002 | [-0.009, 0.005] | 0.002 |
| rnd_blk_L1 | 0.446 | [0.433, 0.458] | 0.153 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). lt_L4 and pc write the readout input, so they equal the swap at r by construction. persist_lt_L1to4 equals lt_L4 at every position by causality at r (r is the last position), so it carries no separate information.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998206, 1.008758], "rnd_lt_L4": [0.999519, 1.000689], "rnd_blk_L1": [0.999905, 1.000129]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | lt_L3 | lt_L4q1 | lt_L4mid | lt_L4q3 |
|---|---|---|---|---|---|---|---|
| a | 179 | 0.001 | 1.000 | 0.292 | 0.369 | 0.520 | 0.608 |
| b | 173 | 0.001 | 0.997 | 0.208 | 0.325 | 0.495 | 0.559 |
| c | 171 | 0.002 | 1.001 | 0.190 | 0.255 | 0.453 | 0.517 |
| d | 167 | -0.001 | 0.999 | 0.169 | 0.237 | 0.418 | 0.493 |
| e | 167 | 0.002 | 1.007 | 0.181 | 0.301 | 0.439 | 0.554 |

Predictions: {"P1_lt_L3_or_L4mid_ge_0.5": false, "P1_lt_L1_below_0.2": true, "P1_holds": false, "P2_blk_L1_ge_0.8": true, "note": "lt_L4 (end of loop 4) is the readout input and equals pc: the positive-control identity, not a result. Facts reference at the same sites (k15 map): at most 0.17."}

Timing: {"seconds_per_tried_item": 3.803, "seconds_per_kept_item": 5.099, "seconds_per_arm_and_item": {"lt_L1": 0.2364, "blk_L1": 0.2424, "lt_L2": 0.2399, "lt_L3": 0.2394, "lt_L4first": 0.2404, "lt_L4q1": 0.2387, "lt_L4mid": 0.2393, "lt_L4q3": 0.239, "lt_L4": 0.2394, "blk_L2": 0.2427, "blk_L3": 0.2427, "blk_L4": 0.2416, "persist_lt_L1to4": 0.2412, "rnd_lt_L1": 0.239, "rnd_lt_L4": 0.2391, "rnd_blk_L1": 0.2429, "pc": 0.2399}, "full_card_hours_at_this_rate": 1.283}
