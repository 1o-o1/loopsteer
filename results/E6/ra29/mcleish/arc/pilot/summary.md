# RA-29 mcleish arc

Items: 1172 built, 120 tried, 22 kept; drops {"screen_A": 68, "gap_below_2": 13, "screen_B": 17}; new gold letter of the kept items {"a": 3, "b": 7, "c": 9, "d": 3, "e": 0}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.000 | [-0.004, 0.004] | 0.000 |
| blk_L1 | 0.002 | [-0.005, 0.007] | 0.000 |
| blk_L2 | -0.001 | [-0.007, 0.006] | 0.000 |
| blk_L3 | -0.003 | [-0.010, 0.004] | 0.000 |
| blk_L31 | 0.692 | [0.628, 0.757] | 0.682 |
| blk_L32 | 0.017 | [0.006, 0.028] | 0.000 |
| blk_persist | 0.804 | [0.741, 0.865] | 0.818 |
| blk_e | 0.998 | [0.995, 1.002] | 1.000 |
| lt_L31 | 0.089 | [0.070, 0.108] | 0.000 |
| lt_L32first | 0.093 | [0.072, 0.113] | 0.000 |
| lt_L32mid | 0.379 | [0.333, 0.424] | 0.136 |
| lt_L32 | 0.881 | [0.867, 0.896] | 0.955 |
| rnd_lt_L1 | -0.002 | [-0.006, 0.002] | 0.000 |
| rnd_lt_L32 | -0.004 | [-0.021, 0.015] | 0.000 |
| rnd_blk_L1 | -0.003 | [-0.010, 0.003] | 0.000 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| sz_fin_blk | 0.000 | [0.000, 0.000] | 0.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.999525, 1.001895], "rnd_lt_L32": [0.999742, 1.000277], "rnd_blk_L1": [0.999949, 1.000077]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |
|---|---|---|---|---|---|---|---|
| a | 3 | 0.000 | 0.000 | 0.599 | 0.130 | 0.369 | 0.895 |
| b | 7 | -0.007 | -0.007 | 0.809 | 0.093 | 0.383 | 0.890 |
| c | 9 | 0.006 | 0.004 | 0.858 | 0.053 | 0.367 | 0.872 |
| d | 3 | -0.005 | 0.016 | 0.833 | 0.149 | 0.416 | 0.875 |

Predictions: {"not_scored": "fewer than 80 scored items (22)"}

Timing: {"seconds_per_tried_item": 1.211, "seconds_per_kept_item": 4.168, "seconds_per_arm_and_item": {"lt_L1": 0.1896, "blk_L1": 0.1893, "blk_L2": 0.1905, "blk_L3": 0.1894, "blk_L31": 0.1955, "blk_L32": 0.1893, "blk_persist": 0.2816, "blk_e": 0.1881, "lt_L31": 0.1866, "lt_L32first": 0.1865, "lt_L32mid": 0.1869, "lt_L32": 0.1874, "rnd_lt_L1": 0.1875, "rnd_lt_L32": 0.1867, "rnd_blk_L1": 0.19, "pc": 0.1867, "sz_fin_blk": 0.1953}, "full_card_hours_at_this_rate": 0.394}
