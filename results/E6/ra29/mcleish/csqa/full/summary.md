# RA-29 mcleish csqa

Items: 1215 built, 1215 tried, 248 kept; drops {"screen_A": 687, "screen_B": 181, "gap_below_2": 99}; new gold letter of the kept items {"a": 58, "b": 50, "c": 55, "d": 45, "e": 40}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | 0.001 | [-0.001, 0.003] | 0.000 |
| blk_L1 | -0.001 | [-0.004, 0.001] | 0.000 |
| blk_L2 | -0.000 | [-0.003, 0.002] | 0.000 |
| blk_L3 | 0.003 | [0.001, 0.006] | 0.000 |
| blk_L31 | 0.651 | [0.627, 0.676] | 0.597 |
| blk_L32 | 0.017 | [0.012, 0.021] | 0.000 |
| blk_persist | 0.813 | [0.796, 0.830] | 0.815 |
| blk_e | 1.000 | [0.997, 1.002] | 0.992 |
| lt_L31 | 0.110 | [0.104, 0.115] | 0.000 |
| lt_L32first | 0.110 | [0.104, 0.116] | 0.000 |
| lt_L32mid | 0.333 | [0.320, 0.346] | 0.125 |
| lt_L32 | 0.947 | [0.942, 0.951] | 0.988 |
| rnd_lt_L1 | -0.001 | [-0.003, 0.001] | 0.000 |
| rnd_lt_L32 | -0.015 | [-0.022, -0.009] | 0.000 |
| rnd_blk_L1 | -0.000 | [-0.003, 0.002] | 0.000 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| sz_fin_blk | 0.000 | [0.000, 0.000] | 0.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998325, 1.003572], "rnd_lt_L32": [0.999491, 1.000747], "rnd_blk_L1": [0.999924, 1.000088]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |
|---|---|---|---|---|---|---|---|
| a | 58 | 0.004 | -0.003 | 0.800 | 0.122 | 0.345 | 0.945 |
| b | 50 | 0.004 | -0.004 | 0.828 | 0.088 | 0.288 | 0.948 |
| c | 55 | 0.003 | 0.001 | 0.830 | 0.095 | 0.330 | 0.945 |
| d | 45 | -0.003 | 0.002 | 0.863 | 0.115 | 0.355 | 0.946 |
| e | 40 | -0.003 | -0.003 | 0.737 | 0.132 | 0.351 | 0.951 |

Predictions: {"PR1_single_loop_block_about_0": false, "PR1_single_loop_block_ci_inside_0.1": false, "PR1_blk_L32_ge_0.5": false, "PR1_blk_persist_ge_0.5": true, "PR1_blk_e_ge_0.9": true, "PR2_lt_L1_and_L31_about_0": false, "PR2_lt_L1_and_L31_ci_inside_0.1": false, "PR2_lt_L32_end_ge_0.5": true, "contraction_ratio_blk32": {"blk_L1": 0.0161, "blk_L2": 0.0161, "blk_L3": 0.0173, "blk_L31": 0.5123, "blk_L32": 1.0, "blk_persist": 1.0, "blk_e": 1.0}, "contraction_ratio_r32": {"lt_L1": 0.0364, "lt_L31": 0.3025, "lt_L32": 1.0}, "blk_persist_minus_blk_L32": {"mean": 0.7969, "ci90": [0.7792, 0.8144], "n_items": 248}, "note": "about 0 = |mean NLD| <= 0.1; lands = mean NLD >= 0.5; reaches the positive control = mean NLD >= 0.9; pc (coda output at r) is the positive-control identity; the raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement", "PR1_holds": false, "PR2_holds": false}

Timing: {"seconds_per_tried_item": 1.309, "seconds_per_kept_item": 4.234, "seconds_per_arm_and_item": {"lt_L1": 0.1918, "blk_L1": 0.1922, "blk_L2": 0.1922, "blk_L3": 0.1921, "blk_L31": 0.1913, "blk_L32": 0.1917, "blk_persist": 0.2906, "blk_e": 0.1927, "lt_L31": 0.1899, "lt_L32first": 0.1899, "lt_L32mid": 0.191, "lt_L32": 0.19, "rnd_lt_L1": 0.1906, "rnd_lt_L32": 0.1896, "rnd_blk_L1": 0.1939, "pc": 0.1906, "sz_fin_blk": 0.1919}, "full_card_hours_at_this_rate": 0.442}
