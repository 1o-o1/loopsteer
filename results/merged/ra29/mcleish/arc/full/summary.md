# RA-29 mcleish arc

Items: 1172 built, 1172 tried, 250 kept; drops {"screen_A": 629, "gap_below_2": 92, "screen_B": 201}; new gold letter of the kept items {"a": 55, "b": 71, "c": 73, "d": 50, "e": 1}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.000 | [-0.002, 0.002] | 0.000 |
| blk_L1 | 0.003 | [0.001, 0.005] | 0.000 |
| blk_L2 | 0.002 | [-0.000, 0.004] | 0.000 |
| blk_L3 | 0.002 | [-0.001, 0.004] | 0.000 |
| blk_L31 | 0.688 | [0.658, 0.715] | 0.664 |
| blk_L32 | 0.023 | [0.019, 0.027] | 0.000 |
| blk_persist | 0.832 | [0.815, 0.850] | 0.872 |
| blk_e | 1.001 | [0.999, 1.004] | 0.996 |
| lt_L31 | 0.101 | [0.094, 0.107] | 0.008 |
| lt_L32first | 0.104 | [0.097, 0.110] | 0.008 |
| lt_L32mid | 0.354 | [0.339, 0.368] | 0.156 |
| lt_L32 | 0.895 | [0.890, 0.899] | 0.972 |
| rnd_lt_L1 | 0.000 | [-0.002, 0.002] | 0.000 |
| rnd_lt_L32 | -0.001 | [-0.008, 0.005] | 0.000 |
| rnd_blk_L1 | 0.004 | [0.002, 0.006] | 0.000 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| sz_fin_blk | 0.000 | [0.000, 0.000] | 0.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998216, 1.003475], "rnd_lt_L32": [0.999584, 1.000563], "rnd_blk_L1": [0.999905, 1.000116]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |
|---|---|---|---|---|---|---|---|
| a | 55 | -0.003 | 0.001 | 0.830 | 0.133 | 0.397 | 0.885 |
| b | 71 | -0.004 | 0.001 | 0.821 | 0.088 | 0.325 | 0.897 |
| c | 73 | 0.004 | 0.004 | 0.818 | 0.079 | 0.345 | 0.893 |
| d | 50 | 0.001 | 0.007 | 0.876 | 0.116 | 0.360 | 0.903 |
| e | 1 | 0.037 | 0.037 | 0.593 | 0.074 | 0.333 | 0.926 |

Predictions: {"PR1_single_loop_block_about_0": false, "PR1_single_loop_block_ci_inside_0.1": false, "PR1_blk_L32_ge_0.5": false, "PR1_blk_persist_ge_0.5": true, "PR1_blk_e_ge_0.9": true, "PR2_lt_L1_and_L31_about_0": false, "PR2_lt_L1_and_L31_ci_inside_0.1": false, "PR2_lt_L32_end_ge_0.5": true, "contraction_ratio_blk32": {"blk_L1": 0.0159, "blk_L2": 0.0159, "blk_L3": 0.0152, "blk_L31": 0.5534, "blk_L32": 1.0, "blk_persist": 1.0, "blk_e": 1.0}, "contraction_ratio_r32": {"lt_L1": 0.0347, "lt_L31": 0.3217, "lt_L32": 1.0}, "blk_persist_minus_blk_L32": {"mean": 0.8091, "ci90": [0.7913, 0.8263], "n_items": 250}, "note": "about 0 = |mean NLD| <= 0.1; lands = mean NLD >= 0.5; reaches the positive control = mean NLD >= 0.9; pc (coda output at r) is the positive-control identity; the raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement", "PR1_holds": false, "PR2_holds": false}

Timing: {"seconds_per_tried_item": 1.339, "seconds_per_kept_item": 4.243, "seconds_per_arm_and_item": {"lt_L1": 0.192, "blk_L1": 0.1928, "blk_L2": 0.1918, "blk_L3": 0.1916, "blk_L31": 0.1915, "blk_L32": 0.1923, "blk_persist": 0.2948, "blk_e": 0.1918, "lt_L31": 0.1907, "lt_L32first": 0.1892, "lt_L32mid": 0.1905, "lt_L32": 0.1892, "rnd_lt_L1": 0.1899, "rnd_lt_L32": 0.1898, "rnd_blk_L1": 0.1928, "pc": 0.1896, "sz_fin_blk": 0.1923}, "full_card_hours_at_this_rate": 0.436}
