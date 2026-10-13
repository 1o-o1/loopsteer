# RA-29 mcleish csqa

Items: 1215 built, 120 tried, 26 kept; drops {"screen_A": 70, "screen_B": 19, "gap_below_2": 5}; new gold letter of the kept items {"a": 7, "b": 3, "c": 8, "d": 4, "e": 4}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.001 | [-0.009, 0.008] | 0.000 |
| blk_L1 | -0.003 | [-0.009, 0.003] | 0.000 |
| blk_L2 | -0.004 | [-0.011, 0.004] | 0.000 |
| blk_L3 | -0.003 | [-0.010, 0.004] | 0.000 |
| blk_L31 | 0.652 | [0.573, 0.727] | 0.577 |
| blk_L32 | 0.013 | [0.002, 0.024] | 0.000 |
| blk_persist | 0.840 | [0.795, 0.884] | 0.846 |
| blk_e | 0.993 | [0.985, 1.001] | 1.000 |
| lt_L31 | 0.115 | [0.096, 0.134] | 0.000 |
| lt_L32first | 0.111 | [0.093, 0.131] | 0.000 |
| lt_L32mid | 0.359 | [0.315, 0.407] | 0.192 |
| lt_L32 | 0.944 | [0.933, 0.954] | 1.000 |
| rnd_lt_L1 | -0.006 | [-0.013, 0.001] | 0.000 |
| rnd_lt_L32 | -0.014 | [-0.035, 0.006] | 0.000 |
| rnd_blk_L1 | -0.005 | [-0.012, 0.002] | 0.000 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| sz_fin_blk | 0.000 | [0.000, 0.000] | 0.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.998884, 1.002404], "rnd_lt_L32": [0.999704, 1.000387], "rnd_blk_L1": [0.999975, 1.000083]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |
|---|---|---|---|---|---|---|---|
| a | 7 | -0.004 | -0.006 | 0.854 | 0.121 | 0.343 | 0.937 |
| b | 3 | 0.017 | -0.017 | 1.008 | 0.122 | 0.378 | 0.957 |
| c | 8 | 0.011 | -0.002 | 0.867 | 0.101 | 0.318 | 0.947 |
| d | 4 | -0.018 | 0.000 | 0.717 | 0.102 | 0.382 | 0.939 |
| e | 4 | -0.013 | 0.008 | 0.755 | 0.141 | 0.434 | 0.949 |

Predictions: {"not_scored": "fewer than 80 scored items (26)"}

Timing: {"seconds_per_tried_item": 1.356, "seconds_per_kept_item": 4.235, "seconds_per_arm_and_item": {"lt_L1": 0.1903, "blk_L1": 0.1951, "blk_L2": 0.1939, "blk_L3": 0.1908, "blk_L31": 0.1911, "blk_L32": 0.1909, "blk_persist": 0.2822, "blk_e": 0.1908, "lt_L31": 0.189, "lt_L32first": 0.1892, "lt_L32mid": 0.1911, "lt_L32": 0.1895, "rnd_lt_L1": 0.1898, "rnd_lt_L32": 0.1907, "rnd_blk_L1": 0.1931, "pc": 0.189, "sz_fin_blk": 0.1913}, "full_card_hours_at_this_rate": 0.458}
