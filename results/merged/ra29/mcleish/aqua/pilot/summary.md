# RA-29 mcleish aqua

Items: 250 built, 120 tried, 9 kept; drops {"screen_A": 80, "screen_B": 19, "gap_below_2": 12}; new gold letter of the kept items {"a": 0, "b": 3, "c": 1, "d": 2, "e": 3}.

| arm | NLD mean | 90% CI | writes B's letter |
|---|---|---|---|
| lt_L1 | -0.009 | [-0.024, 0.005] | 0.000 |
| blk_L1 | -0.009 | [-0.024, 0.005] | 0.000 |
| blk_L2 | -0.005 | [-0.015, 0.000] | 0.000 |
| blk_L3 | -0.009 | [-0.025, 0.004] | 0.000 |
| blk_L31 | 0.766 | [0.628, 0.908] | 0.778 |
| blk_L32 | 0.013 | [-0.004, 0.029] | 0.000 |
| blk_persist | 0.801 | [0.683, 0.919] | 0.778 |
| blk_e | 1.007 | [0.995, 1.021] | 1.000 |
| lt_L31 | 0.069 | [0.051, 0.085] | 0.000 |
| lt_L32first | 0.072 | [0.045, 0.099] | 0.000 |
| lt_L32mid | 0.339 | [0.289, 0.395] | 0.111 |
| lt_L32 | 0.924 | [0.902, 0.943] | 1.000 |
| rnd_lt_L1 | 0.004 | [-0.014, 0.020] | 0.000 |
| rnd_lt_L32 | -0.022 | [-0.058, 0.010] | 0.000 |
| rnd_blk_L1 | 0.001 | [-0.014, 0.014] | 0.000 |
| pc | 1.000 | [1.000, 1.000] | 1.000 |
| sz_fin_blk | 0.000 | [0.000, 0.000] | 0.000 |
| swap | 1.000 | [1.000, 1.000] | 1.000 |
| clean_A | 0.000 | [0.000, 0.000] | 0.000 |

Default baseline first (lt_L1: last-token donor patch at the end of loop 1). pc writes the coda's output at r (the readout input) and equals the swap at r by construction; sz_fin_blk (the same write at the block) is a structural zero. lt_L32 (end of loop 32) is a result: the coda still reads the block positions. blk_e writes B's prelude output at the block (the block then runs as in B in every loop); blk_L32 is the coda's direct read of the block. The raven rows are read as the refresh-model case of the placement rule, not as a commit-point measurement.

Random controls, norm ratio min and max over items: {"rnd_lt_L1": [0.99932, 1.000973], "rnd_lt_L32": [0.99975, 1.000433], "rnd_blk_L1": [0.999947, 1.00007]}

Mean NLD by the gold letter of prompt B (check of the letter bias):

| B's letter | n | lt_L1 | blk_L1 | blk_persist | lt_L31 | lt_L32mid | lt_L32 |
|---|---|---|---|---|---|---|---|
| b | 3 | 0.000 | 0.000 | 0.926 | 0.089 | 0.334 | 0.914 |
| c | 1 | -0.045 | -0.045 | 0.682 | 0.000 | 0.364 | 0.955 |
| d | 2 | 0.015 | 0.015 | 1.018 | 0.074 | 0.387 | 0.888 |
| e | 3 | -0.021 | -0.021 | 0.572 | 0.067 | 0.303 | 0.947 |

Predictions: {"not_scored": "fewer than 80 scored items (9)"}

Timing: {"seconds_per_tried_item": 0.988, "seconds_per_kept_item": 5.578, "seconds_per_arm_and_item": {"lt_L1": 0.1956, "blk_L1": 0.2073, "blk_L2": 0.2202, "blk_L3": 0.2198, "blk_L31": 0.2136, "blk_L32": 0.2001, "blk_persist": 0.4997, "blk_e": 0.2087, "lt_L31": 0.1986, "lt_L32first": 0.1955, "lt_L32mid": 0.1939, "lt_L32": 0.2009, "rnd_lt_L1": 0.2021, "rnd_lt_L32": 0.2047, "rnd_blk_L1": 0.2127, "pc": 0.2136, "sz_fin_blk": 0.2328}, "full_card_hours_at_this_rate": 0.069}
