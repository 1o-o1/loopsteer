# Card-hours per queue group (estimate)

Per-item rates are seconds per item measured in the pilot artifacts on the fast host, one process per card,
except where marked: rates measured on a GPU shared with other jobs are upper bounds for a dedicated card;
assumed rates have no measurement behind them (see rate_source in joblist.tsv). 48 cards, one process each.

| group | jobs | card-hours | from shared-GPU rates | from assumed rates | longest chain (h) | wall at 48 cards (h) |
|---|---|---|---|---|---|---|
| g0_xllm | 1 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| g1_copy | 49 | 47.5 | 41.8 | 3.1 | 2.9 | 2.9 |
| g6_ifm | 51 | 54.5 | 31.2 | 2.1 | 6.8 | 6.8 |
| g2_gen | 188 | 72.1 | 38.6 | 20.7 | 5.1 | 5.1 |
| g3_map | 85 | 106.6 | 60.6 | 36.5 | 14.1 | 14.1 |
| g4_cd | 4 | 3.5 | 0.0 | 3.5 | 1.4 | 1.4 |
| g7_ra23 | 175 | 112.1 | 26.6 | 97.3 | 8.0 | 8.0 |
| g5_huginn | 88 | 135.1 | 11.2 | 118.2 | 4.1 | 4.1 |
| g9_merge | 1 | 0.2 | 0.0 | 0.0 | 0.2 | 0.2 |
| total | 642 | 532.6 | | | | |
