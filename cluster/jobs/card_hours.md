# Card-hours per queue group (estimate)

Per-item rates are seconds per item measured in the pilot artifacts on the fast host, one process per card,
except where marked: rates measured on a GPU shared with other jobs are upper bounds for a dedicated card;
assumed rates have no measurement behind them (see rate_source in joblist.tsv). 48 cards, one process each.

| group | jobs | card-hours | from shared-GPU rates | from assumed rates | longest chain (h) | wall at 48 cards (h) |
|---|---|---|---|---|---|---|
| g0_xllm | 1 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| g1_copy | 20 | 22.9 | 21.7 | 0.8 | 2.1 | 2.1 |
| g6_ifm | 50 | 52.9 | 29.6 | 2.1 | 6.8 | 6.8 |
| g2_gen | 144 | 62.9 | 35.0 | 18.9 | 5.2 | 5.2 |
| g11_gen26t | 107 | 236.2 | 0.0 | 197.5 | 30.6 | 30.6 |
| g3_map | 85 | 106.6 | 60.6 | 36.5 | 14.1 | 14.1 |
| g4_cd | 4 | 3.5 | 0.0 | 3.5 | 1.4 | 1.4 |
| g7_ra23 | 175 | 112.1 | 26.6 | 97.3 | 8.0 | 8.0 |
| g5_huginn | 88 | 135.1 | 11.2 | 118.2 | 4.1 | 4.1 |
| g13_paththink | 7 | 50.2 | 0.0 | 50.2 | 17.0 | 17.0 |
| g14_ifmy1 | 7 | 12.8 | 12.8 | 0.0 | 2.5 | 2.5 |
| g9_merge | 1 | 0.2 | 0.0 | 0.0 | 0.2 | 0.2 |
| total | 689 | 796.4 | | | | |
