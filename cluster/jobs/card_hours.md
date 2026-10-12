# Card-hours per queue group (estimate)

Per-item rates are seconds per item measured in the pilot artifacts on the fast host, one process per card,
except where marked: rates measured on a GPU shared with other jobs are upper bounds for a dedicated card;
assumed rates have no measurement behind them (see rate_source in joblist.tsv). 48 cards, one process each.

| group | jobs | card-hours | from shared-GPU rates | from assumed rates | longest chain (h) | wall at 48 cards (h) |
|---|---|---|---|---|---|---|
| g0_xllm | 1 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| g1_copy | 20 | 22.9 | 21.7 | 0.8 | 2.1 | 2.1 |
| g6_ifm | 50 | 52.9 | 29.6 | 2.1 | 6.8 | 6.8 |
| g12_ifmv1 | 27 | 23.4 | 0.0 | 2.1 | 2.6 | 2.6 |
| g2_gen | 144 | 62.9 | 35.0 | 18.9 | 5.2 | 5.2 |
| g11_gen26t | 107 | 236.2 | 0.0 | 197.5 | 30.6 | 30.6 |
| g17_fillmap | 31 | 32.9 | 12.7 | 18.4 | 2.6 | 2.6 |
| g3_map | 85 | 106.6 | 60.6 | 36.5 | 14.1 | 14.1 |
| g4_cd | 4 | 3.5 | 0.0 | 3.5 | 1.4 | 1.4 |
| g7_ra23 | 175 | 112.1 | 26.6 | 97.3 | 8.0 | 8.0 |
| g5_huginn | 88 | 135.1 | 11.2 | 118.2 | 4.1 | 4.1 |
| g13_paththink | 7 | 50.2 | 0.0 | 50.2 | 17.0 | 17.0 |
| g14_ifmy1 | 7 | 12.8 | 12.8 | 0.0 | 2.5 | 2.5 |
| g18_fillgen | 135 | 133.0 | 132.2 | 0.0 | 3.2 | 3.2 |
| g15_hgen | 26 | 25.3 | 25.2 | 0.0 | 2.2 | 2.2 |
| g16_ifmgrid | 2 | 2.7 | 2.7 | 0.0 | 1.6 | 1.6 |
| g19_e6ra30 | 2 | 3.6 | 0.0 | 0.0 | 3.6 | 3.6 |
| g20_e6ra29 | 4 | 2.7 | 0.0 | 0.0 | 2.1 | 2.1 |
| g21_e6ra31a | 2 | 3.3 | 0.0 | 0.0 | 3.3 | 3.3 |
| g22_e6ra31b | 16 | 20.1 | 0.0 | 0.0 | 1.8 | 1.8 |
| g23_e6merge | 3 | 0.2 | 0.0 | 0.0 | 0.1 | 0.1 |
| g24_e6bra29 | 4 | 10.7 | 0.0 | 0.0 | 4.8 | 4.8 |
| g25_e6bra31t | 4 | 4.5 | 0.0 | 0.0 | 3.0 | 3.0 |
| g26_e6bra30 | 2 | 4.1 | 0.0 | 0.0 | 4.1 | 4.1 |
| g28_e6bmerge | 3 | 0.2 | 0.0 | 0.0 | 0.1 | 0.1 |
| g9_merge | 1 | 0.2 | 0.0 | 0.0 | 0.2 | 0.2 |
| total | 950 | 1063.1 | | | | |
