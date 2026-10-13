# K15 full A summary

Effect e = (m_edit - m_A)/(m_B - m_A), m = logit tB - logit tA at the final readout; flip = top-1 is tB. Cells: mean [95% bootstrap CI over items, 1000 resamples, seed 0]. Rows with final top-1 margin < 1.0 logit rerun alone at batch 1. Arm A: own = k13 immediate effect = change in m at the edited loop's end over the clean m_B - m_A at that same loop end (None, class unreadable, when that clean gap is < 1.0 logit); final = change in m at loop K over m_B - m_A at the final readout. Classes with threshold 0.25: never read (own < t, final < t), re-derived (own >= t, final < t), persisted (both >= t), delayed (own < t, final >= t).

GPU minutes: {'cluster': 13.42} (total 13.42).

## ifm_d28

Host cluster, K 1, layers {'L0': 0, 'Lmid': 14, 'Llast': 27}, readout: m.output(state at the last loop-end point), as k13. Load 19s, total 805s.

### ifm_d28 / D1

Pairs 136 ({'capital': 91, 'city': 45}), scanned 158 of 313 candidates, truncated False, know stats {'table_rejects': 0, 'single_rejects': 22, 'table_pass_single_fail': 0, 'no_para': 0}. Paraphrase modes {'aligned': 242, 'last_token': 30}. Reruns at batch 1: 822 of 5032 rows; max in-batch vs batch-1 clean logit gap 0.281. O5 effect 1.00 [1.00, 1.00].

Arm S (source edit at subject/operand positions), effect [CI] / flips:

| op a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 0.5 | 0.49 [0.47, 0.52] / 0.23 | 0.45 [0.44, 0.47] / 0.15 | 0.00 [0.00, 0.00] / 0.00 |
| O2 1 | 0.98 [0.97, 0.99] / 0.99 | 0.96 [0.95, 0.97] / 1.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 2 | 1.04 [1.01, 1.06] / 0.85 | 1.16 [1.14, 1.17] / 0.77 | 0.00 [0.00, 0.00] / 0.00 |
| O4 0.5 | 0.01 [0.00, 0.01] / 0.00 | 0.00 [0.00, 0.01] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 1 | 0.05 [0.03, 0.06] / 0.00 | 0.02 [0.02, 0.03] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 2 | 0.21 [0.19, 0.24] / 0.00 | 0.11 [0.09, 0.12] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |

Arm S change in logit A / logit B (a = 1):

| op | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 | -10.57 / 10.57 | -9.90 / 10.82 | 0.00 / 0.00 |
| O4 | -0.95 / 0.06 | -0.39 / 0.19 | 0.00 / 0.00 |

Arm A (derived edit at the last position): own / final [CI of final] / majority class (fractions never read / re-derived / persisted / delayed / unreadable):

| a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| 1 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.01 / 0.01 [0.01, 0.02] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.99 / 0.99 [0.99, 1.00] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 2 | 0.00 / 0.00 [0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.03 / 0.03 [0.03, 0.04] / never_read (1.00/0.00/0.00/0.00/0.00) | 1.64 / 1.64 [1.63, 1.66] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 4 | 0.00 / 0.00 [0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.10 / 0.10 [0.08, 0.11] / never_read (0.96/0.00/0.04/0.00/0.00) | 2.04 / 2.04 [2.00, 2.08] / persisted (0.00/0.00/1.00/0.00/0.00) |

Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):

- L0@1: change in m / D 0.00; state |dh|/|a v| 9.77; cos(dh, v) 0.01; flip at own loop 0.00
- Lmid@1: change in m / D 0.03; state |dh|/|a v| 2.65; cos(dh, v) 0.34; flip at own loop 0.00
- Llast@1: change in m / D 1.64; state |dh|/|a v| 1.00; cos(dh, v) 1.00; flip at own loop 0.91

Clean m_B - m_A at each loop end, logits: 21.52 [20.54, 22.59]; as a fraction of the final gap: 1.00 [1.00, 1.00]

Arm C (O1 at Llast@1, subject positions):

| a | effect | flips | mid fraction [0.2, 0.8] | BC |
|---|---|---|---|---|
| 0.3 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.35 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.4 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.45 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.5 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.55 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.6 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.65 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.7 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |

### ifm_d28 / D2

Pairs 567 ({'op1': 282, 'op2': 285}), scanned 585 of 592 candidates, truncated False, know stats {'table_rejects': 18, 'single_rejects': 0, 'table_pass_single_fail': 0, 'no_para': 0}. Paraphrase modes {'aligned': 1134}. Reruns at batch 1: 1289 of 20979 rows; max in-batch vs batch-1 clean logit gap 0.406. O5 effect 1.00 [1.00, 1.00].

Arm S (source edit at subject/operand positions), effect [CI] / flips:

| op a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 0.5 | 0.48 [0.47, 0.50] / 0.37 | 0.01 [0.01, 0.01] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 1 | 1.00 [1.00, 1.00] / 1.00 | 0.02 [0.02, 0.03] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 2 | 1.00 [0.99, 1.01] / 0.89 | 0.05 [0.04, 0.05] / 0.01 | 0.00 [0.00, 0.00] / 0.00 |
| O4 0.5 | 0.01 [0.00, 0.01] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 1 | 0.02 [0.02, 0.03] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 2 | 0.14 [0.12, 0.15] / 0.03 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |

Arm S change in logit A / logit B (a = 1):

| op | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 | -7.24 / 7.14 | -0.04 / 0.29 | 0.00 / 0.00 |
| O4 | -0.18 / 0.16 | 0.00 / 0.00 | 0.00 / 0.00 |

Arm A (derived edit at the last position): own / final [CI of final] / majority class (fractions never read / re-derived / persisted / delayed / unreadable):

| a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| 1 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.79 / 0.79 [0.78, 0.80] / persisted (0.00/0.00/1.00/0.00/0.00) | 0.84 / 0.84 [0.83, 0.85] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 2 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.93 / 0.93 [0.91, 0.94] / persisted (0.00/0.00/1.00/0.00/0.00) | 1.50 / 1.50 [1.48, 1.51] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 4 | 0.00 / 0.00 [0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.68 / 0.68 [0.66, 0.70] / persisted (0.04/0.00/0.96/0.00/0.00) | 2.05 / 2.05 [2.03, 2.07] / persisted (0.00/0.00/1.00/0.00/0.00) |

Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):

- L0@1: change in m / D 0.00; state |dh|/|a v| 21.65; cos(dh, v) 0.01; flip at own loop 0.00
- Lmid@1: change in m / D 0.93; state |dh|/|a v| 4.90; cos(dh, v) 0.15; flip at own loop 0.81
- Llast@1: change in m / D 1.50; state |dh|/|a v| 1.00; cos(dh, v) 1.00; flip at own loop 0.99

Clean m_B - m_A at each loop end, logits: 14.42 [14.12, 14.72]; as a fraction of the final gap: 1.00 [1.00, 1.00]

Arm C (O1 at Llast@1, subject positions):

| a | effect | flips | mid fraction [0.2, 0.8] | BC |
|---|---|---|---|---|
| 0.3 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.35 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.4 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.45 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.5 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.55 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.6 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.65 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.7 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |

### ifm_d28 / CF

Pairs 287 ({'P103': 200, 'P364': 84, 'P641': 3}), scanned 1764 of 1764 candidates, truncated False, know stats {'table_rejects': 1472, 'single_rejects': 5, 'table_pass_single_fail': 5, 'no_para': 0}. Paraphrase modes {'aligned': 572, 'last_token': 2}. Reruns at batch 1: 3946 of 10619 rows; max in-batch vs batch-1 clean logit gap 0.258. O5 effect 1.00 [1.00, 1.00].

Arm S (source edit at subject/operand positions), effect [CI] / flips:

| op a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 0.5 | 0.48 [0.46, 0.51] / 0.37 | 0.47 [0.46, 0.49] / 0.37 | 0.00 [0.00, 0.00] / 0.00 |
| O2 1 | 1.00 [0.99, 1.00] / 0.97 | 0.97 [0.95, 0.98] / 0.92 | 0.00 [0.00, 0.00] / 0.00 |
| O2 2 | 0.97 [0.95, 1.00] / 0.69 | 1.10 [1.07, 1.12] / 0.83 | 0.00 [0.00, 0.00] / 0.00 |
| O4 0.5 | 0.05 [0.04, 0.07] / 0.02 | 0.02 [0.01, 0.03] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 1 | 0.16 [0.13, 0.18] / 0.05 | 0.08 [0.07, 0.09] / 0.02 | 0.00 [0.00, 0.00] / 0.00 |
| O4 2 | 0.33 [0.30, 0.36] / 0.05 | 0.23 [0.21, 0.25] / 0.06 | 0.00 [0.00, 0.00] / 0.00 |

Arm S change in logit A / logit B (a = 1):

| op | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 | -4.15 / 4.10 | -3.97 / 4.03 | 0.00 / 0.00 |
| O4 | -1.81 / -0.59 | -0.54 / 0.09 | 0.00 / 0.00 |

Arm S by CounterFact relation, O2 a=1, effect [CI] (n):

| relation | n | L0@1 | Lmid@1 | Llast@1 | L0@1 |
|---|---|---|---|---|---|
| P103 | 200 | 1.00 [0.99, 1.00] | 0.98 [0.96, 1.00] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] |
| P364 | 84 | 1.00 [0.99, 1.00] | 0.94 [0.91, 0.97] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] |
| P641 | 3 | 1.00 [0.96, 1.06] | 0.91 [0.78, 1.00] | 0.00 [0.00, 0.00] | 1.00 [0.96, 1.06] |

Arm A (derived edit at the last position): own / final [CI of final] / majority class (fractions never read / re-derived / persisted / delayed / unreadable):

| a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| 1 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.03 / 0.03 [0.02, 0.04] / never_read (0.99/0.00/0.01/0.00/0.00) | 0.80 / 0.80 [0.78, 0.83] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 2 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.06 / 0.06 [0.05, 0.08] / never_read (0.90/0.00/0.10/0.00/0.00) | 1.50 / 1.50 [1.46, 1.55] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 4 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.15 / 0.15 [0.12, 0.19] / never_read (0.72/0.00/0.28/0.00/0.00) | 2.32 / 2.32 [2.25, 2.39] / persisted (0.00/0.00/1.00/0.00/0.00) |

Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):

- L0@1: change in m / D 0.00; state |dh|/|a v| 5.57; cos(dh, v) 0.04; flip at own loop 0.00
- Lmid@1: change in m / D 0.06; state |dh|/|a v| 2.59; cos(dh, v) 0.34; flip at own loop 0.01
- Llast@1: change in m / D 1.50; state |dh|/|a v| 1.00; cos(dh, v) 1.00; flip at own loop 0.75

Clean m_B - m_A at each loop end, logits: 8.28 [7.87, 8.68]; as a fraction of the final gap: 1.00 [1.00, 1.00]

Arm C (O1 at Llast@1, subject positions):

| a | effect | flips | mid fraction [0.2, 0.8] | BC |
|---|---|---|---|---|
| 0.3 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.35 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.4 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.45 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.5 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.55 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.6 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.65 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.7 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |

### ifm_d28 / D4

Pairs 320 ({'T1': 143, 'T2': 177}), scanned 379 of 806 candidates, truncated False, know stats {'table_rejects': 58, 'single_rejects': 1, 'table_pass_single_fail': 1, 'no_para': 0}. Paraphrase modes {'aligned': 640}. Reruns at batch 1: 1691 of 11840 rows; max in-batch vs batch-1 clean logit gap 0.438. O5 effect 1.00 [1.00, 1.00].

Arm S (source edit at subject/operand positions), effect [CI] / flips:

| op a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 0.5 | 0.49 [0.47, 0.51] / 0.43 | 0.13 [0.12, 0.14] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 1 | 1.00 [0.99, 1.00] / 0.99 | 0.29 [0.26, 0.31] / 0.13 | 0.00 [0.00, 0.00] / 0.00 |
| O2 2 | 1.03 [1.02, 1.03] / 0.97 | 0.34 [0.32, 0.36] / 0.21 | 0.00 [0.00, 0.00] / 0.00 |
| O4 0.5 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.01] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 1 | 0.00 [-0.00, 0.01] / 0.00 | 0.01 [0.01, 0.02] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 2 | 0.03 [0.02, 0.03] / 0.00 | 0.03 [0.03, 0.04] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |

Arm S change in logit A / logit B (a = 1):

| op | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 | -4.27 / 4.29 | -0.58 / 2.00 | 0.00 / 0.00 |
| O4 | -0.02 / -0.00 | -0.06 / 0.05 | 0.00 / 0.00 |

Arm A (derived edit at the last position): own / final [CI of final] / majority class (fractions never read / re-derived / persisted / delayed / unreadable):

| a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| 1 | 0.00 / 0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.58 / 0.58 [0.56, 0.60] / persisted (0.02/0.00/0.98/0.00/0.00) | 1.04 / 1.04 [1.03, 1.06] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 2 | -0.00 / -0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 0.91 / 0.91 [0.90, 0.94] / persisted (0.00/0.00/1.00/0.00/0.00) | 1.94 / 1.94 [1.91, 1.97] / persisted (0.00/0.00/1.00/0.00/0.00) |
| 4 | -0.00 / -0.00 [-0.00, 0.00] / never_read (1.00/0.00/0.00/0.00/0.00) | 1.08 / 1.08 [1.04, 1.12] / persisted (0.03/0.00/0.97/0.00/0.00) | 2.97 / 2.97 [2.92, 3.02] / persisted (0.00/0.00/1.00/0.00/0.00) |

Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):

- L0@1: change in m / D -0.00; state |dh|/|a v| 40.25; cos(dh, v) 0.00; flip at own loop 0.00
- Lmid@1: change in m / D 0.91; state |dh|/|a v| 3.26; cos(dh, v) 0.34; flip at own loop 0.95
- Llast@1: change in m / D 1.94; state |dh|/|a v| 1.00; cos(dh, v) 1.00; flip at own loop 1.00

Clean m_B - m_A at each loop end, logits: 8.60 [8.37, 8.81]; as a fraction of the final gap: 1.00 [1.00, 1.00]

Arm C (O1 at Llast@1, subject positions):

| a | effect | flips | mid fraction [0.2, 0.8] | BC |
|---|---|---|---|---|
| 0.3 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.35 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.4 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.45 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.5 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.55 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.6 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.65 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.7 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |

### ifm_d28 / D5

Pairs 4 ({'T2': 4}), scanned 592 of 594 candidates, truncated False, know stats {'table_rejects': 588, 'single_rejects': 0, 'table_pass_single_fail': 0, 'no_para': 0}. Paraphrase modes {'aligned': 8}. Reruns at batch 1: 113 of 148 rows; max in-batch vs batch-1 clean logit gap 0.211. O5 effect 1.00 [1.00, 1.00].

Arm S (source edit at subject/operand positions), effect [CI] / flips:

| op a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 0.5 | 0.40 [0.24, 0.57] / 0.25 | 0.06 [0.00, 0.12] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 1 | 0.96 [0.89, 1.00] / 0.75 | 0.17 [0.04, 0.33] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O2 2 | 1.06 [0.87, 1.24] / 0.75 | 0.29 [0.08, 0.57] / 0.50 | 0.00 [0.00, 0.00] / 0.00 |
| O4 0.5 | 0.01 [-0.07, 0.10] / 0.00 | -0.03 [-0.07, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 1 | -0.05 [-0.09, 0.00] / 0.00 | -0.03 [-0.07, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| O4 2 | 0.07 [-0.09, 0.30] / 0.00 | -0.01 [-0.03, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |

Arm S change in logit A / logit B (a = 1):

| op | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| O2 | -1.38 / 0.88 | 0.00 / 0.22 | 0.00 / 0.00 |
| O4 | -0.06 / -0.16 | 0.06 / 0.00 | 0.00 / 0.00 |

Arm A (derived edit at the last position): own / final [CI of final] / majority class (fractions never read / re-derived / persisted / delayed / unreadable):

| a | L0@1 | Lmid@1 | Llast@1 |
|---|---|---|---|
| 1 | -0.03 / -0.02 [-0.07, 0.00] / never_read (0.75/0.00/0.00/0.00/0.25) | 0.56 / 0.53 [0.41, 0.64] / persisted (0.00/0.00/0.75/0.00/0.25) | 0.80 / 0.63 [0.30, 0.97] / persisted (0.00/0.00/0.75/0.00/0.25) |
| 2 | -0.02 / -0.02 [-0.07, 0.02] / never_read (0.75/0.00/0.00/0.00/0.25) | 1.28 / 1.18 [0.92, 1.55] / persisted (0.00/0.00/0.75/0.00/0.25) | 1.51 / 1.13 [0.36, 1.86] / persisted (0.00/0.00/0.75/0.00/0.25) |
| 4 | -0.02 / -0.02 [-0.07, 0.02] / never_read (0.75/0.00/0.00/0.00/0.25) | 1.83 / 1.41 [0.48, 2.34] / persisted (0.00/0.00/0.75/0.00/0.25) | 2.53 / 1.93 [0.64, 3.27] / persisted (0.00/0.00/0.75/0.00/0.25) |

Arm A per-loop trajectory at a = 2, at each loop end from the edited loop on (state diagnostic from the in-batch rows; Ouro loop ends are post-norm, so |dh| there is on another scale):

- L0@1: change in m / D -0.02; state |dh|/|a v| 44.24; cos(dh, v) 0.01; flip at own loop 0.00
- Lmid@1: change in m / D 1.18; state |dh|/|a v| 4.41; cos(dh, v) 0.19; flip at own loop 0.75
- Llast@1: change in m / D 1.13; state |dh|/|a v| 1.00; cos(dh, v) 1.00; flip at own loop 0.50

Clean m_B - m_A at each loop end, logits: 2.28 [1.12, 3.44]; as a fraction of the final gap: 1.00 [1.00, 1.00]

Arm C (O1 at Llast@1, subject positions):

| a | effect | flips | mid fraction [0.2, 0.8] | BC |
|---|---|---|---|---|
| 0.3 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.35 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.4 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.45 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.5 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.55 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.6 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.65 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |
| 0.7 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | - |

### ifm_d28 / arm B (sentiment CAA at Lmid, all positions, 150 prompts)

Site points [[1, 14]]; 160 contrast pairs; |v| [52.21]; residual norm [175.12].

| schedule | vector | a | positive rate [CI] | NLL [CI] | damage agree [undecided] | broken |
|---|---|---|---|---|---|---|
| clean | caa | 0 | - | 1.23 [1.19, 1.27] | 1.00 [7] |  |
| L1 | caa | 0.05 | - | 1.26 [1.22, 1.30] | 1.00 [6] |  |
| L1 | caa | 0.1 | - | 1.31 [1.26, 1.35] | 1.00 [6] |  |
| L1 | caa | 0.2 | - | 1.41 [1.37, 1.45] | 0.95 [7] |  |
| L1 | caa | 0.4 | - | 1.74 [1.68, 1.79] | 0.80 [8] |  |
| L1 | random | 0.05 | - | 1.24 [1.21, 1.29] | 0.95 [8] |  |
| L1 | random | 0.1 | - | 1.27 [1.22, 1.31] | 0.95 [8] |  |
| L1 | random | 0.2 | - | 1.34 [1.30, 1.38] | 0.85 [9] |  |
| L1 | random | 0.4 | - | 1.53 [1.49, 1.58] | 0.70 [12] | yes |

Matched total strength (positive rate; diff = all minus single, paired bootstrap):

| comparison | all-loops | single loop | diff [CI] | damage all / single |
|---|---|---|---|---|

Example (prompt 0): clean caa 0: ' bought a bunch of stuff. I was very happy with my purchase. I was very happy with my purchase. I was' || L1 caa 0.2: " bought a bunch of stuff. I also bought a bunch of stuff that I didn't need. I also bought a bunch" || L1 random 0.2: " bought a bunch of stuff. I was going to take a picture of the store, but I didn't want to take"
