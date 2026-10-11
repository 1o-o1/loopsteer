# Completeness matrix (read this before any number)

States: done (every planned item and arm has a record), pending (named what is missing; the cell has no number), skip (reason), done earlier (not rerun on the cluster), not run (scope).

Counts: done 86, done earlier (not rerun) 21, not run (scope) 50, pending 3, skip 161

## Grid A, generation placement

| model | GSM8K_gen | SVAMP_gen | ASDIV_gen | MAWPS_gen |
|---|---|---|---|---|
| Ouro 1.4B base | done | done | done | done |
| Ouro 2.6B base | pending: no results.csv / items_done.json (job not finished or no extractor for this script) | done | done | done |
| Ouro 1.4B Thinking | done | pending: no results.csv / items_done.json (job not finished or no extractor for this script) | done | done |
| Ouro 2.6B Thinking | skip: time: CKPTS and SETTINGS have no ouro_2_6b_think entry | skip: time: CKPTS and SETTINGS have no ouro_2_6b_think entry | skip: time: steer_gen has no Ouro-2.6B-Thinking setting (same as GSM8K and SVAMP) | skip: time: same |
| IFM no-injection | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| IFM injection twin | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| IFM middle pair, no injection | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| IFM middle pair, prelude injection | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| 819M design fixed-r5 | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| 819M design fixed-pln5 | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| 819M design orthoinj | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| d28 | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| d112 | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) | skip: time: steer_gen loads Ouro checkpoints only (SteeredOuroAdapter over prod OuroAdapter) |  |  |
| McLeish r32 | pending: no results.csv / items_done.json (job not finished or no extractor for this script) | skip: time: cluster/pilot/steer_gen_raven.py runs the McLeish path on the GSM8K pairs only | skip: time: steer_gen_math.py runs on McLeish (smoke), but at 19.5 s per problem a full-pool row | skip: time: same |
| Huginn K32 | skip: time: no generation script loads McLeish or Huginn today | skip: time: no generation script loads McLeish or Huginn today |  |  |
| Parcae 370m | skip: time: steer_gen loads Ouro checkpoints only | skip: time: steer_gen loads Ouro checkpoints only |  |  |
| Parcae 1.3b | skip: time: steer_gen loads Ouro checkpoints only | skip: time: steer_gen loads Ouro checkpoints only |  |  |

## Grid A, appendix: loop-2 contrast

| model | GSM8K_gen |
|---|---|
| Ouro 1.4B base | not run (scope): descriptive result |

## Grid A, appendix: LoopCD accuracy rows

| model | GSM8K_gen |
|---|---|
| Ouro 1.4B Thinking | not run (scope): claim dropped (no effect beyond drift) |

## Grid B, GSM8K copy and derived

| model | VB | VBD |
|---|---|---|
| Ouro 1.4B base | done earlier (1,053 pairs (production)) | done |
| Ouro 2.6B base | done earlier (1,069 (production)) | done |
| Ouro 1.4B Thinking | done | done |
| Ouro 2.6B Thinking | done | done |
| IFM no-injection | done | done |
| IFM injection twin | done | done |
| IFM middle pair, no injection | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| IFM middle pair, prelude injection | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| 819M design fixed-r5 | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| 819M design fixed-pln5 | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| 819M design orthoinj | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| d28 | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r | skip: time: registry has ouro x4, ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 only (per-model dicts r |
| d112 | done | done |
| McLeish r32 | done | skip: time: derived rows exist only in k17_varbench, which loads Ouro and three IFM models |
| McLeish non-recurrent twin | done | skip: time: derived rows exist only in k17_varbench, which loads Ouro and three IFM models |
| Huginn K32 | done | skip: time: derived rows exist only in k17_varbench, which loads Ouro and three IFM models |
| Parcae 370m | done | skip: time: derived rows exist only in k17_varbench |
| Parcae 1.3b | done | skip: time: derived rows exist only in k17_varbench |

## Grid B, edit-site map

| model | D1 | CF | D2 | D4 | D5 | LRE |
|---|---|---|---|---|---|---|
| Ouro 1.4B base | done earlier (production, 267) | done earlier | done earlier | done earlier | done | skip: time: no pilot script runs the facts edit-site map on this checkpoint (production ran fact |
| Ouro 2.6B base | done earlier | done earlier | done earlier | done earlier | done | skip: time: no pilot script runs the facts edit-site map on this checkpoint (production ran fact |
| Ouro 1.4B Thinking | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | done | done | done | done |
| Ouro 2.6B Thinking | skip: time: REPOS has ouro, ouro26, ouro_think only | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | skip: time: REPOS has ouro, ouro26, ouro_think only | skip: time: REPOS has ouro, ouro26, ouro_think only | skip: time: REPOS has ouro, ouro26, ouro_think only | skip: time: no pilot script runs the facts edit-site map on this checkpoint (production ran fact |
| IFM no-injection | done | done | done | done | done | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| IFM injection twin | done | done | done | done | skip: ability (the clean-pass scan kept 32 items, fewer than 40) | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| IFM middle pair, no injection | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | done | done | skip: ability (the clean-pass scan kept 2 items, fewer than 40) | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| IFM middle pair, prelude injection | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | done | done | done | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| 819M design fixed-r5 | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| 819M design fixed-pln5 | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | done | skip: time: no edit-site route: k15_pairs.py runs D1 (and D2) only | skip: time: same as D4: no pilot route for two-step math on ifm_p2m_diag | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| 819M design orthoinj | done | skip: time: no listed edit-site script accepts CF for this model (k16_path_ouro / k16_path_ifm:  | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| d28 | skip: time: k15_full.py runs ifm_d28 (k15_smoke.IfmM loads it), but its summarize keeps the mode | skip: time: k15_full.py runs ifm_d28 (k15_smoke.IfmM loads it), but its summarize keeps the mode | skip: time: k15_full.py runs ifm_d28 (k15_smoke.IfmM loads it), but its summarize keeps the mode | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: ARMS has ifm_p1_ouro, ifm_p1_ouro_inj, ifm_p1_mid, ifm_p1_mid_inj, ifm_p2m_diag, ifm | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| d112 | done | done | done | done | done | skip: time: k15_full and k15_pairs build facts in code from the capital and city tables |
| McLeish r32 | done | done | done | done | skip: ability (the clean-pass scan kept 7 items, fewer than 40) | skip: ability (the clean-pass scan kept 26 items, fewer than 40) |
| McLeish non-recurrent twin | done | done | done | done | skip: ability (the clean-pass scan kept 7 items, fewer than 40) | skip: ability (the clean-pass scan kept 26 items, fewer than 40) |
| Huginn K32 | skip: ability (the clean-pass scan kept 32 items, fewer than 40) | done | done | done | skip: ability (the clean-pass scan kept 0 items) | done |
| Parcae 370m | done | skip: time: k17_parcae DATASETS = D1, D2, D4, D5, VB (no CF) | skip: ability (clean filter 0 to 3 percent, ledger 2026-10-08 02:10) | skip: ability (same) | skip: ability (same) | skip: time: data_steer/lre holds no pairs on the Parcae tokenizer |
| Parcae 1.3b | done | skip: time: k17_parcae DATASETS = D1, D2, D4, D5, VB (no CF) | skip: ability (same) | skip: ability (same) | skip: ability (same) | skip: time: data_steer/lre holds no pairs on the Parcae tokenizer |

## Grid B, decay and schedule rows (sentiment)

| model | SENT300 |
|---|---|
| Ouro 1.4B base | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| Ouro 2.6B base | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| Ouro 1.4B Thinking | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| Ouro 2.6B Thinking | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| IFM no-injection | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| McLeish r32 | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |
| Huginn K32 | skip: time: no script holds the 300 new neutral prompts (k17_reread uses k15_data.NEUTRAL150) |

## Grid B, loops against depth

| model | D1 | D2 |
|---|---|---|
| IFM no-injection | done earlier (done on IFM against d112) | done earlier |
| McLeish r32 | skip: time: the manifest asks for an analysis of McLeish against its non-recurrent twin from the | skip: time: same as D1 |

## Grid B, loop trace

| model | D1 | D2 |
|---|---|---|
| Ouro 1.4B base | done earlier | done earlier |
| Ouro 2.6B base | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |

## Grid C, branch-freeze

| model | D1 | D2 | D4 | D5 |
|---|---|---|---|---|
| Ouro 1.4B base | done earlier (k17_freeze full run) | done earlier | done earlier | done earlier |
| Ouro 2.6B base | skip: time: k17_freeze.py --model loads the two Thinking checkpoints since RA-23 item 1 | skip: time: k17_freeze.py --model loads the two Thinking checkpoints since RA-23 item 1 | skip: time: k17_freeze.py --model loads the two Thinking checkpoints since RA-23 item 1 | skip: time: k17_freeze.py --model loads the two Thinking checkpoints since RA-23 item 1 |
| Ouro 1.4B Thinking | done | done | done | skip: time: k17_freeze.py loads both Thinking checkpoints since RA-23 item 1, which runs facts,  |
| Ouro 2.6B Thinking | done | done | done | skip: time: k17_freeze.py loads both Thinking checkpoints since RA-23 item 1, which runs facts,  |
| McLeish r32 | done | done |  |  |
| Huginn K32 | done | done |  |  |
| IFM no-injection | done | done | done |  |

## Grid C, carried-state ablation

| model | D1 | D2 |
|---|---|---|
| Ouro 1.4B base | done earlier (k17_x4e full run) | done earlier |
| Ouro 2.6B base | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| Ouro 1.4B Thinking | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| Ouro 2.6B Thinking | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| IFM no-injection | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| IFM injection twin | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| IFM middle pair, no injection | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| IFM middle pair, prelude injection | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| 819M design fixed-r5 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| 819M design fixed-pln5 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| 819M design orthoinj | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| d28 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| d112 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| McLeish r32 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| McLeish non-recurrent twin | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| Huginn K32 | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| Parcae 370m | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |
| Parcae 1.3b | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) | skip: time: REPO fixed to ByteDance/Ouro-1.4B (no --model flag) |

## Grid C, re-read

| model | NEUTRAL150 |
|---|---|
| Ouro 1.4B base | done earlier (k17_reread full run) |

## Grid D, selective control

| model | D3 | CF |
|---|---|---|
| Ouro 1.4B base | not run (scope): failed as a selectivity claim on four models | not run (scope): failed as a selectivity claim on four models |
| Ouro 2.6B base | not run (scope): failed as a selectivity claim on four models | not run (scope): failed as a selectivity claim on four models |
| IFM no-injection | not run (scope): failed as a selectivity claim on four models | not run (scope): failed as a selectivity claim on four models |
| IFM injection twin | not run (scope): failed as a selectivity claim on four models | not run (scope): failed as a selectivity claim on four models |

## Grid D, site selection

| model | VB | D1 | D2 | D4 | VBD |
|---|---|---|---|---|---|
| Ouro 1.4B base | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| Ouro 2.6B base | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| Ouro 1.4B Thinking | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| Ouro 2.6B Thinking | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| IFM no-injection | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| Parcae 370m | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |
| Parcae 1.3b | not run (scope): never beats the latest-site rule | not run (scope): same | not run (scope): same | not run (scope): same | not run (scope): same |

## Grid D, math specificity readout

| model | VBD |
|---|---|
| Ouro 1.4B base | not run (scope): uninformative as built |

## Grid E, Huginn K16 and K8

| model | D2 | D4 |
|---|---|---|
| Huginn K32 | not run (scope): scope | not run (scope): scope |

## Grid E, MoE screen

| model | D1 |
|---|---|
| Ouro 1.4B base | not run (scope): scope |

## Grid E, edit-rank ablation

| model | D1 |
|---|---|
| Ouro 1.4B base | not run (scope): scope |

## Grid B, strength ablation (edit-site map at a 0.25 and 1, three random seeds)

| model | D1 | CF | D2 | D4 |
|---|---|---|---|---|
| Ouro 1.4B base | skip: time: RA-23 item 3 is not a config: no pilot script runs the Ouro base edit-site map with  | skip: time: same | skip: time: same | skip: time: same |
| Ouro 2.6B base | skip: time: RA-23 item 3 is not a config: no pilot script runs the Ouro base edit-site map with  | skip: time: same | skip: time: same | skip: time: same |

