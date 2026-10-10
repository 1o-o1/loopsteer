# K17 FREEZE plan (brief RA-3, matrix card X4a, check C13). Frozen before any scored row.

Written 2026-10-08 by the code chat. The sha256 of this file goes into LEDGER.md (the coordinator writes it).
Nothing in this file changes after the hash. Deviations found later are reported, not edited in.

## 1. Question

R1: the loss of a loop-1 last-token answer edit on Ouro-1.4B is caused by later attention at the last position
(re-reading the source). R1 predicts that the edit is still erased when attention is the only live branch at the
last position. The branch-freeze 2x2 tests this directly on four datasets.

## 2. Frozen predictions (copied from brief RA-3, written by the RA chat before any run)

- Facts (D1): e_attn about 1.4 (CPU pilot 28.09 / 20.3). e_mlp about 0 (CPU pilot 0.07 / 20.3).
- Addition (D2): e_attn and e_mlp both at least 0.7 (the edit persists either way).
- Code and two-step math: no prediction.

## 3. Pass rule (as card X4a)

Per dataset cell, with one-sided 95% bounds (the bounds of the 90% two-sided percentile cluster bootstrap):
- R1 falsified on the cell when the lower bound of e_attn >= 0.5.
- R1 supported on the cell when the upper bound of e_attn <= 0.15.
- Otherwise "mixed".
- The harness identities of section 6 must hold bit-exact on every item of the cell. If one fails, the cell has
  no verdict.

## 4. Model and run configuration

- Ouro-1.4B base (ByteDance/Ouro-1.4B, revision 574fa66), k = 4, bf16, eager attention, batch 1, no cache.
- Hooks: production package loopedcot prod/steer only (EagerOuroAdapter, OuroBinding, HookRegistry,
  interventions.from_spec, interventions.freeze_spec, readouts.Recorder). No new hook code. The script
  loop-flow/src/k17_freeze.py builds the specs and payloads and computes the endpoints.
- Read-out: the loop-j exit read-out, lm_head(LOOP_END_j) at the last position (k16 convention). The loop-4
  read-out must equal the model's own final logits bit for bit (checked per item).
- The last position r = T - 1 (the answer position). Positions: absolute {r} for answer rows, the source span
  for source rows.

## 5. Items

- Facts (D1): the knockout run's pooled_all set: the 80 stored pilot items (k16p_ouro.json) plus every passing
  new item of k17ko_ouro.json, 267 items in 71 A-side clusters (75 subjects on either side). The item identities
  are frozen in items_d1.json, built on CPU from those two files. Prompts are rebuilt by k15_data.d1_candidates
  and matched on (rel, a, b, half); pos, T, tA and tB must equal the stored values.
- Addition (D2): k15_data.d2_candidates (seed 15, 600 raw pairs), scanned in candidate order. Cluster = the
  A-side value of the edited operand ("x=<value>", as k17_varbench). A candidate is skipped (no forward) when its
  cluster already holds 4 kept items. Clean filter: clean A top-1 = tA and clean B top-1 = tB at the loop-4
  exit read-out. Every passing candidate under the cap is kept. Expected at least 240 items. A smaller count is
  reported as it is.
- Code (D4): k16_path_ouro.ext_candidates("D4", "ouro") and k15_data.select with the k16 rule (cap 2 per
  cluster side per (half, kind)) and quota 40 per (half, kind): 80 reads (T1) and 80 sums (T2), dev and test
  halves. The k16 pilot's 20 per quota key are the first 20 of this scan. Cluster = cluster_id (k16).
- Two-step math (D5): the same with quota 80 per half: 160 items. Cluster = cluster_id (k16).
- No tuning: the freezes, edit sites and endpoints are fixed here, so dev and test items are pooled.

## 6. Rows per item (each one batch-1 forward)

1. nohook: clean A through the unhooked model (original layer forward).
2. clean A with recorders: LOOP_END at r for loops 1-4, ATTN_OUT and MLP_OUT at r at every loop call (the
   freeze payloads).
3. clean B with recorders: LOOP_END at all positions for loops 1-4, RESID_POST at loop 1 layer 0 (donors).
4. zero: the answer patch at a = 0 (registry installed, write-back on).
5. E0: answer patch at loop1.loop_end, position r, donor = clean B loop-1 state at r, a = 1.
6. ans_frz_mlp: row 5 plus r's MLP_OUT frozen to its clean-A value at every call with executed index >= F
   (freeze_spec, F = first_changed(loop1.loop_end) = loop 2 layer 0): loops 2-4. Attention live.
7. ans_frz_attn: row 5 plus r's ATTN_OUT frozen the same way. MLP live.
8. frz_mlp: the row-6 freeze without the edit.
9. frz_attn: the row-7 freeze without the edit.
10. src: source-span donor patch at loop1.L0.resid_post (the X3 edit site, as the production X4a config).
11. src_frz_attn: row 10 plus r's ATTN_OUT frozen from F = first_changed(loop1.L0.resid_post).
12. pc_final: patch at loop4.loop_end, position r, donor = clean B loop-4 state at r.
13. sz: patch at loop4.loop_end, source span, donor = clean B loop-4 states at the span.

Identities, bit-exact on every item (float tensors compared with torch.equal):
- hook identity: row 2 logits = row 1 logits = row 4 logits (all positions, all vocabulary).
- read-out identity: the loop-4 exit read-out of row 2 = row 2's final logits at r.
- freeze identity (C13): rows 8 and 9 logits = row 2 logits (all positions).
- harness identity (C13): row 11 final logits at r = row 2's, and LOOP_END at r equal row 2's in every loop,
  so e = 0 exactly. Row 10 must differ from row 2 at r (the identity is not vacuous).
- positive control: row 12 final logits at r = row 3's at r, so dm = D exactly.
- structural zero: row 13 final logits at r = row 2's at r, so dm = 0 exactly.

## 7. Endpoints

- m = z(tB) - z(tA) at the last position. dm(row) = m(row) - m(clean A). D = m(clean B) - m(clean A) at the
  loop-4 read-out.
- e0 = dm(E0) / D. e_attn = dm(ans_frz_mlp) / D (only attention acts on r after loop 1). e_mlp = dm(ans_frz_attn)
  / D (only the MLP acts on r after loop 1).
- Per item ratios, defined where D >= 2.0 (k16 GAP_E). The cell value is the mean over items with the ratio
  defined. The ratio of mean dm to mean D is reported beside it (the CPU pilot's form).
- CIs: percentile cluster bootstrap, 10,000 resamples, seed 0, 90% two-sided (one-sided 95% bounds) and 95%.
  Facts: dyadic (two-way subject, A and B side) bootstrap as k17_ko240.tost_twoway is primary, A-side cluster
  bootstrap reported beside it. Other datasets: A-side cluster bootstrap (the clusters of section 5).
- Every cell reports n_items, n_items with the ratio defined, n_clusters, the strata (D1 capital / city; D4 T1 /
  T2; D2 op1 / op2) and its positive-control value.

## 8. G5 split (PLAN.md amendment G5), descriptive, no pass rule

Per loop end j = 1..4, for E0, ans_frz_mlp and ans_frz_attn, with H the LOOP_END state at r (float32):
- delta_j = H_row_j - H_A_j. target_j = H_B_j - H_A_j.
- (a) norm ratio_j = ||delta_j|| / ||target_j|| (cancellation of the residual difference).
- (b) cos_j = cosine(delta_j, target_j) (rotation away from the target direction).
- (c) e_j = dm_j / D, dm_j at the loop-j exit read-out (reduction of the answer effect).
- At j = 1 the answer rows have delta_1 = target_1, so ratio 1 and cosine 1 (sanity check).
Native accuracy: the freezes without the edit are bit-identical to clean A, so native accuracy is unchanged by
construction. The clean top-1 rates are reported.

## 9. Statistics and cost

About 830 items x 13 forwards plus the D2 / D4 / D5 scans (2 forwards per scanned candidate). Expected under 30
GPU-minutes on one card. Budget check: gpu_minutes_le_90.

## 10. Outputs

Remote out/k17_freeze/ (pulled to loop-flow/artifacts_spark/k17_freeze/): config.json (hashed at launch), the
item identity file, per-dataset selection and item JSONL records, checks.json, summary.md, hashes.json (sha256 of
every output file, written after the run and before the summary). No host or user names in any record.
