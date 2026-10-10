# K17 FREEZE RAVEN plan (brief RA-19 item 3; RA-3 rows on Huginn-0125 and McLeish r32). Frozen before any scored row.

Written 2026-10-08 by the code chat. The sha256 of this file goes into LEDGER.md before the first scored row.
Nothing in this file changes after the hash. Deviations found later are reported, not edited in.

## 1. Question and condition

RA-3 (notes/k17_freeze_plan.md) tested on Ouro-1.4B whether a loop-1 last-token answer edit is lost through later
attention at the last position (R1). RA-19 item 3 asks for the same rows, identities and pass rule on McLeish r32
and Huginn-0125, on facts and addition, only if the production raven adapter already exposes the MLP and attention
writes. It does [read]: prod/steer/adapters/raven.py transcribes SandwichBlock.forward for both families with one
`reg.site` call at ATTN_OUT and at MLP_OUT of every prelude, core and coda block (`_HUGINN_BLOCK`,
`_MCLEISH_BLOCK`), and prod/steer/interventions.py `freeze_spec` builds the X4a freeze for every family except
IFM. No hook is built. The production package is imported from a frozen copy of the laptop tree (md5 per file in
config.json) and is not edited.

## 2. Frozen predictions (builder, written before any raven freeze row)

Evidence: the RA-1 public core (LEDGER 2026-10-08 16:20): on both raven models an answer state edit at Llast@j for
j <= 16 is erased (e 0.000); Huginn keeps more of a late edit than McLeish.
- P1 E0 (loop-1-end last-token edit alone): e <= 0.05 on both models and both datasets [belief].
- P2 e_attn (MLP writes at r frozen from loop 2 on): upper 90% bound <= 0.15 on both models, both datasets, so the
  frozen rule labels every cell "R1 supported" [belief]. Reading fixed now: on raven models r's state also passes
  through the adapter (state and injected e mixed at every loop) and, on Huginn, through the sandwich norms after
  each write; neither is frozen. A "supported" label therefore says "the edit is erased when the MLP writes are
  clean", not "attention erases it". The adapter refresh is the mechanism the public core already measured.
- P3 e_mlp (attention writes at r frozen): upper bound <= 0.15 on both models and datasets [belief].
- No prediction on the G5 split.

## 3. Pass rule (as RA-3 / card X4a, unchanged)

Per (model, dataset) cell, with one-sided 95% bounds (bounds of the 90% two-sided percentile cluster bootstrap):
R1 falsified when the lower bound of e_attn >= 0.5; R1 supported when the upper bound of e_attn <= 0.15; otherwise
"mixed". The identities of section 6 must hold bit-exact on every item of the cell, else the cell has no verdict.
Facts: the dyadic two-way bound is primary (k17_ko240.tost_twoway), the A-side cluster bound beside it. Addition:
A-side cluster bootstrap.

## 4. Models and run configuration

- Huginn-0125 (prod name huginn_0125, revision bb6621b6) and McLeish r32 (mcleish_llama32_r32, a5f6f126), K = 32,
  bf16, batch 1, no cache. Loader: prod.models.get(name).load() (prod's attention patch as installed by the loader),
  set_depth(32), prod.steer.adapters.bind. Initial state: RavenBinding fixed_table, t_max = the longest kept-or-
  scanned item, row key = the item key, so A, B and every edited row of an item start from the same x0 at every
  position.
- Hooks: production HookRegistry, interventions.from_spec / expand / freeze_spec, readouts.Recorder. The driver
  `loop-flow/src/k17_freeze_raven.py` (new) builds specs, payloads and endpoints and reuses the statistics helpers
  of k17_freeze.py and prep_item of k17_public.py by import (both unchanged).
- Read-out: per-loop exit read-out = RavenBinding.readout(j, LOOP_END_j over all positions, "pure") (coda, ln_f,
  lm_head in the model dtype), taken at r. The loop-32 read-out must equal the model's final logits at r bit for bit.
- r = T - 1, the answer position. Prompt formats, BOS and position-0 rules as the PUBLIC plan DC3: McLeish BOS
  128000 prepended, Huginn no BOS and items whose source span includes position 0 dropped.

## 5. Items

- Facts (D1): every candidate of out/k17_public/data/cands_<tok>_D1.json (tok huginn or llama; sha256 in the run
  config) in file order, clean filter: top-1 tA on A and top-1 tB on B at r (no-hook forward). Every passing
  candidate is kept (full passing pool, as RA-3). Cluster = A-side subject (cl[0]); dyadic units cl[0], cl[1].
- Addition (D2): every candidate of cands_<tok>_D2.json in file order, skipped without a forward when its A-side
  operand cluster (cl[0], "x=<value>") already holds 4 kept items (RA-3 cap); same clean filter.
- Huginn D2 endpoints are leading-digit tokens (PUBLIC plan 1.4); no full-answer filter (RA-3 has none).
- A cell with fewer than 40 kept items is reported and labelled descriptive.

## 6. Rows per item (each one batch-1 forward) and identities

Rows as RA-3 section 6 with two family changes:
1. nohook: clean A through the model with the same x0 passed as input_states, no registry.
2. clean A with recorders (LOOP_END all positions, loops 1-32; ATTN_OUT and MLP_OUT at r at every block call).
3. clean B with recorders (LOOP_END all positions; loop1.L0.resid_post at the source span; the final read-out
   input at all positions).
4. zero (answer patch at a = 0). 5. E0. 6. ans_frz_mlp. 7. ans_frz_attn. 8. frz_mlp. 9. frz_attn. 10. src.
11. src_frz_attn. Freezes = freeze_spec(component, r, F): every call with executed index >= F, including the
   coda calls (production design X4a; Ouro has no coda). F = first_changed(loop1.loop_end) = loop 2 layer 0 for
   the answer rows, first_changed(loop1.L0.resid_post) for row 11.
12. pc_final and 13. sz: patch at the FINAL READ-OUT INPUT (last coda block output), position r (donor clean B) and
   the source span (donor clean B). Family change: on raven the coda attends over positions after the last loop,
   so a loop-32-end patch at r is not a read-out-identity site; the final read-out input is the raven site with
   the property RA-3's loop-4-end site has on Ouro (only norms and the head between site and logits).
Identities, bit-exact (torch.equal), per item: hook identity (row 1 = row 2 = row 4, all positions and vocabulary);
read-out identity (loop-32 read-out of rows 2 and 3 at r = their final logits at r); freeze identity (rows 8, 9 =
row 2); harness identity (row 11 final logits and every LOOP_END at r = row 2's; row 10 differs from row 2 at r);
positive control (row 12 at r = row 3 at r); structural zero (row 13 at r = row 2 at r); the answer rows' loop-1
state at r equals the donor.

## 7. Endpoints, CIs, G5

As RA-3 sections 7 and 8 (m = z(tB) - z(tA) at r; e = dm / D per item where D >= 2; mean over items; 10,000
resamples, seed 0; G5 norm ratio, cosine and e_j per loop end) with loops 1-32. Ratio of means beside each cell.

## 8. Order, cost, outputs

Debug (2 items per dataset per model) into out/k17_freeze/raven/debug/<model>/, then the full run, McLeish first,
then Huginn, one process at a time on the fast host card 1. Expected about 15-40 GPU-minutes per model [belief].
Remote out/k17_freeze/raven/<model>/ (config.json hashed at launch, selection_<ds>.json, items_<ds>.jsonl,
progress.json, hashes.json, checks.json, results.json, summary.md), pulled to
loop-flow/artifacts_spark/k17_freeze/raven/. No host or user names in any record.
