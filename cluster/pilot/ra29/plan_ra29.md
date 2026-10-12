# RA-29 plan: MCQ late edit, out-of-sample test of the read-schedule rule (Ouro-1.4B and 2.6B base)

Authority: notes/ra_briefs.md RA-29, "RA-29 and RA-30 amendment 1", "Schedule amendment", "Schedule correction"
(binding), PLAN.md B-RA29-31, the RA chat's update of 2026-10-12 (notes/ra_feas_ra29.md), the senior review
review_ra29.md (GO after fixes B1 to B3; N1 to N4 and N7 applied), the RA's P1 change of 2026-10-12, and the RA's
addition after the pilots of 2026-10-12 (RA-29 kept on three datasets; loop-4 quarter arms; main-run order), and
the RA's breadth cells for the cluster (MMLU and HellaSwag on 1.4B and 2.6B; the dataset axis only, no new arm and
no new prediction).
Status: FROZEN. Its sha256 is in plan_ra29.sha256; run.sh passes it and ra29.py stops with exit 2 on a mismatch.
Nothing in this file changes after the hash. Deviations found later are reported, not edited in.
History: plan 960f88cd ran the three pilots (all gates passed); this plan adds two arms and the per-session
identity gate and accepts those pilots (section 7); plan 99994c6e added the two quarter arms; this plan adds the
MMLU and HellaSwag cells (dataset axis only; arms, gates, readout and predictions unchanged).

## 1. Question and predictions (written before any card run)

The placement rule: an edit lands if it sits at a position the later loops read, before their last read, or at the
answer position after the answer has committed. A multiple-choice letter is chosen late, at the answer position.
- P1: the last-token patch lands before the last loop ends: at the end of loop 3 or the middle of loop 4 it gives
  NLD at least 0.5, where facts give at most 0.17 at the same sites, and at the end of loop 1 it gives below 0.2.
  Scored as: max(mean NLD of lt_L3, mean NLD of lt_L4mid) >= 0.5 and mean NLD of lt_L1 < 0.2. The facts value is
  the k15 map's, quoted, not recomputed here.
- P2: the option-block patch at the end of loop 1 lands (mean NLD at least 0.8).
The end-of-loop-4 last-token value (lt_L4) is the readout input; it is reported as the positive-control identity,
not as a result. The loop-4 arms at layer 0, one quarter, the middle and three quarters of the depth (lt_L4first,
lt_L4q1, lt_L4mid, lt_L4q3) locate the commit point inside loop 4; they are descriptive and do not change P1's
scoring. Both outcomes of P1 and P2 are reported in the same sentence form: a held prediction with its number, a
failed one as a negative result.

## 2. Models, pins, harness

- Ouro-1.4B base (registry name ouro_1_4b_base, revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1) and Ouro-2.6B
  base (ouro_2_6b_base, revision 1ed04250da1a9936042725d302e81c8fa2ab5abd), both in the card's cache. K = 4 loops,
  bf16, eager attention, batch 1, no cache, transformers 4.56.2. A pin mismatch stops the run with exit 2
  (ra29.load_ctx).
- Harness of record (the edit-site map harness as RUN 1c runs it), copied unchanged into this folder:
  lib/ = loop-flow/src/run1c/lib (k17_freeze.Ctx, load_model, prod_hashes; k17_ko240.boot_dist), tree/prod =
  run1c/tree/prod (the freeze-pilot md5 pins, checked at load, ra29.PROD_MD5), tests/prod_fakes.py = run1c's CPU
  fake of Ouro. Patches use the production op: interventions.from_spec({"op": "patch", "target": {"kind":
  "donor"}, "site": ..., "positions": {"kind": "abs", ...}}). ra29.py adds no hook.
- Sites. loop<j>.loop_end is the state after loop j (post-norm; the input of loop j+1 and of the exit read-out of
  loop j); at j = 4 it is the input of the final read-out. The loop-4 layer-output sites are built the way the k15
  map's sites.matrix_site builds L0@j and Lmid@j: canonical(Site("loop", 4, layer, RESID_POST)) at layer 0, n_core
  // 4, n_core // 2 and 3 * n_core // 4 (ra29.mid_sites, line 94; L0 and Lmid are asserted equal to matrix_site's
  L0@4 and Lmid@4). Layers 0 / 6 / 12 / 18 of 24 on 1.4B, 0 / 12 / 24 / 36 of 48 on 2.6B. Llast@4 = loop4.loop_end.

## 3. Items (make_data.py; built on the GPU host under the pin; data/sources.json holds every sha256)

- Pools: AQuA test 254, CSQA dev 1,221, ARC-Challenge test 1,172, MMLU 2,000 and HellaSwag 2,000 (the grid's own
  draws) = loopedcot/prod/tasks/data/rows_*.jsonl, joined on row_idx with the grid's k = 4 cells
  (loopedcot/artifacts/cells_<model>_<task>_natural_k4.jsonl, budget 0; present for both models and all five
  datasets). ARC-Easy is not on disk and is dropped (no download).
- MMLU and HellaSwag use the grid's other layout (prod.tasks _COT_LAYOUT): exemplars + "\n\nQ: " + question +
  "\nOptions:\n(A) t1\n(B) t2\n(C) t3\n(D) t4" + "\nA: Let's think step by step." + the cue "\nSo the answer is"
  + " (". The labels are uppercase A to D (single tokens; make_data stores the letters lowercase and the labels in
  meta.labels). MMLU takes one of four exemplar files per row (its prefix_key: stem, humanities, social sciences,
  other); each item names its head and the heads are stored as data/prefix_mmlu__<key>.txt. The block runs from the
  newline before "(A)" (or "(A)" when that newline is merged into the previous token) to the end of option D.
  Every row has four options, so the new gold letter is uniform over a to d.
- Prompt A (the grid's own prompt): prod.tasks.build_prompts base branch = few-shot chain exemplars + "\n\nQ: " +
  input ("... Answer Choices: (a) ... (e) ...") + "\nA:", then the grid's forced cue ids ("\nThe answer is" AQuA,
  "\nSo the answer is" CSQA and ARC), then " (" (one token). The next token is the lowercase letter (a to e are
  single tokens). This is the grid's budget-0 forced read-out; the grid's own answers start " (x).".
  Anchor: the prompt token count equals the grid cell's n_prompt_tokens on every built item (checked).
- Prompt B: one transposition: the gold option's text and one other option's text exchange places. The new gold
  letter is uniform over the letters: items are walked in the order of record (random.Random(29) over the sorted
  keys) and each takes the least-used letter so far among its non-gold letters, ties in a seeded per-item order
  (sha256("ra29:<task>:<key>")). A partner is kept only if T is equal, the block boundaries are equal and every id
  outside the block is equal; else the next letter is tried.
  Reason for a transposition over a random full permutation: it is the smallest edit that moves the gold letter;
  the other options keep their letters, so the A-B difference is the gold letter only.
- Block positions: from the " (" token of "(a)" to the last token of the last option text, by prefix
  tokenization (a token that straddles a boundary drops the item). Answer position r = T - 1 (the " (" token).
- Result of the build: AQuA 250 built (4 dropped, gold text duplicated), CSQA 1,215 (6 dropped, same), ARC 1,172.
  New gold letter counts of the built items: AQuA 50 each, CSQA 243 each, ARC 292/293/292/292 (+3 at e, the
  five-option items). Every built item kept its first partner. Median T: AQuA 628, CSQA 602, ARC 355. Median block
  length 35 / 23 / 36 tokens, of which 4 / 6 / 13 differ position-wise between A and B. The files are the GPU host's
  build under transformers 4.56.2 (the laptop's 5.7 tokenizer splits some non-ASCII characters differently: aqua:37
  and 18 MMLU and HellaSwag items); the 1.4B and 2.6B tokenizer files are identical, so one build serves both
  models. The rebuild with the two new datasets reproduced the three earlier item files byte for byte.
- MMLU: 1,999 built (1 dropped, gold text duplicated); HellaSwag: 2,000 built. New gold letter counts: MMLU
  500/500/499/500, HellaSwag 500 each. Gold letter of prompt A: MMLU 455/471/532/541, HellaSwag 481/488/524/507.
  Every built item kept its first partner. Median prompt tokens (grid): MMLU 460, HellaSwag 787. The grid's
  direct (budget 0) accuracy at four loops: MMLU 0.680 (1.4B) and 0.747 (2.6B), HellaSwag 0.673 and 0.820.
- On the card, before the item loop (review B3): the cue, the " (" token and the letters a to e are re-encoded
  with the card's tokenizer and compared with the stored ids (ra29.check_tail_ids), then every built item is
  re-tokenized and its ids sha256 compared with the stored one (ra29.materialize, line 151). Any mismatch stops
  the run with exit 2 before the first item.

## 4. Screen and drops (every drop logged with its reason in records.jsonl and counted in checks.json)

Before the card (make_data.py): parse_fail, duplicate_gold_text, grid_token_count_mismatch, block_boundary_A,
no_valid_transposition, tokenizer_2_6b_differs. Observed: only duplicate_gold_text (4, 6, 0).
On the card, per item, in this order (ra29.process_item lines 225 to 240):
1. screen_A: the clean A top-1 next token (full vocabulary) is not A's gold letter;
2. screen_B: the clean B top-1 is not B's gold letter;
3. gap_below_2: D = m(B) - m(A) < 2.0 logits, m = z[r, letter_B] - z[r, letter_A] of the final logits.
A kept (scored) item passes all three and runs every arm. The 80-item rule counts scored items, after the gap drop
(review N3). Main-run rule: 2.6B AQuA runs only if the 1.4B AQuA full run scores fewer than 80 items (run.sh
aqua26 reads checks.json field counts.n_kept and prints which way it went). Breadth cells (MMLU, HellaSwag, each
model): the rule applies per cell; run.sh cell runs the cell's pilot, projects the scored count from the pilot's
keep rate over the built items, starts the full run only at 80 or more, and prints the full run's scored count
against 80.

## 5. Arms (one teacher-forced forward each on prompt A; donor = clean B at the same site and positions)

| # | arm | what | role |
|---|---|---|---|
| 1 | lt_L1 | last-token (r) donor patch at the end of loop 1 | default baseline (first row) |
| 2 | blk_L1 | option-block donor patch at the end of loop 1 | the rule's site (P2) |
| 3-4 | lt_L2, lt_L3 | last-token patch at the end of loop 2, 3 | P1 curve |
| 5 | lt_L4first | last-token patch at the output of loop 4's layer 0 (L0@4) | commit point |
| 6 | lt_L4q1 | last-token patch at the output of loop 4's layer n_core // 4 | commit point |
| 7 | lt_L4mid | last-token patch at the output of loop 4's layer n_core // 2 (Lmid@4) | P1, commit point |
| 8 | lt_L4q3 | last-token patch at the output of loop 4's layer 3 * n_core // 4 | commit point |
| 9 | lt_L4 | last-token patch at the end of loop 4 (= readout input) | positive-control identity, not a result |
| 10-12 | blk_L2, blk_L3, blk_L4 | option-block patch at the end of loop 2, 3, 4 | block curve (blk_L4 a structural zero) |
| 13 | persist_lt_L1to4 | last-token patch at the ends of loops 1 to 4 | as briefed; identical to lt_L4 (N1) |
| 14 | rnd_lt_L1 | random same-norm patch at r, end of loop 1 | control |
| 15 | rnd_lt_L4 | random same-norm patch at r, end of loop 4 | control |
| 16 | rnd_blk_L1 | random same-norm patch at the block, end of loop 1 | control |
| 17 | pc | last-token patch at the input of the readout (loop4.loop_end, r) | positive control, must be 1.000 |
| - | swap | clean B | ceiling (NLD 1 by definition) |
| - | clean_A | clean A | floor (NLD 0) |

Plus one zero-strength patch (a = 0 at loop1.loop_end, r) per item for the hook identity. 21 forwards per kept
item (no-hook A, clean A, clean B, zero, 17 arms), 3 per screened-out item, and 8 one-position exit read-outs
(clean per-loop margins; review N6: the loop-4 one-row read-out is a different GEMM shape from the final logits,
so it is not compared with m(A)).
N1. r = T - 1 is the last position, so no other position attends to r. A patch at r at the end of loops 1 to 3
    reaches only r's own stream, and the next patch at r overwrites it. Both persistent arms at r therefore reduce,
    by causality at r, to the last single-loop arm: persist_lt_L1to4 equals lt_L4 at every position (gated
    identity persist14_equals_lt4) and is reported as identical to lt_L4, not as a separate result. (A loops 1 to
    3 persistent arm would equal lt_L3 bit-exact; it is not run.) lt_L4 and pc are the same write at the readout
    input; their final logits equal clean B at r bit-exact (gated).
N2. blk_L4 is a structural zero: nothing reads the loop-4 end state of the block positions (gated identity).
Readout, per arm: NLD = (m(arm) - m(A)) / D. Free decoding, one token: the top-1 token at r of the arm's final
logits (the greedy first token; nothing is decoded past it), classed B's letter / A's letter / other letter / non
letter. The summary prints the share writing B's letter for every arm, and the table of mean NLD by B's gold letter
for lt_L1, blk_L1, lt_L3, lt_L4q1, lt_L4mid and lt_L4q3.

## 6. Technical proofs T1 to T5 (amendment 1), with the code that does each

T1 Cache semantics. The forward is no-cache: model(input_ids, cache_position=arange(T), use_cache=False)
   (tree/prod/steer/adapters/ouro.py line 235; lib/k17_freeze.py line 136 for the no-hook run). A patch at the end
   of loop j at positions P changes the loop-j end state at P only; from loop j+1 on it changes the states and
   K/V entries at P and at every later position (attention), never a position before min(P), never loop j or an
   earlier loop. A patch at a loop-4 layer output at r changes r's stream from that layer on and no other position;
   the loop-4 sites that run before it are unchanged.
   There is no decode step: the one-token readout is the top-1 of the prefix pass's own logits at r, so no cache is
   read after the prefix, and the brief's decoding identity reduces to the hook identity (review N2).
   Proved per item and gated (ra29.process_item):
   - hook identity: hooked clean A run equals the no-hook run bit-exact (line 219 against the clean run);
   - zero identity: an installed patch at a = 0 equals the clean run bit-exact (line 271);
   - before_patch_unchanged: every recorded state of loops before j equals clean A and the loop-j states outside P
     equal clean A; for every last-token patch in loop 4, the loop-4 layer-output sites that run before it (all
     four for a loop-end patch) equal clean A at r (lines 285 to 292);
   - causal_prefix_unchanged: the token before the block is unchanged at every loop in every arm (line 294).
   The two state checks cover the recorded set only: the block positions, r and the token before the block, at
   every loop end, plus the four loop-4 layer outputs at r (review N2).
T2 Positions. Equal T and equal r for A and B, block boundaries equal, ids outside the block equal: built in
   make_data.build_item (lines 170 to 176), asserted again in the tests on every item, re-checked on the card for
   every item before the loop (section 3). Under the pin every item reproduces its stored ids (CPU test
   test_T2_ids_reproduce_the_host_build). Token print: test_T2_tokens_on_cached_tokenizer prints three items per
   dataset, and the card run writes block_maps (three items, A and B block tokens, the r token and the token
   before the block) into checks.json.
T3 Readout. NLD on the letter margin at r of the final logits (ra29.margin_at line 168, nld line 297), D >= 2.0
   (line 240), positive control exactly 1.000 on every item: torch.equal(z_pc[r], z_B[r]) and NLD == 1.0
   (line 311, gated).
T4 Donor. Donor states are clean B's states at the same site and positions (payload HB[j][P] line 252, MB at the
   loop-4 layer outputs line 250), recorded in the same session with the same recorders as clean A. patch_written
   checks that the patched state equals the donor bit-exact at every patched site (line 281).
T5 Random control. Per position p in P: HA[p] + ||HB[p] - HA[p]|| u_p, u_p unit Gaussian rows, one seeded draw per
   item and arm (ra29.rnd_rows line 159, seed sha256("ra29:<model>:<ds>:<key>:<arm>")), built in the model dtype
   before it is written (line 256, the RA-28 lesson); patch_written compares against that bf16 vector. The norm
   ratio ||applied delta|| / ||donor delta|| over P is stored per item and arm; it is not gated (review N1). The
   pilot report prints its min and max per random arm (debug_gate.json field norm_ratio_min_max, summary.md).

## 7. Gates and resume

Per item (exit 4 on any failure on a scored item; the hook identity also on dropped items): hook_identity,
zero_identity, pc_exact, lt4_equals_pc, persist14_equals_lt4, blk4_structural_zero, patch_written,
before_patch_unchanged, causal_prefix_unchanged.
Identity gate, every session (pilot and full, also on resume), before any scored row: the first items of the order
of record are processed until 3 are scored (at most 60 tried); every gated identity must hold on them, else exit 3
with gate.json naming the failures, and nothing is written to records.jsonl (ra29.run, line 604). The gate items
are scored again in the item loop (same deterministic forwards).
Resume (review B2): a run that finds a record with an identity failure in its records.jsonl stops with exit 4
before any new item, so a rerun after exit 4 exits 4 again; a full run also exits 4 after its outputs if any
identity failure is counted. A trailing records line that does not parse (a write cut by SIGKILL) is skipped and
removed (review N7). SIGTERM stops after the current item with exit 5; the same command resumes.
Pilot gate (debug_gate.json, exit 3 on failure): no identity failure, the positive control 1.000 on every scored
item, at least one scored item; screen counts, block maps and the norm-ratio min and max printed. A full run starts
only with that model and dataset's pilot debug_gate.json passed under this plan or under plan 960f88cd
(ra29.ACCEPTED_PILOT_PLANS, line 60; the RA accepted those pilots; plan 99994c6e's pilots are accepted as well),
else exit 3. The two new arms of this plan pass
the identity gate above on the gate items before any scored row.

## 8. Pilots done, main runs, measured cost

Pilots (plan 960f88cd, all gates passed): 1.4B AQuA 30 scored of 98 tried (screen_A 47, screen_B 13, gap 8);
2.6B AQuA 30 of 59; 1.4B CSQA 30 of 60. Measured on the card: 0.236 s per arm and item on 1.4B AQuA, 0.228 on
1.4B CSQA, 0.470 on 2.6B AQuA.
Main runs (run.sh mains, in this order, stops at the first failure):
1. full ouro14 aqua: about 77 scored projected (pilot keep rate 0.31 x 250), about 0.15 card-hours;
2. full ouro14 csqa: about 610 scored, about 0.95 card-hours;
3. pilot ouro14 arc (no ARC pilot exists; 30 scored or 120 tried, about 3 minutes), then full ouro14 arc: about
   0.7 card-hours [belief: T about 355 against 600, no measured ARC rate];
4. aqua26: full ouro26 aqua only if step 1 scores fewer than 80 (projected to run): about 127 scored, about 0.4
   card-hours.
Total about 2.2 card-hours.
Breadth cells (cluster, one job each: `run.sh cell MODEL DATASET` = pilot, 80 rule, full run). Projection at the
measured single-card per-token rate (1.4B 0.38 ms and 2.6B 0.75 ms per prompt token per forward), the median T of the
dataset, and a scored share of about 0.7 times the grid's direct accuracy (the pilots' ratio on 1.4B; 0.8 on 2.6B):
| cell | scored (projected) | single-card (GB10) card-hours | A100 card-hours (single card / 3) |
|---|---|---|---|
| ouro14 mmlu | about 960 | about 1.2 | about 0.4 |
| ouro26 mmlu | about 1,200 | about 2.75 | about 0.9 |
| ouro14 hellaswag | about 940 | about 2.0 | about 0.65 |
| ouro26 hellaswag | about 1,240 | about 4.75 | about 1.6 |
Total about 10.7 single-card card-hours or 3.6 A100 card-hours; every cell is under 6 hours of wall time on one card. The
A100 factor of 3 is a belief (bf16 throughput and memory bandwidth of an A100 against the GB10 single card), not a
measurement; each cell's pilot prints its measured rate and full-run card-hours before the full run. Per scored item at 21 forwards: 1.4B about 5.1 s (AQuA), 4.9 s (CSQA); 2.6B about
10.0 s. Per screened-out item 0.7 s (1.4B) and 1.5 s (2.6B). The full runs add the identity gate (3 scored items,
under a minute). checks.json prints the measured rates and the card-hours of each run.

## 9. Deviations from the brief (recorded before the run)

- D1 (review N4). The brief says the permutation is "seeded from the item key". The new gold letter here also
  depends on the walk order (least-used letter so far), because the RA asked for a new gold letter uniform over the
  letters. The order of record and the per-item tie order are seeded; the build is deterministic.
- D2. Prompt B is one transposition, not a random full permutation (section 3, reason given there).
- D3. The grid's lowercase "(a)" format and the forced cue are used, not the brief's "A to E" labels, by the RA's
  update (the base model was measured in the grid format only).
- D4. ARC-Easy is dropped (not on disk; RA update).
- D5. The briefed persistent arm is identical to lt_L4 (N1); it is run and gated as an identity, not reported as a
  result.
- D6. The main runs use pilots that ran under plan 960f88cd (15 arms); this plan has 17 arms. The RA accepted the
  pilots; the two new arms pass the per-session identity gate before any scored row.
- D7. MMLU and HellaSwag use the grid's chain layout ("Let's think step by step." then the forced cue at budget 0)
  and uppercase labels; the item construction, screen, gap drop, arms and readout are the same as for the first
  three datasets.

## 10. Exact edit list (all inside loop-flow/src/ra29; nothing outside is touched)

| file | function | change |
|---|---|---|
| lib/*, tree/prod/*, tests/prod_fakes.py | - | unchanged copies of run1c/lib, run1c/tree/prod, run1c/tests/prod_fakes.py |
| make_data.py | parse_input, build_input, block_chars, parse_lines, build_lines, block_chars_lines, fmt_fns, prefix_len, build_item, load_cells, main | items, prompt B, block map, drops, letter balance, both option layouts, per-row heads; writes data/ |
| data/items_{aqua,csqa,arc}.json, prefix_*.txt, sources.json | - | written by make_data.py on the GPU host under the pin |
| ra29.py | prefix_files, load_items, materialize, check_tail_ids | items to ids (per-item heads, uppercase labels), tail and item drift checks before the loop (exit 2) |
| ra29.py | mid_layers, mid_sites | the four loop-4 layer-output sites (k15 matrix_site construction) |
| ra29.py | process_item | screen, 17 arms + zero patch through k17_freeze.Ctx.run and interventions.from_spec, letter readout, identities (incl. the earlier loop-4 sites check) |
| ra29.py | rnd_rows, margin_at, writes | T5 directions, letter margin, free-decoding class |
| ra29.py | cells, counts, timing, result_rows, write_outputs, write_summary | bootstrap (k17_ko240.boot_dist, 10,000, seed 0, item clusters), NLD by B's gold letter, P1 and P2 verdicts, outputs |
| ra29.py | load_ctx | KF.load_model with the model's registry name, pins (exit 2) |
| ra29.py | run, summarize, main, parse_args, load_jsonl | plan hash, data hashes, accepted pilot plans, per-session identity gate, resume with the identity check, trailing-line repair, SIGTERM (exit 5), pilot gate, full-run gate |
| tests/test_ra29.py, tests/write_plan_sha.py | - | CPU tests (section 11); plan hash writer |
| run.sh, RUNSHEET.md, plan_ra29.sha256 | - | driver (test, hash, pilot, pilots, full, aqua26, mains, cell, summarize), run sheet, frozen hash |

Outputs per job: out/<model>/<dataset>/{pilot,full}/records.jsonl, results.csv (one row per item and arm),
results.json (means, 90 percent bootstrap, NLD by B's gold letter, the predictions), checks.json (gates, identities,
counts, drops with reasons, block maps, timing), summary.md (arms as rows, default baseline first, the letter-split
table), gate.json, config.json, and debug_gate.json for a pilot.

## 11. CPU tests (tests/test_ra29.py)

Data: files match sources.json; counts, drop reasons and letter balance; every item is one transposition with equal
T, r = T - 1 and the block inside the prompt (both layouts, all five datasets); prompt count equals the grid's;
order is a permutation; T2 ids reproduce the host build (all items under the pin; off the pin only items with a
non-ASCII character may differ); the labels re-encode to the stored letter ids; T2 token print.
Fake Ouro (prod CPU fake, 8 layers, through the real prod hooks): every arm and every gated identity on three items,
21 forwards; pc, lt_L4 = 1.0 and blk_L4 = 0.0 exactly; random norm ratio 1; the loop-4 sites are layers 0, n // 4,
n // 2, 3n // 4 and L0, Lmid equal the k15 matrix_site; dropped loop-end and loop-4 patches are detected
(patch_written); a write at an earlier loop-4 site is detected (before_patch_unchanged); screen and gap drops;
seeded unit random rows; the random payload in bf16 passes patch_written.
Outputs: statistics, results.csv row count and order, NLD by letter, predictions, summary order.
End to end: pilot -> gate -> full -> resume (no new rows) -> summarize, pilot rows reproduce in the full run; the
pilot of plan 960f88cd is accepted and another plan's is refused (exit 3); exit 2 on plan mismatch, on tokenizer
drift and on a tail-id mismatch before any item; exit 3 without the pilot gate and on an identity-gate failure
(no scored row written); exit 4 on an identity failure on a scored item, again on the rerun, and on a full run
over failed records; a trailing partial line is skipped and repaired. Plan hash file matches; P1 text present.
Folder: no host, user or home string.

## 12. Open points

- R1 AQuA on 1.4B is projected at about 77 scored items, under 80; 2.6B AQuA then runs (section 8 step 4).
- R2 lt_L4 and persist_lt_L1to4 carry no result (N1); P1 rests on lt_L3, lt_L4mid and lt_L1.
- R3 Letter bias: the new gold letter is balanced before the screen, not after; the summary prints the kept letter
  counts and NLD by B's gold letter.
- R4 The ARC rate is not measured until its pilot.
