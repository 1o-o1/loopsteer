# RA-30 plan: edit the intermediate value in the chain, with the neuron arm set, Ouro-1.4B base

Authority: loop-signal/notes/ra_briefs.md RA-30, "RA-29 and RA-30 amendment 1" (with the neuron arm set),
"Schedule amendment", "Schedule correction" (binding); PLAN.md B-RA29-31; the RA brief update of 2026-10-12 with
loop-signal/notes/ra_feas_ra30.md; the senior review review_ra30.md (GO after fixes B1 to B4) and the RA's call on
B1 (P4 wording, identities, kv1 replaces kv4); the RA decision after the GSM8K pilot (section 6a); the Ouro-2.6B axis (section 6b); McLeish r32 (section 6c); the summary-time operand flag (6d); the Thinking cells (6e); McLeish ASDiv
(6f); the ASDiv gate waiver (6g). Frozen: the sha256 of this file is in plan_ra30.sha256; run.sh
passes it and ra30.py stops with exit 2 on a mismatch. Nothing here changes after the hash; deviations found later
are reported, not edited in.

## 1. Question and predictions (written before any GPU row)

Does the loop carry a value its own chain has written, or re-derive it from the prompt? We edit the first derived
value of the clean greedy chain at the token positions where it is written, at a loop before the later tokens'
last read of those positions, and decode onward.
- P3: the chain-value patch at the end of loop 1 gives answer_B on at least 0.5 of the text-edit ceiling
  (CA(cv1) / CA(text) >= 0.5).
- P4: CA cv1 >= cv2 >= cv3, the decay of the span patch in Table 2.
- P5: the default baseline (last prompt token patched with prompt B's loop-1 state, the Table 2 setting) gives
  CA 0.00.
- Pre-registered identities (review B1): cvP = cv1 and kv4 = cv3. Both are proved at the key-value-cache level by a
  CPU test; on the card the gate items print the cache equality and the token-identical share (a divergence
  passes only at a near-tie under the G2 rule). cvP is reported as the identity with cv1, not as its own result.
- The summary prints CA(cv1), CA(kv1) and CA(text) side by side: text = cv1 + the loop-1 key-value entries at P
  (review V2), so kv1 reads the loop-1 key-value share of the text edit.
Both outcomes are findings. If the model carries its written value, the text-edit ceiling is high and cv1 nears
it. If it re-derives the value from the prompt at every step, the ceiling itself is low and the chain is not the
carrier (gate G6 stops the dataset and reports this).

## 2. Model, harness, code reuse

- Ouro-1.4B base, revision 574fa66, k = 4, bf16, SDPA, the steer_gen setting a1 (horizon 512 chain tokens, base
  prompt, natural stop). transformers 4.56.2 on the host (exit 2 otherwise).
- tree/prod: an unchanged copy of the cluster bundle's prod tree (byte-identical to ra28/tree/prod). Used
  unchanged: SteerHooks, SteeredOuroAdapter (record_donor, decode; built with kv=True), OuroAdapter.decode (prefill
  once with a StaticUTCache, then greedy at full batch width), tasks.build_prompts, make_find_cut, parse_own,
  strip_tail, unit_dirs, load_tokenizer.
- The decode path is the one anschain_run uses (steer_gen.py 2127-2189): steer.rows[key] with a_ids = prompt A +
  prefix, b_ids = the donor sequence, explicit positions, donor_range, t0, then ad.decode(seqs, n_new, k, eos,
  stop_strings, row_ids=keys), here to the natural stop.
- lib/ra28_score.py and tests/prod_fakes.py: unchanged copies from ra28.

## 3. Datasets and pools

- Main: GSM8K, then ASDiv. Dropped (RA update): SVAMP and MAWPS, reported as "too few multi-step chains".
- Pools: the Table 2 single-card a1 scored lists (steer_gen select on the single card: own answer gold_A on prompt
  A and gold_B on prompt B): GSM8K 395 (cluster_local/ra20_g2 a1_gsm8k/items.json), ASDiv 1,036 (cluster_local/
  ra20_tail2 a1_asdiv/items.json). Table notes name these lists; the cluster counts are not quoted (review R4).
  Pair texts from tree/data_steer, joined on (idx, var_name). Hashes in data/sources.json.
- Order and precompute split (stored, data/split_<ds>.json): the pool ids in random.Random("ra30:split:<ds>")
  order. The split of the dataset-level neuron set is the first 50 eligible items of that order (10 in the
  pilot); the scored items are the eligible items after them. Split items run every arm except ds5 (review R6:
  ASDiv ds5 reads about 40 items; n and interval printed).
- No stored chain is used: every scanned item's clean greedy chain of prompt A is regenerated (batch 1, the same
  decode path) and parsed. Every token id the run uses is built on the host tokenizer at run time; the step
  `tokens` rebuilds all pool prompts on the host tokenizer first (out/tokens, exit 2 above 1 percent mismatches).
  The laptop prescreen files (data/prescreen_*.json, transformers 5.7) are estimates and are never read by a run.

## 4. Screen and drop rules (one reason per item, in screen.jsonl, counted in checks.json)

In order (lib/chain.py analyze and token_geometry; ra30.screen_item):
1. plen_mismatch: prompt A and prompt B rebuilt on the host tokenizer must both have the pool's plen, be equal
   outside the pool span and differ inside it (review B3).
2. clean_wrong: the own answer of the clean chain (prod parse_own on the natural cut) is not gold_A.
3. no_equation (no "<<expr=res>>" annotation before the last "####"; no prose fallback, review R1); one_step.
4. chain_arithmetic_error; last_result_not_answer; expr_parse.
5. Chain-value rule of record (RA update): the derived value is the result of the first equation e_i, i < m, with
   an operand equal to value_A, whose result changes when value_A is replaced by value_B, and that is not the
   final answer; none: no_changed_step. The brief's r_1 rule is computed for every item, information only.
6. no_intermediate (the derived value is a number of prompt A); ambiguous_value (value_A occurs more than once).
7. derived_B_format; token_count (digit count; again on ids: token_count_ids).
8. Consistency: e_(i+1)..e_m re-evaluated with derived_A -> derived_B propagated through equal-valued operands
   (prompt numbers unchanged) must give gold_B, else inconsistent; ambiguous_operand.
9. The first written occurrence of derived_A (after the previous annotation): value_not_in_chain, value_format;
   value_earlier (review B4): derived_A is already written earlier in the chain, or the occurrence lies past e_i's
   annotation.
10. token_boundary; length_mismatch (T2).
No cap. ASDiv under 80 eligible in the full screen is a gate failure (reported, not run further).
Laptop prescreen on the Table 2 single-card chains (estimate): GSM8K 156 of 395, ASDiv 90 of 1,036.

## 5. Construction per item

prefix_A = the clean chain ids up to and including the derived value's tokens (the prefix ends on the value, so
P = [plen + t0, plen + t1) holds the last prefix position). prefix_B = prefix_A with those tokens replaced by the
tokens of derived_B (same count). Example (GSM8K idx 989): "...90/6 = <<90/6=15" -> "...90/6 = <<90/6=10", P =
[630, 632]. P includes unchanged leading digits (here "1"); the donor difference there is numerical noise, so cv
and the random arm act little at those positions (review N9; n_lead_same is counted). Donors (T4): the states of
prompt A + prefix_B, one batch-1 no-cache forward (record_donor), never prompt B's own chain. The scratch arms
(base, span1, ns_*) take prompt B's states as in Table 2.

## 6. Arm table (default baseline first; one decode per arm and item, greedy to the natural stop)

| arm | batch | sequence | edit | role |
|---|---|---|---|---|
| base | scratch | prompt A | post@1, last prompt token, donor prompt B | default baseline (Table 2 r06) |
| text | prefix | A + prefix_B | none | text edit, ceiling |
| floor | prefix | A + prefix_A | none (hooks idle) | unedited floor |
| cv1, cv2, cv3 | prefix | A + prefix_A | post@j at P, donor A + prefix_B | chain-value patch, end of loop j |
| kv1 | prefix | A + prefix_A | kv op, loop 1, at P | loop-1 key-value patch (text = cv1 + kv1) |
| cvP | prefix | A + prefix_A | post@1..4 at P | persistent patch; identity with cv1 |
| rnd1, rnd3 | prefix | A + prefix_A | rand post@1 / post@3 at P, per-position norm of the donor difference | random same-norm |
| nc_full, nc_t1, nc_t5, nc_t20, nc_r5, nc_ds5 | prefix | A + prefix_A | neuron hook, layer 23, loop 1, at P | chain-value neuron arms |
| span1 | scratch | prompt A | post@1 at the prompt span, donor prompt B | span reference (Table 2 r03) |
| clean | scratch | prompt A | none | in-batch reference of the scratch batch |
| ns_full, ns_t1, ns_t5, ns_t20, ns_r5, ns_ds5 | scratch | prompt A | neuron hook, layer 23, loop 1, at the span | span neuron arms |
| kv4 (gate items only) | alone | A + prefix_A | kv op, loop 4, at P | identity with cv3 |

- A post patch at the end of loop 4 reaches no later token (T1), so the ladder is post@1..3; kv4 equals cv3 for
  every later token and runs only as the identity row on the gate items. rnd3 is the matched control of cv3
  (review R7).
- Neuron arms: a = the down_proj input of layer 23 at loop 1 at the edited positions, recorded for the base and the
  donor sequence (batch-1 no-cache forwards, model dtype). The selected neurons take the donor's values exactly,
  all others keep their live values: the arm carries only the last layer's MLP part of the loop-1 state difference
  (review N7; the decode is nonlinear, so nc/cv1 is not capped at 1). Sets per item: full; top 1, 5, 20 percent
  (56, 282, 1,126 of 5,632) by the L2 norm over positions of d = a_donor - a_A; random 5 percent (282,
  random_subset seeded by "ra30", model, dataset, setting, item); ds5 = the top 5 percent by mean |d| over the
  split, scored items only. Sets stored (sets.json, items_done.jsonl).
- Batching: per item, one prefix batch of 15 rows and one scratch batch of 9 rows, equal lengths, no padding. A
  batch runs at full width until its longest row stops (review V4), so arms are compared inside their batch.
- Keys and state (review B2): row keys "<ds>:<idx>|<arm>"; steer.rows, steer.donors, steer.stats and the neuron
  specs are cleared after every item; the prefix donor is recorded once from the kv1 row (so it holds K/V) and
  shared by the prefix rows; the scratch donor once from the base row.

## 6a. RA decision after the GSM8K pilot (pilot of record: plan 7f96eb63, all gates passed)

Pilot GSM8K (35 items): base 0.00, text 0.94, cv1 0.86, cv2 0.60, cv3 0.40, kv1 0.91, all 12 neuron arms 0.000.
- The neuron arms (nc_*, ns_*) are not run in the mains. The code stays; the recorded setting is ra30.py
  RUN_NEURON_ARMS = False (written into config.json). The batches become 9 prefix rows (text, floor, cv1, cv2, cv3,
  kv1, cvP, rnd1, rnd3) and 3 scratch rows (base, span1, clean). With no neuron arm there is no precompute split:
  every eligible item is scored.
- Neuron-hook non-vacuity check (ra30.neuron_nonvacuity), on the card on the first gate item of each full run:
  nc_full and ns_full applied once; printed: the norm of the applied change of the down_proj input at the edited
  positions and of the MLP output, whether the next-token logits differ from the idle pass (torch.equal and the
  max absolute difference), and whether a equals the donor. The summary writes "inert hook" if the change is 0 or
  the logits are bit-equal, otherwise "no effect of the last-layer MLP part". The pilot's zero CA is not a result.
- The full run accepts the GSM8K pilot of plan 7f96eb63 (ACCEPTED_PILOT_PLANS) and reads from it the projected
  eligible count (the 80-item gate) and the horizon cap decision (use the cap: 0 of 819 parsed rows past it).
- ASDiv pilot: 20 scored items, no split. The full ASDiv run starts only if the pilot's projected eligible count is
  at least 80; otherwise it prints "ASDiv under 80: gate failure" and exits 3.
- Card order (run.sh mains, stops at the first failure): full gsm8k, tokens asdiv, pilot asdiv, full asdiv.
- Examples: examples.json always holds idx 11 of GSM8K (the paper's example box) before the other items.
- Measured on the pilot (24 arms): clean decode 9.0 s per scanned item, 26.7 s per armed item (1.1 s per arm and
  item), eligible rate 0.389 (projected 154 of 395).

## 6b. Ouro-2.6B axis, GSM8K and ASDiv (overflow jobs; the model axis is the only change)

- Model: Ouro-2.6B base, revision 1ed04250 (prod model_revisions.json; same modeling md5 pin as 1.4B, 48 layers),
  bf16, k = 4, the same 12 rows, gates, readouts, cap rule and predictions. No new arm. ra30.py --model
  ouro_2_6b_base; the default stays ouro_1_4b_base.
- Pools: the Table 2 single-card a2 scored lists, built by make_data.py --model ouro_2_6b_base: GSM8K
  (cluster_local/ra20_g2 a2_gsm8k/items.json, 425 problems) and ASDiv (cluster_local/ra20_tail2 a2_asdiv/items.json,
  1,059 problems; the select step of that single-card run finished, its rows did not, and only the scored list is
  used). Files data/pool_<ds>_ouro_2_6b_base.json, their own orders (data/split_<ds>_ouro_2_6b_base.json, key
  "ra30:split:<ds>:ouro_2_6b_base") and one sources file (data/sources_ouro_2_6b_base.json). The 1.4B data files and sources.json are unchanged. The 2.6B tokenizer.json
  is byte-identical to 1.4B's; token ids are still rebuilt on the host tokenizer (run.sh tokens2, CPU only, no
  card visible).
- Paths: out/ouro_2_6b_base/{tokens,pilot,full}/<ds>. The 1.4B paths stay out/{tokens,pilot,full}/<ds>.
- Steps per dataset: tokens2 <ds>, pilot2 <ds> (20 scored items), full2 <ds> (reads the 2.6B pilot of that
  dataset: plan accepted, the 80-item gate on its projected eligible count, the cap decision). ASDiv on 2.6B is a
  breadth cell: same code and arms, no new prediction. The 1.4B steps, including mains, are those of plan ed722548; the GSM8K pilot of 7f96eb63 and
  any pilot under ed722548 or 352bd818 stay accepted (ACCEPTED_PILOT_PLANS).
- Estimate [belief: the single-card a2 batch-1 job bw1_r00 ran 0.138 s per token with the load, 1.32 times the
  1.4B rate; the eligible rate assumed equal to the 1.4B pilot's 0.389]: clean decode about 12 s, armed item about
  32 s. Pilot2 about 52 scanned x 12 s + 20 x 32 s + 20 gate items x 33 s = about 0.55 card-hours. Full2 425 x 12 s
  (1.4 h) + about 165 x 32 s (1.5 h) + gates = about 3.0 card-hours (worst about 4.5 if rows run to the horizon).
  ASDiv [belief: the 1.4B ASDiv clean decode about 6 s times 1.32, the eligible rate of the 1.4B laptop prescreen,
  0.087]: pilot2 about 230 scanned x 8 s + 20 x 32 s + 20 gate items x 33 s = about 0.9 card-hours; full2 1,059 x
  8 s (2.4 h) + about 92 x 32 s (0.8 h) + gates = about 3.2 card-hours. The projected eligible count (about 92)
  sits near the 80-item gate.

## 6c. McLeish r32, GSM8K (PI breadth cell; its own arm set and pre-registered predictions)

- Model: smcleish/Recurrent-Llama-3.2-train-recurrence-32, revision a5f6f126, K = 32 recurrences (trained depth),
  prelude 4, core 6, coda 4 blocks; the core input of every loop is adapter(state, e) with e the prelude output of
  the position's own token (input re-injection). Decode, hooks and x0 come from the cluster bundle's McLeish
  generation path, lib/steer_gen_raven.py (unchanged copy; sha256 a862710f...): RavenSteerHooks and
  SteeredMcLeishAdapter, the fixed per-item x0 table (seed 1000003 x problem id, as in Table 2). Driver
  ra30_mcl.py (ra30.py, lib/chain.py and lib/ra30_data.py are unchanged); the only subclass maps a row key
  (problem id x 100 + arm number) to its problem's x0 table, so every arm, donor and the screen share it.
- Pool: the steer_gen_raven McLeish GSM8K scored list, 237 problems (cluster bundle pilot/data/mcleish_setting/
  items.json = cluster_local/g2_gen/mcleish/items.json, sha256 0b6a8ee2...), spans and plen counting the BOS;
  data/pool_gsm8k_mcleish_llama32_r32.json, its own order (key "ra30:split:gsm8k:mcleish_llama32_r32") and
  data/sources_mcleish_llama32_r32.json. Prompt: G.main's McLeish encoding (BOS + the GSM8K exemplar prompt).
  Horizon 512, stops "

Question:" and "Final Answer:", eos = the tokenizer eos plus the adapter's stop ids.
- Parser: McLeish writes GSM8K-format chains like Ouro ("60 - 20 = <<60-20=40>>40 miles ... #### 25"); the same
  lib/chain.py rules apply (annotation equations, the first-changed rule, consistency, first occurrence). The
  Llama tokenizer writes numbers in chunks of up to three digits ("1250" = "125" + "0") with the space as its own
  token; equal digit count gives equal token count, and token_boundary and token_count_ids check every item.
  Chained results ("<<4*1.5+2*4+20=6+8+20=34>>") do not parse and drop as chain_arithmetic_error. One added rule
  for McLeish only: value_is_operand, when the derived value also equals an operand of its own equation (the first
  written copy can then be the operand, "4 / 2 = 2"). The Ouro runs keep their frozen rules; on the Ouro laptop
  prescreen this case is 2 of 158 GSM8K items (idx 321, 480; P sits on the operand) and 0 of 90 ASDiv items,
  reported as a deviation of those runs.
- Arms (default baseline first; prefix batch of 8 rows, scratch batch of 3 rows):
  | arm | batch | edit |
  |---|---|---|
  | base | scratch | post@1 at the last prompt token, donor prompt B |
  | text | prefix | prompt A + prefix_B |
  | floor | prefix | prompt A + prefix_A |
  | cv1 | prefix | post@1 at P, donor prompt A + prefix_B |
  | kv1 | prefix | kv op, loop 1, at P (the core blocks' K and V before rotary) |
  | rnd1 | prefix | random same-norm at post@1 at P |
  | cvP | prefix | post@j at P for every loop j = 1..32 |
  | kvP | prefix | kv op at P for loop 0 (the prelude) and every loop 1..32 (the hook has no coda site) |
  | eP | prefix | prelude output e at P = the donor's (site pre_out, loop 0); interpretation control (review B2) |
  | span1 | scratch | post@1 at the prompt span, donor prompt B |
  | clean | scratch | prompt A |
- T1 (McLeish cache semantics, proven on the CPU fake of McLeish): the raven static cache holds one K/V entry per
  block call (prelude 0..3, core 4 + 6 (j - 1) + l, coda the last 4 entries); later tokens read the entry of the
  same call. post@1 at P changes the entries of loops 2..32 and of the coda at P and later positions, never the
  prelude or loop 1 entries, never earlier positions. kv1 replaces the loop-1 core entries at P with the donor
  sequence's own entries. kvP replaces the prelude and every core loop's entries at P with the donor's; the coda
  K/V at P are mixed (computed from a residual that attended to the donor's core K/V). cvP is not the text edit at
  the cache level: e is re-injected at every loop from A's own token at P, so the core entries at P carry A's e,
  the prelude entries at P are A's, and only the coda entries and the read-out at P are the donor's. eP: e at P is
  the donor's, so the core and coda entries at P equal the donor sequence's and only the prelude entries stay A's.
  Reading (an interpretation control, not a new prediction): eP close to text removes the token channel through the
  prelude K/V at P; then cvP below eP is the input-refresh effect.
- Pre-registered predictions (written before any McLeish row):
  - M1: CA(cv1) is about 0, because the next loop refreshes from the input.
  - M2: CA(cvP) gives most of the text-edit ceiling.
  - M3: CA(kv1) is about 0.
  "About 0" = CA <= 0.05 with its 90 percent upper bound <= 0.10. "Most" = CA(cvP)/CA(text) >= 0.67 with the 90
  percent lower bound of the ratio >= 0.5 (paired bootstrap). Printed beside them: CA(kvP)/CA(text),
  CA(eP)/CA(text), CA(cvP)/CA(eP), CA(base).
- Gates (ra30_mcl.gates_item, on the first 10 eligible items in the pilot, 3 in the full run): G1 idle hooks
  against removed hooks (logits torch.equal), G2 the floor alone against the clean chain (tie rule), G3 the
  teacher-forced prefix, G4 positions, G5 cv1 and kv1 land on a fresh donor record; on the first gate item also
  every loop of cvP (post@1..32), kvP (K/V of loops 0..32) and eP (e at P) lands (review N2). G6: CA(text) >= 0.5
  on the pilot.
- RA D1 (breadth cell): the 80-item gate is waived for McLeish. The full run starts with an accepted McLeish pilot
  (this plan, plan 086581bf of push 2, or ACCEPTED_PILOT_PLANS), at least 20 projected eligible items (else "McLeish GSM8K under 20: gate
  failure", exit 3) and the pilot's cap decision (ra30_mcl.pilot_of_record). checks.json prints gate_80 as
  "waived" and gate_n20; the summary marks the cell as breadth and prints n with every interval.
- RA D2 (shards): the full run is split by items: shard s of N takes the order positions s, s + N, ... (order[s::N]);
  each shard runs its own gates on its first 3 eligible items, the first of them with the all-loop G5 check;
  summarize_mcl full merges the shard folders and stops if an item appears in two shards. N = 3 if the pilot's
  projected_full_card_hours <= 16.2 (3 x 9 h x 0.6), else ceil(projected / 5.4) (review N6); run.sh full_mcl S N.
- The full run screens every pool item again in its own folders, the pilot's about 62 items included (about 1
  card-hour, review N8).
- Pilot: 20 scored items, no split. Paths out/mcleish_llama32_r32/{tokens,pilot,full}/gsm8k; run.sh tokens_mcl
  (CPU only, no card), pilot_mcl, full_mcl, summarize_mcl.
- Estimate [belief: the Table 2 McLeish r00 job ran about 1.1 s per decode step at batch 16; batch 1 taken as
  0.85 s; mean clean chain 70 tokens on the stored chains; eligible rate 0.325 on the laptop prescreen of the
  stored chains (77 of 237)]: clean decode about 60 s, armed item about 180 s, gate item about 75 s. Pilot about
  62 scanned x 60 s + 20 x 180 s + 10 x 75 s = about 2.2 card-hours. Full 237 x 60 s (4.0 h) + about 77 x 180 s
  (3.9 h) = about 7.9 card-hours, 3 shards of about 2.6 card-hours (about 4.4 h at 0.6x, plus about 10 min of
  load and gates). The 8th prefix row (eP) adds little: a batch runs to its longest row.

## 6d. Operand flag at summary time (RA decision; the Ouro runs)

value_is_operand (section 6c) is a run-time drop for McLeish only. For the Ouro runs (1.4B and 2.6B, including the
runs of the frozen plan 3cf3fc98) it is a flag computed at summary time by ra30_flags.py (run.sh flags, flags2):
the step re-parses every scored item's stored clean chain with the run's own parser and rules, stops with exit 4 if
the re-parse does not reproduce the stored derived value and equation, flags the items whose derived value equals
an operand of its own equation, and writes operand_flags.json and summary_operand.md with CA of every arm on all
items and without the flagged items. No run file and no frozen code changes. On the 1.4B GSM8K pilot it flags 0 of
35 items; the laptop prescreen of the stored chains found 2 of 158 GSM8K and 0 of 90 ASDiv items.
Audit wording limits (notes/ra_audit_ra30_gsm8k.md W2, W10), for every RA-30 summary: the same step
(ra30_flags.notes, run by run.sh flags and flags2, and called by the McLeish and Thinking summaries) writes
summary_notes.md and notes.json. It never calls cvP = cv1 or kv4 = cv3 an identity: it prints "cvP equals cv1 token
for token on X of N chains (CA difference d, 90 percent interval)", the kv4 = cv3 token-identical share with "kv4
ran on the G gate items only (no CA)" (or "kv4 was not run in this cell"), and the carried-value floor (the items
whose derived_B is a number of prompt A, and the carry-B share that this alone explains in unedited arms). The
Thinking summary drops ra30.summarize's identity line and appends the notes. On the 1.4B GSM8K full run: cvP equals
cv1 on 134 of 151 chains (CA difference +0.020, 0.007 to 0.040); kv4 equals cv3 on 4 of 5 gate items; derived_B is
a prompt number on 6 of 151 items, a carry-B share of 0.040, as observed in floor, clean and base.

## 6e. Ouro Thinking cells: 1.4B-Thinking and 2.6B-Thinking, GSM8K and ASDiv (PI grid; gold-intermediate matching)

- Models: ByteDance/Ouro-1.4B-Thinking 3aaa2224 and Ouro-2.6B-Thinking f1edd81e, k = 4, bf16, loaded from their
  checkpoint folder (K17_THINK_DIR) by prod's SteeredOuroAdapter (kv=True); the steer_gen Thinking setting a3 / a4:
  chat template with the exemplars in the user turn, the think tag not a stop, the task's stop strings
  ("\n\nQuestion:", "Final Answer:") and eos, horizon 4096, the Table 2 parser (parse_own reading after the
  think tag). Driver ra30_think.py with lib/gold.py. ra30.py, lib/chain.py, lib/arms.py and
  lib/ra30_data.py are unchanged.
- Pools: the Table 2 Thinking scored lists, on the cluster only, read at run time from OUT_ROOT:
  A/steer_gen/a3_gsm8k, A/steer_gen/a4_gsm8k, A/steer_gen_math/a3_asdiv, A/steer_gen_math/a4_asdiv (items.json),
  joined to the pairs files for the question texts. A missing file stops with exit 2. The pilot records the file's
  sha256 (pool_record.json); every full shard stops with exit 2 when the file differs (the sha256 pin). The chat
  geometry (prompt ids, the changed number's token span) is recomputed on the host tokenizer and must equal the
  pool's span and plen (else plen_mismatch). Order: random.Random("ra30:split:<ds>:<model>") over the pool ids.
- Matching (no equation parsing; Thinking chains are prose with no annotations): the expected intermediates come
  from the gold solution of problem A and of the changed problem B. GSM8K: the GSM8K test solution of A
  (data/gold_gsm8k.json, openai/gsm8k 740312a main/test, 1,319 solutions, its annotations); B's values are the same
  equations with value_A -> value_B in the literals and every changed earlier result propagated; the last result
  must equal gold_A and gold_B. ASDiv: equation_A and equation_B of the pair (the same tree), the inner node values;
  the roots must equal gold_A and gold_B. Candidates: intermediates that change from A to B, are not a number of
  question A and are not the final answer. P = the first number span of the clean chain (the cut text, think block
  included) whose value equals a candidate's A value; derived_B = that candidate's B value, written like A's
  occurrence (plain digits or thousands commas, same length). Rules kept: the token-count rule (equal length, then
  token_count_ids and token_boundary on ids) and the consistency rule (the clean chain's own answer is gold_A). Drop
  reasons: plen_mismatch, clean_wrong, no_gold_solution, gold_parse, gold_inconsistent, no_changed_intermediate,
  value_not_in_chain, value_format, derived_B_format, token_count, token_boundary, token_count_ids,
  length_mismatch. Carry (prose): the first number after the prefix equal to derived_A or derived_B.
- Arms, gates, readouts: the 12 rows of the Ouro mains (section 6a) and the same predictions P3 to P5 read as
  breadth; gates G1 to G5 of section 9 (no identity decodes) on 10 pilot items and 3 items per full shard; G6
  CA(text) >= 0.5 on the pilot. The full shards read the pilot: plan accepted, G6 passed (else exit 3), the same
  pool file (sha256), at least 20 projected eligible items (breadth cell, no 80-item gate), the cap decision of
  section 10 on the 4096 horizon.
- RA decision (tonight's unit): pilots only, no Thinking full job. The full runs come later, only for a cell whose
  G6 passes; the full-shard code (think_full) stays. GSM8K pilots (1.4B and 2.6B Thinking) are never dropped and
  scan until 30 eligible items or the end of the pool; the ASDiv pilots keep 20 eligible items and may be dropped by
  the planner. Every chain is stored (screen.jsonl clean chains, rows.jsonl every arm).
- Summary of a Thinking pilot: CA(text), CA(cv1), CA(kv1) and CA(span1) with 90 percent intervals, and rewrite_back
  per prefix arm: the share of chains that, after P, write value_A again (the original value of the edited step)
  before their answer marker. Computed per row from the continuation after the prefix (ra30_think.rewrite_back).
- Pilot jobs (run.sh think_pilot): a screen job (phase screen), then N parallel rows jobs over disjoint eligible
  items (phase rows, part p of N takes eligible items p, p + N, ...; each runs its share of the 10 gate items,
  ceil(10 / N)), then a CPU merge job (phase merge) that joins the parts, checks that every part finished, and
  computes G6 and the summary on the merged rows. N = 2 for 1.4B-Thinking GSM8K and 2.6B-Thinking ASDiv, 3 for
  2.6B-Thinking GSM8K, 1 for 1.4B-Thinking ASDiv, so every job stays at or under about 4.5 h at 0.6x.
- Laptop smoke (section 6e smoke; not a run of record): smoke_think.py on a local RTX 4070 (8 GB), Ouro-1.4B-Thinking,
  8 problems of data/pool_gsm8k.json (the 1.4B base pool, chat geometry recomputed), horizon 768, sub-batches of 2
  rows (the 8 GB card cannot hold a 9-row batch); outputs out/smoke_think_laptop/. Result (before the hash):
  4 of 8 eligible (drops: clean_wrong 2 at the 768-token cut, token_count 2); P sits on the result of the matched
  step in every eligible chain ("125 + 50 = $175" -> 191, "50 * 3 = 150" -> 273); every gate passes on both armed
  items (G1, G3, G4, G5 exact; G2 by the tie rule, divergence at margins 0.25 and 0.125); cvP = cv1 token-identical
  on 1 of 2. On both armed items the text edit still ends in answer A (the model re-checks the step and writes the
  correct value), span1 ends in B and every other arm in A. Risk: the Thinking pilots may stop at G6 (ceiling
  under 0.5); that is then the cell's result ("the Thinking chain is not the carrier").
- Estimate [beliefs: single-card batch-1 rate 0.095 s per token on 1.4B (the RA-30 pilot), 1.32 times that on
  2.6B; Thinking chain length about 950 tokens on GSM8K (stored 1.4B-Thinking chains at k = 2: mean 944, median
  625, many at the 4096 horizon) and about 500 on ASDiv; a 9-row batch runs to its longest row, about 3,000 tokens;
  per item: clean decode 90 s (GSM8K) and 48 s (ASDiv), armed item 540 s and 300 s, gate item 100 s and 60 s on
  1.4B, times 1.32 on 2.6B; eligible rate 0.45 on GSM8K (gold candidates on 294 of 395 problems) and 0.13 on ASDiv
  (220 of 1,036)]. Pilot jobs, hours at 0.6x (single-card hours / 0.6):
  | cell | screen | rows parts (each) | merge |
  |---|---|---|---|
  | 1.4B-Thinking GSM8K (30 items, about 67 scanned) | 2.8 h | 2 x 4.0 h | CPU, minutes |
  | 2.6B-Thinking GSM8K (30 items) | 3.7 h | 3 x 3.5 h | CPU, minutes |
  | 1.4B-Thinking ASDiv (20 items, about 154 scanned) | 3.4 h | 1 x 3.1 h | CPU, minutes |
  | 2.6B-Thinking ASDiv (20 items) | 4.5 h | 2 x 2.0 h | CPU, minutes |
  Later full runs (only where G6 passes), single card / at 0.6x: 1.4B GSM8K 27.5 h / 45.8 h, 2.6B GSM8K 30.7 h /
  51.2 h, 1.4B ASDiv 24.1 h / 40.2 h, 2.6B ASDiv 31.8 h / 53.0 h, in shards of at most 9 h at 0.6x. These
  numbers are uncertain by about a factor of 2 (Thinking chain lengths at k = 4 and the eligible rates are
  unmeasured); the pilots replace them.

## 6f. McLeish r32 ASDiv (PI grid; the section 6c arms, predictions, gates, D1 and D2)

- Pool: the EXT5 priority-2 McLeish ASDiv screen and select output on the cluster, OUT_ROOT/A/steer_gen_math/
  mcleish_asdiv/items.json, written by job g18_fillgen.s3_select.002 (steer_gen_math.py --dataset asdiv mcleish
  select --setting mcl --n-screened 290). No McLeish ASDiv pool exists on this machine or in the bundle setting
  folders. The cell chains behind that job (afterok). Read at run time, joined to pairs_asdiv for the texts; a
  missing file stops with exit 2; the pilot records its sha256 and the full shards stop when it differs. Prompt:
  McLeish's encoding of the SVAMP-registered ASDiv prompt (the GSM8K exemplars, BOS first). Order:
  random.Random("ra30:split:asdiv:mcleish_llama32_r32") over the pool ids.
- Everything else is section 6c: arms (8 prefix rows with eP, 3 scratch rows), M1 to M3 with their thresholds as
  breadth readings, value_is_operand at run time, gates, D1 (n >= 20 projected eligible) and D2 (shards, merge).
  run.sh mcl_asdiv_tokens, mcl_asdiv_pilot, mcl_asdiv_full S N, mcl_asdiv_summarize.
- Estimate [beliefs: McLeish ASDiv chains about 40 tokens, 0.85 s per decode step at batch 1; the select keeps
  about 250 problems; eligible rate about 0.1 (ASDiv is mostly one-step)]: pilot about 200 scanned x 35 s + 20 x
  120 s + 10 gate items = about 2.8 h single card, 4.6 h at 0.6x; full about 250 x 35 s + 25 x 120 s = about 3.2 h,
  3 shards of about 1.1 h (1.8 h at 0.6x). The projected eligible count may fall under 20 (then the full run stops
  with "McLeish ASDiv under 20: gate failure").

## 6g. ASDiv: the 80-item gate is waived (RA decision, one rule for every ASDiv cell)

Every ASDiv cell (Ouro-1.4B and Ouro-2.6B through ra30.py, McLeish through ra30_mcl.py, the Thinking pilots through
ra30_think.py) runs in full on whatever is eligible, with n >= 20 projected eligible items as the only floor
(ra30.GATE_MIN_BY_DS; "ASDiv under 20: gate failure", exit 3, below it), prints n with every interval, and carries
the note "mostly one-step problems; under the 80 rule" in summary.md and checks.json (note_asdiv). GSM8K keeps the
80-item gate unchanged (a test proves that a GSM8K pilot projecting 79 still stops with exit 3). The 1.4B ASDiv
pilot of the card (plan 352bd818, out/pilot/asdiv: 20 scored items, projected 65 eligible, G6 CA(text) 0.85) is
the pilot of record for the 1.4B ASDiv full run (352bd818 is in ACCEPTED_PILOT_PLANS).

## 7. Readouts and statistics

Per item and arm, from the stored chain (prefix included; prod cut and parse_own): outcome B (CA), A, or neither
(break, no parse included); carry = the first complete annotation after the prefix (after the prompt for scratch
arms) with an operand equal to derived_A or derived_B: B, A or none; chain length, horizon hit, answer-marker token
index, first divergence from the in-batch floor (or clean). Shares with a 90 percent percentile bootstrap over
items (10,000 resamples, seed 0). Ratios (P3: CA(cv1)/CA(text), CA(kv1)/CA(text); neuron share: CA(nc_s)/CA(cv1),
CA(ns_s)/CA(span1)) by the paired bootstrap of the ratio of sums; differences (P4: cv1 - cv2, cv2 - cv3; neuron gap
t5 - r5) by the paired bootstrap of the mean difference. Also printed per arm (reviews N4, N8): CA on the items
where the in-batch floor and clean rows both end in answer A; CA without items where gold_B is a number of prompt A
or prefix_A; carry without items where derived_B is a prompt number. Tables print base first, then text and floor.

## 8. Technical proofs (T1 to T5)

- T1 Cache semantics. A post patch at the end of loop j at P (steer_gen _post_norm, _apply lerp) is the input of
  loop j+1: it changes states and key-value entries of loops j+1..4 at P and later positions, never loops 1..j and
  never earlier positions. K/V entries live at cache index current_ut * L + layer of the StaticUTCache the prefill
  fills (models/ouro.py 379-382); decoding reuses it unchanged (ouro.py 439). A post patch at loop 4 writes no
  entry (it moves only its own position's read-out). kv1 writes the loop-1 K/V at P equal to the donor sequence's;
  kv4 the loop-4 K/V. CPU-proven on the fake: cache semantics, the post@4 structural zero, kv1 and kv4 against the
  donor, prefix pass + decode reproduce the clean chain, idle hooks bit-exact, the identities cvP = cv1 and
  kv4 = cv3 at the cache level (float32 and bf16), the full 15-row and 9-row batches equal single rows. Card: G1,
  G2, G2b and the identity checks.
- T2 Positions: token_geometry (P from per-token decode offsets, tokens decode to the digits, equal lengths, the
  prefix ends on P). Card: G4 prints three items per dataset.
- T3 Readout: ra30_data.cut_parse = G.main's strip_tail, make_find_cut, parse_own (the Table 2 parser); outcome
  and carry from the stored chain text; chains stored for every item and arm (rows.jsonl).
- T4 Donor: b_ids = prompt A + prefix_B, donor_range = P; the patched loop-j state at P equals the donor bit-exact
  (CPU in float32 and bf16). Card: G5 against a fresh record_donor on every gate item (review B2).
- T5 Random: steer_gen rand op, per position the norm of donor minus current at the same site and loop, direction
  seeded by (item key, site, position). Two printed ratios (review N5): the pre-rounding ratio from steer.stats
  (float32, before the bf16 cast) and, on the gate items, the applied bf16 delta pooled over P against cv's, with a
  tolerance of 1e-2 (reported, not a stop).

## 9. Gates

Every session, on the first gate items of the scored list (20 in the pilot, 5 in the full run), before any scored
row (exit 3 on failure, no scored row written):
- G1 hook identity: prefill logits and every K/V entry of A + prefix_A with the floor row registered and idle
  equal those with no row keys (torch.equal).
- G2 the floor decoded alone at batch 1 reproduces the clean chain after the prefix token by token; a divergence
  passes only at a top-2 logit margin below 1.0 at the divergent step. G2b (reported): the in-batch floor against
  the floor alone, same rule.
- G3 teacher-forced prefix: argmax at every prefix position equals the next prefix token (same tie rule).
- G4 positions printed; equal lengths; P ends the prefix.
- G5 post@1..3 at P equal the fresh donor bit-exact; kv1's loop-1 K/V outputs at P equal the fresh donor's; the
  full neuron set's a at P equals the donor's.
- Identities printed: cache equality and token-identical share of cvP against cv1 and kv4 against cv3.
- G6 (pilot, after the rows): CA(text) >= 0.5 on the scored pilot items, else the dataset stops and is reported.
Decision rules (RA, after each pilot report): a dataset needs >= 80 eligible items (pilot projection, then the full
screen; ASDiv under 80 is a gate failure); a setting's neuron arms are dropped if its full-MLP arm carries less than
0.1 of its full-state effect (nc_full/cv1, ns_full/span1); an arm flat within its interval across its levels is
dropped. Exit 2: plan sha256, pin, data hash, host-token or input mismatch.

## 10. Horizon cap rule (pre-registered, review N2)

The pilot keeps 512 tokens (the clean screen and Table 2 a1 use it) and logs, per row with a parsed answer, the
token index of the answer marker. If fewer than 1 percent of those rows place it after min(512 - prefix,
2 x the clean chain's continuation after the prefix + 64) (prefix 0 for the scratch batch), the full run uses that
cap per item and batch (ra30.cap_new); otherwise 512. The pilot's checks.json prints the decision and the full run
reads it into config.json.

## 11. Pilot and estimates (review N1 corrected)

- Pilot, card order GSM8K then ASDiv: per dataset, scan the order until 10 split + 25 scored eligible items; every
  scanned item gets its clean decode. Report: tables, gates, dropped counts with reasons, 5 stored chains per
  dataset (clean, text, cv1, kv1), seconds per arm and item, eligible rate, projected full card-hours, the cap
  decision.
- Rate [belief, single-card batch-1 job bw1_r00 of a1_gsm8k: 248.1 s for 20 problems and 2,376 tokens, 0.104 s per
  token with the load]: clean decode about 10 s (GSM8K) and 6 s (ASDiv); per armed item about 54 s (two batches of
  about 200 tokens and 4 s of records), about 2.3 s per arm and item; about 110 s when a batch runs to the horizon,
  which a single broken row causes (review V4), so the worst case is common. Gate items about 25 s each.
- Pilot: GSM8K 90 x 10 s + 35 x 54 s + 20 x 25 s = about 55 min; ASDiv 390 x 6 s + 35 x 54 s + 20 x 25 s = about
  80 min; about 2.2 card-hours.
- Full: GSM8K 395 x 10 s (1.1 h) + 160 x 54 s (2.4 h) = about 3.5 card-hours (worst about 6.0); ASDiv 1,036 x 6 s
  (1.7 h) + 95 x 54 s (1.4 h) = about 3.1 (worst about 4.6). The pilot's measured rate replaces these; GSM8K stops
  for report if its projection exceeds 4 hours.
- After the GSM8K pilot (section 6a; 12 arms, the cap on): GSM8K full 395 x 9.0 s (1.0 h) + 154 x about 24 s
  (1.0 h) + gates = about 2.1 card-hours; ASDiv pilot about 230 scanned x 6 s + 20 x 24 s + 20 gate items x 25 s =
  about 0.65 h; ASDiv full 1,036 x 6 s (1.7 h) + about 90 x 24 s (0.6 h) = about 2.3 h; mains about 5.1 card-hours.

## 12. Outputs

out/tokens/<ds>.jsonl and <ds>_check.json (host tokens). out/<pilot|full>/<ds>/: config.json, hash_line.json
(plan, config, code, data sha256, before the first row; a changed input on resume stops with exit 2),
screen.jsonl (every scanned item), sets.json (split ids and dataset-level sets), gates.json (and one per session),
rows.jsonl (item x arm: chain ids, cut text, own, outcome, carry, lengths, edit stats), items_done.jsonl (per item:
seconds, pre-rounding norm ratios, neuron sets, G2b), g6.json (pilot), results.csv, results.json, checks.json,
summary.md, examples.json (5 in the pilot, 20 in the full run). No host, user or home string in any file.

## 13. Exact edit list

No reused file is edited. New files in loop-flow/src/ra30:
- lib/chain.py: equations, safe_eval, substitute, analyze ("first_changed" of record, "r1" information; drops incl.
  value_earlier and plen_mismatch; fields derived_B_in_prompt, gold_B_in_prompt_or_prefix, n_lead_same),
  token_geometry, outcome, carry.
- lib/arms.py: ARMS (15 prefix, 9 scratch), IDENTITY_ARMS (kv4), IDENTITIES, DONOR_ROW, key_of, steer_edits (patch,
  kv, rand with rkey = item key), row_info (asserts prompt B against prompt A for scratch rows), NeuronHook,
  record_acts, neuron set functions.
- lib/ra30_data.py: DATASETS, pools, draw_order, prompt_ids, cut_parse.
- make_data.py (run on the laptop; writes no token-id file the run reads).
- ra30.py: load_ctx, clear_item, screen_item, geom_of, acts_for, dataset_sets, build_rows, readout, decode_batch,
  run_item, prefill, gates_item, tie_check, host_tokens, run, summarize, main.
- tests/test_ra30.py, tests/test_ra30_driver.py (and tests/test_ra30_mcl.py: 54 tests, 48 pass and 6 float32-only skips in the bf16 class).
- run.sh (test, hash, tokens, pilot, full, summarize, mains; the ra28 layout), RUNSHEET.md, plan_ra30.sha256.
- Section 6a additions: RUN_NEURON_ARMS, ACTIVE_ARMS, ACCEPTED_PILOT_PLANS, EXAMPLE_IDX, pilot_of_record,
  neuron_nonvacuity, N_SCORED_PILOT per dataset.
- Section 6b additions: ra30.py --model, set_model, model_root, pool_of, order_of, n_scored_pilot,
  N_SCORED_PILOT_MODEL, check_data per model; lib/ra30_data.py model_suffix, sources_name, order_key and an
  optional model argument of load_pool, load_split, draw_order (defaults give the 1.4B files); make_data.py
  --model (build_model, one sources file per model); run.sh tokens2, pilot2, full2, summarize2 for gsm8k and asdiv.
- Section 6d: ra30_flags.py, tests/test_ra30_flags.py, run.sh flags and flags2.
- Section 6e: ra30_think.py, lib/gold.py, data/gold_gsm8k.json and data/sources_think.json (make_data.py --gold),
  smoke_think.py, tests/test_ra30_think.py, run.sh think_tokens, think_pilot, think_full, think_summarize.
- Section 6f: ra30_mcl.py --ds asdiv (set_ds, get_pool, the pool pin in pilot_of_record), run.sh mcl_asdiv_*.
- Section 6c additions: ra30_mcl.py (McLeish driver), lib/steer_gen_raven.py (unchanged copy of the bundle file),
  tests/test_ra30_mcl.py, make_data.py --model mcleish_llama32_r32, run.sh tokens_mcl, pilot_mcl, full_mcl,
  summarize_mcl.

## 14. Open risks

- R1 Annotation-only parsing and the chain_arithmetic_error drop keep GSM8K near 156 and ASDiv near 90 (estimate);
  no fallback (review decision). ASDiv under 80 on the fresh chains is reported as a gate failure.
- R2 The text edit leaves visibly wrong arithmetic ("90/6=10"); G6 decides per dataset.
- R3 Eligibility comes from the batch-1 clean chain, the readouts from the batches; N4's subset CA is printed.
- R5 Tokenizer drift: the host step `tokens` and the plen_mismatch drop cover it.
- R8 Time: a batch runs to its longest row; the worst-case card-hours apply if random rows run long.
- R9 On the card the identities hold at the token level (pilot: cvP = cv1 0.95, kv4 = cv3 0.90 token-identical on
  20 gate items) but not bit-exact at the cache level (kv_equal share 0.0): the donor comes from a no-cache bf16
  forward and the patched run from the cached prefill, so the replaced states differ in the last bits. kv4 = cv3
  covers the K/V entries only: the first token after the prefix is read from P's own loop-4 state, which cv3
  changes and kv4 does not (review V1); in 1 of 20 gate items this flipped that token (idx 639, margin 6.8).
  Reported, not a gate.
