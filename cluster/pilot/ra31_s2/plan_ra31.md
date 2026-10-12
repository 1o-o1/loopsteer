# RA-31 plan: scaled writes, the dose curve of the branch writes, whole write and unit subsets, Ouro base

Authority: notes/ra_briefs.md, RA-31 (2026-10-12), the RA-29/30 amendment 1 (T1 to T5), the schedule amendment and
the binding schedule correction; PLAN.md B-RA29-31; the RA chat's five decisions of 2026-10-12 on
notes/ra_feas_ra31.md (sections 3 and 9 below state where each is applied); the senior review review_ra31.md ("GO after
fixes": B1 to B3 and N1 to N8 applied below), the coordinator's Part B changes, and the RA addition after the pilot
(2026-10-12: size sets top1 / top20 with random sets, the attribution store, the size curve; sections 5, 8a, 8). The
pilot ran under the plan before that addition (sha256 ae5405db...7b92), passed every gate (out/pilot_1.4b) and is
accepted as the pilot gate of the full run (ra31.ACCEPTED_PILOT_PLANS); the new arms pass G1 on the gate items before
any scored row. Re-frozen after the addition: its sha256 is in plan_ra31.sha256; nothing changes after the hash.

## 1. Question and decision

Run 1c held one point of a curve: the last prompt token's MLP write in loops 2 to 4 at its clean value (alpha 0)
restores a loop-1 patch that the normal run (alpha 1) erases. RA-31 measures the whole curve. Part A (this card,
main line): interpolated write w' = w_clean + alpha (w - w_clean), loop-1 last-token patch present, teacher-forced,
NLD at the readout. Part B (cluster overflow, lower priority): scaled write w' = alpha w on the clean run, free
greedy decoding on the GSM8K pool of Table 2, accuracy against alpha. alpha in {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}.

Why the test is well posed either way: if the restoring write is a threshold mechanism, NLD stays high until a
critical alpha and then drops; if it is linear in the amount of recomputation, NLD falls on the chord between alpha 0
and alpha 1. Concentrated (top 5 percent carries most of the drop) and spread (top close to random) are both
findings, and the random set of equal count separates them.

## 2. The write, and why subsets are defined on the post-norm write (decision 2)

Ouro ends each branch in an RMSNorm before the residual add: attention w_att = input_layernorm_2(o_proj(z)), MLP
w_mlp = post_attention_layernorm_2(down_proj(a)) (modeling_ouro.py 414 and 421; prod adapters/ouro.py 361-372, the
sites attn_out and mlp_out). The write is the post-norm output. Scaling neuron activations or head outputs before
the projection is wrong: the norm rescales the whole vector, and scaling all units by alpha is a no-op for alpha > 0.
With u = W x (no bias, checked for down_proj and o_proj) and rms(u) = sqrt(mean(u^2) + eps):
  c_S = g * (W[:, S] x_S) / rms(u),  exact, and the c_S of a partition sum to w.
Subset edit (interpolated, Part A): w' = w + (alpha - 1) (c_S - c_S_clean), c_S from the live run's own x and
rms(u), c_S_clean from clean A's own x and rms(u). Units outside S stay at alpha 1. At S = every unit, c_S = w up to
rounding, so the edit reduces to the whole-write interpolation; the "dall" arms print that gap (section 6, G4).
Subset edit (scaled, Part B): w' = w + (alpha - 1) c_S. Heads: S = the o_proj input columns of the heads, W = o_proj.

## 3. Ops per arm (decision 1)

| arm | op | code |
|---|---|---|
| Part A whole write ("all"), every alpha | prod patch edit, torch.lerp, a = 1 - alpha, target clean, sites loop*.L*.mlp_out or attn_out, calls from F_ans, positions abs [r] (freeze_spec with a replaced) | ra31.process_item, frz_iv |
| Part A alpha 0, "all" | the same spec at a = 1: lerp returns the stored clean write exactly = the run 1c hold | as above |
| Part A alpha 1, "all" | a = 0: _edit_patch returns x untouched | as above |
| Part A "top", "rand", "dall" | RA-31 subset edit at the norm output, w + (alpha - 1)(c_S - c_S_clean), float32, cast once | lib/ra31_hooks.Units31._norm_edit |
| Part A donor and random patch at loop1.loop_end | prod patch edit a = 1 (run 1c, unchanged); the random arms change only alpha | process_item, piv |
| Part B whole write | new exact scale op y = (w.float() * alpha).to(dtype), forward hooks on the two post-branch norms | Phase 3, ra31_gen.py |
| Part B subsets | w + (alpha - 1) c_S at the same norm hooks; inputs from down_proj and o_proj pre-hooks | Phase 3 |

## 4. Models, items, dropped-item rules (decision 4)

- Ouro-1.4B base first: revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1, k = 4, bf16, eager, batch 1, no cache,
  transformers 4.56.2, prod md5 pins of run 1c (ra31.load_model checks the run1c.py pin constants; exit 2 on any
  pin difference).
  Items: the run 1c sets, facts 267 (data/items_d1.json via KF.build_d1) and addition 263 (data/ouro_D2.json via
  run1c.build_items), rebuilt on the tokenizer, exact match or exit 2.
- Ouro-2.6B base second (registry ouro_2_6b_base, 48 layers, same OuroDecoderLayer source hash; revision read from
  the prod pins file at load and written into hash_line.json; no download, HF_HUB_OFFLINE=1). Items drawn with the
  same selection code, never the 1.4B files: facts KF.build_d1_pool (every d1_candidates pair) then the 2.6B clean
  filter KF.make_know (A top-1 = tA and B top-1 = tB); addition: the build_scan / select_d2 rule on
  k15_data.d2_candidates (cap 4 per A-side operand value, same filter), written out in ra31.build_items so the pilot
  can stop at 25 kept items. Counts reported; no padding.
- r = T - 1. Prompt A and B must have equal length (checked in build_items; run 1c items satisfy it).
- Dropped and counted with the reason (selection_<ds>.json field drops): unequal prompt length of A and B (both
  models; review N3), 2.6B clean filter fail on the A side or the B side; at the readout, clean gap D < 2.0 (NLD
  undefined; in run 1c all 530 items pass). Not drops: rebuild mismatch (exit 2), identity failure (exit 4).
- Pilot items: the first 25 of each dataset in the stored order (1.4B) or in candidate order (2.6B, the scan stops at
  25 kept items, a prefix of the full selection).

## 5. Rows and arms per item

Base rows (10 forwards): no-hook A; clean A with recorders (loop ends, the 192 branch writes at r, Units records of
a, z, u at r); clean B; zero patch (a = 0); MLP hold and attention hold without the patch (run 1c freeze_no_edit);
positive control (loop4.loop_end at r from clean B); structural zero (loop4.loop_end at the source span); donor patch
E0 with records; random patch with records (run 1c seeds "run1c:<ds>:<key>", payload built in bf16).

Sets per item and patch: "all"; "top" = the top 5 percent of the L x I neurons (6,758 on 1.4B, 13,517 on 2.6B) by
sum over loops 2 to 4 of |a_patched - a_clean| at r, and the top 2 of the L x H heads by the sum of the L2 norm of
the head-output difference (z, o_proj input); stable sort, ties to the lower index; "rand" = a seeded draw of the
same count ("ra31:neurons:<ds>:<key>", "ra31:heads:<ds>:<key>"), the same draw for the donor and the random patch.
The top set of the random patch comes from the random-patched run. One mask for all three loops. Records store the
top sets only; the random sets are rebuilt from the seed (ra31.full_sets; review N7).

Size sets (RA addition, MLP and donor only): the same ranking as "top" (sum over loops 2 to 4 of |a_patched -
a_clean|), cut at 1 percent ("top1", 1,352 neurons on 1.4B) and 20 percent ("top20", 27,034); top1 within top within
top20 (prefixes of one rank order). "rand1" and "rand20" are seeded draws of the same counts ("ra31:neurons1:<ds>:<key>",
"ra31:neurons20:<ds>:<key>"). The rand sets are kept because the forward count stays at or under 84 (83).

Arm table (each arm one forward; 72 per item; the default baseline, the last-token patch at loop 1 alone, is row 1):

| # | patch | branch | set | alphas | forwards |
|---|---|---|---|---|---|
| 1 | donor, patch alone (E0 = alpha 1) | - | - | - | base |
| 2 | donor | MLP, attention | all (whole write) | 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2 | 16 |
| 3 | donor | MLP, attention | top | same 8 | 16 |
| 4 | donor | MLP, attention | rand | same 8 | 16 |
| 5 | donor | MLP, attention | dall (decomposed, every unit; reduction check) | 0, 2 | 4 |
| 6 | random same-norm | MLP, attention | all, top, rand | 0, 2 | 12 |
| 7 | donor | MLP | top1, top20, rand1, rand20 (RA addition) | 0, 2 | 8 |

alpha 1 rows of all, top and rand are run because they carry the identity of each op path (the prod lerp and the
decomposed edit). The dall alpha-1 arms of the pilot are dropped: they ran the same edit code as top and rand at
alpha 1 (also proven bit-exact by the N1 test), and dropping them keeps the rand1 / rand20 sets inside 84 forwards.
Per item: 10 base rows + 72 arms + 1 attribution pass = 83 forwards (85 forward-equivalents, the backward counted as
2; section 11).

## 6. Gates and identities (decision 5)

G1 identity gate, every session, first 3 items per dataset, before any scored row (exit 3, no scored row). The keys
that exit 3 (gate.json field exit3_keys; review N2) are the eight bit-exact identities (torch.equal) and two info
checks:
- hook_identity: no-hook A = registry A with Units recording = zero patch (a = 0);
- freeze_no_edit_identity: each hold without the patch = clean A;
- pc_exact: the positive-control row = clean B at r; sz_exact: the structural-zero row = clean A at r;
- donor_loop1_state: E0's loop-1 state at r = clean B's; rnd_loop1_state: the random row's = the bf16 payload;
- alpha1_equals_patched: every donor alpha-1 arm, all sets and branches, full logits = E0;
- set_counts: top and rand sets of the planned size for both patches;
- first_site_scale (info): the whole-write arm at the first call of its branch in loop 2, projection ratio
  ((w' - w_c) . D) / ||D||^2 with D = w_E0 - w_c, within 0.02 of alpha (review B2);
- sets_act (info): every donor alpha-0 arm changes the logits.
G2 scored rows: any of the eight identities failing stops the run with exit 4 after the record is written.
G3 run 1c reproduction (1.4B; the pilot decision rule "alpha 1 reproduces the normal run and alpha 0 matches run 1c"):
per dataset and branch, mean |NLD(all, alpha 0) - NLD(run 1c hold)| <= 0.005 against data/run1c_ref.json (per-item
values of the run 1c full run). Printed beside it: max |gap| and the number of items with a nonzero gap (on the same
card and stack it should be 0; a nonzero count points to a stack difference; review N4), and the same for E0 and for
the random arm at alpha 0 against rnd_hold_*. Miss: exit 3 after the records are written; the full run is blocked.
G4 reduction (review B1): dall - all = (alpha - 1)[(c_all - w) - (c_all_clean - w_clean)] is unbiased bf16 noise of
about one ulp at each of 72 sites, the same size at alpha 0 and 2; a code error (W, g, eps, rms, keys, clean records)
gives a systematic gap. Per dataset and branch at alpha 0: |mean signed gap| <= 0.01 (bias) and mean |gap| <= 0.03
(noise ceiling). Printed: max |gap| and the alpha-2 mean signed, mean and max |gap|; the alpha-2 gap is the noise
floor for reading top and rand above alpha 1. Miss: exit 3 at the end; the RA decides whether the full run drops the
subset arms (--drop) or stops.
G5 budget: the full run starts only if the pilot gate passed and the pilot's seconds per item, scaled to the full
run's forwards per item and item count (gate items twice), is at most 1.5 card-hours on 1.4B (5.5 on 2.6B, raised from 3.5 on 2026-10-12 so a cluster card at 0.8 x the single card's speed fits); else
exit 3 (NO FIT).
Exit codes: 2 plan, pin, data or resume mismatch; 3 a gate (G1, G3, G4, G5, missing or failed pilot gate); 4 an
identity on a scored row; 5 SIGTERM (resumable).

## 7. T1 to T5 of amendment 1, where they apply

T1 Cache semantics. Part A: not applicable as a decode test. Every forward is binding.forward with use_cache=False
   and cache_position = arange(T) (prod adapters/ouro.py 225-235); the edit at r in loops 2 to 4 changes r's later
   states and r's keys and values in those loops inside the same forward, and the readout is that forward's final
   logits at r. The Units31 hooks refuse any call whose length is not T (_check_T). Part B: applies (section 10).
T2 Positions. r = T - 1 for A and B; equal length asserted per item; the pilot prints the tokens at r and at the
   source span of A and B for 3 items per dataset, and the subset masks for the same items (mask_report: neurons
   per layer for top and rand, the (layer, head) pairs, the top-rand overlap).
T3 Readout. m = z(tB) - z(tA) at r of the final logits, D = m(clean B) - m(clean A), NLD = (m(row) - m(clean A)) / D
   for D >= 2. Positive control exactly 1.000 and structural zero exactly 0.000 per item (G1 bit-exact rows).
T4 Donor. The donor is clean B's loop1.loop_end state at r (run 1c), checked bit-exact in the E0 row.
T5 Random control. The run 1c random patch: H_A + ||H_B - H_A|| u, u from run1c.rnd_direction with the run 1c seeds,
   built in bf16 before any equality check; the applied-to-target norm ratio printed per item (1.000 expected). The
   random unit sets are seeded per item, count equal to the top set (asserted).
Also printed per alpha (brief): the norm of the changed write, per branch and loop 2 to 4, sqrt(sum over layers of
||w_arm - w_clean||^2) at r.

## 8. Readouts, tables, predictions

Per model, dataset, patch, branch and set: alpha as rows, mean NLD [90 percent percentile cluster bootstrap, 10,000
resamples, seed 0] (items / clusters), facts also the two-way subject bound; the write norm per loop. Derived, with
the paired cluster bootstrap of run 1c (boot_ratio):
- share(set, alpha) = sum(NLD_set - NLD_E0) / sum(NLD_all - NLD_E0) for top and rand at alpha != 1, printed where
  |mean(NLD_all - NLD_E0)| >= 0.1; at alpha 0 and 2 also with dall as the denominator, so both sides use the same op
  (review N8);
- top - rand per alpha (paired, per item);
- chord deviation of the all curve at alpha 0.25, 0.5 and 0.75 from the line between alpha 0 and alpha 1.
- size curve (MLP, donor, alpha 0 and 2): NLD and share of the all-set effect for top1, top5, top20 and all, beside
  rand1, rand and rand20 (RA addition).
Curves to results.csv (one row per curve point) for the figure script.

Prediction (verbatim from the brief, before the run): "the all-neuron MLP curve falls from about 1.28 at alpha 0 to
0.00 by alpha 1 and stays at 0 above; the top-5-percent curve falls less than the all-neuron curve by at most half of
the gap if the restoring write is spread, by more if it is concentrated; the attention curves stay flat near the
patched-alone value at every alpha; random under alpha 0 reproduces 0.57 and 0.63."

Numeric readings, fixed before the run (review B3), on both datasets:
- Endpoints: alpha 0 (the run 1c hold: facts 1.28, addition 1.41) and alpha 1 (E0: facts 0.00, addition 0.81) are
  fixed by identity and are not a test.
- Shape (MLP and attention, all set): the chord deviation at 0.25, 0.5 and 0.75. "linear" if fewer than two of the
  three 90 percent intervals exclude 0; "threshold" only if all three exclude 0 with the same sign; otherwise "not
  linear" (review N5).
- "Stays above" (MLP): for alpha in {1.25, 1.5, 2}, NLD(all) - NLD(E0) has a 90 percent interval inside [-0.1, 0.1].
- Attention flat: at every alpha, NLD(attention all) - NLD(E0) has a 90 percent interval inside [-0.1, 0.1].
- Concentration (MLP; a correction of the brief's clause, which is inverted: a concentrated write makes top close to
  all): concentrated = share(top, 0) >= 0.5 and the 90 percent interval of top - rand at alpha 0 excludes 0; spread =
  share(top, 0) < 0.5; otherwise "neither". The summary also prints share(top, 0) with its sign and interval; the
  label follows this rule, and the sign is the finding (a negative share means the top set's write works against the
  restoration).
- Random at alpha 0 (MLP all): printed beside run 1c's 0.5664 (facts) and 0.626 (addition); on 1.4B the same items,
  seeds and op give the same per-item values (G3 prints the gap).
Prune rule (proposed, review N6; the RA decides at the pilot): keep the attention all arm in every case (flat
attention is the prediction); if it is flat by the rule above, drop attention top and rand from the full run
(--drop attn:top,attn:rand; about 25 percent of card time). summary.md prints the proposal.

## 8a. Attribution store (RA addition)

Per item, on the patched run E0, for every neuron i of the top-20-percent set and each loop t in 2 to 4 at r:
  est(t, l, i) = grad_w(t, l) . (c_i_clean(t, l) - c_i(t, l)),
grad_w(t, l) = d margin / d w(t, l), the gradient of m = z(tB) - z(tA) at r of the final logits with respect to the
post-norm MLP write of layer l in loop t at r (zero leaves added to that write in one forward under enable_grad,
then one backward); c_i = g * W_down[:, i] a_i / rms(u) is the neuron's exact contribution to the write (section 2),
c_i from E0's own a and rms, c_i_clean from clean A's. est is the first-order predicted margin change from holding
neuron i at its clean value in loop t alone; the sum over the set and loops is the first-order prediction for the
top20 alpha-0 arm. Computed as (W_down^T (g * grad_w)) * (a_C / rms_C - a_P / rms_P), all neurons at once.
Stored per dataset in attr_<ds>.npz (packed by --summarize from per-item shards attr_<ds>/<n>.npz): keys [N]; idx
int32 [N, n20], the flat neuron index l * I + i in rank order (top1 and top5 are its prefixes); est float16 [N, 3,
n20] (loops 2, 3, 4); scalars float32 [N, 5] = D, margin of clean A, margin of E0, margin of the top20 alpha-0 arm
(measured), summed estimate. Raw size 530 x (27,034 x 10 + 20) bytes = 143.3 MB on 1.4B. The record keeps
attr.pred, attr.measured (margin top20 alpha 0 minus margin E0) and attr.grad_pass_logits_equal_E0 (info: the grad
pass reproduces E0's logits). summary.md prints mean predicted against mean measured, the Pearson r over items and
the mean absolute error relative to the mean absolute measured change. The CPU test shows the estimate equals the
finite difference for single neurons and for the whole set at small step; at alpha 0 the set is a large step, so
the measured value, not the estimate, is the result, and the gap is the size of the nonlinearity.

## 9. Exact edit list

New, owned by RA-31 (nothing outside loop-flow/src/ra31 is touched):
1. lib/ra31_hooks.py: Spec31 (edit_mlp, edit_attn, alpha, T); rms; contrib (c_S); edit_row (w + (alpha - 1)(c - c0));
   head_cols; Units31(ra28_hooks.Units): install (refuses RA-28 holds, scaling, gradients and blocks; adds forward
   hooks on post_attention_layernorm_2 and input_layernorm_2), _in_pre / _down_pre / _o_pre (record x at r, live input
   for the edit), _check_T, _norm_edit.
2. ra31.py (Phase 1 part, written): constants, arm_name, arm_list, seed_of, n_top_neurons, neuron_scores,
   head_scores, top_units, rand_units, item_sets, mask_report, edit_tables, Runner (the RA-28 pattern with Units31),
   write_norms, process_item, nld, run1c_gap, curve.
3. ra31.py (Phase 2, written): parse_args (--model 1.4b|2.6b, --pilot N, --pilot_gate, --drop), code_hashes,
   check_data, write_hash_line, load_model (KF.load_model with the run 1c pins: 1.4B revision 574fa66c, transformers,
   eager, bf16, prod md5; the 2.6B revision recorded), build_items (section 4, length drops counted), load_ref,
   gate_check (G1), g3_repro (G3), g4_reduction (G4), check_pilot_gate and projected_hours (G5), t2_report, run
   (pilot / full, resume, SIGTERM, progress, exit codes 2 to 5), diff_cell, share, inside, excludes0, ds_cells,
   shape_reading, concentration_reading, predictions, results_rows, summarize, write_summary, main. Phase 1 changes:
   write_norms returns the projection ratio (B2); process_item stores the top sets only (N7); full_sets; arm_list
   takes the drop groups; run1c_gap counts nonzero gaps (N4).
4. make_data.py (written, run): data/items_d1.json, ouro_D2.json, ref_values.json (copies from run1c/data),
   data/run1c_ref.json (per-item D and dm of 8 run 1c rows from run1c/out/full/ouro), data/sources.json (sha256).
5. tests/test_ra31.py (31 tests, including N1: top with every unit listed equals dall bit-exact; the size sets; the
   attribution against finite differences; the store round trip and its size at 530 items; the accepted pilot plan),
   tests/prod_fakes.py (copy of ra28/tests/prod_fakes.py).
8. RA addition (after the pilot): ra31.py SIZE_SETS, SIZE_FRAC, rank_order, item_sets (ranked sets and order20),
   attribution, attr_bytes, forwards_equiv, save_attr, pack_attr, Runner.run_grad, process_item (the attribution
   pass), ds_cells (size curve), predictions (signed share), ACCEPTED_PILOT_PLANS; DALL_ALPHAS = (0, 2);
   lib/ra31_hooks.py Spec31.grad_w and the zero leaf on the post-norm MLP write.
6. run.sh (test, hash, pilot, full, summarize; the ra28 layout), RUNSHEET.md, plan_ra31.sha256.
Copies, unchanged (byte-equal to the source): lib/*.py other than ra31_hooks.py from ra28/lib (which equal the run 1c
lib plus ra28_hooks, ra28_score, run1c); tree/prod from ra28/tree/prod (the run 1c prod md5 pins).
7. Phase 3 (Part B, on request): ra31_gen.py with GenWriteHooks(SteerHooks) in this folder (section 10).

## 10. Part B plan (cluster overflow; packaged only on request, from the same edit code)

- Items: the GSM8K own set of Table 2, 391 problems on 1.4B (cluster job a1_gsm8k, prompt A), and the 2.6B pool as
  the second model. Unedited reference: that job's r00 chains (k = 4) and bw1_r00 (batch width 1, 20 items).
- Harness: the cluster-bundle steer_gen copy in tree/prod (md5 3b6e3153; per the review it equals the bundle at
  commit 662fd94, with no change on the 1.4B base path; loopedcot/prod/steer_gen.py is a development copy and is not
  used). Phase 3 confirms the md5 against the file_md5 in the r00 meta before any arm. A subclass
  GenWriteHooks(SteerHooks) in ra31_gen.py adds forward hooks on input_layernorm_2 and post_attention_layernorm_2 of
  every layer (loop from self.f["ut"], layer from self.f["li"]) and down_proj / o_proj pre-hooks for the subset inputs
  (decision 3). Positions: from the last prompt token through every generated token, [plen - 1, None], per-row
  columns from G's plan (left padding handled by G's absolute positions). Loops 2 to 4.
- Ops: whole write y = alpha w (an exact scale op, not a lerp to zero); subsets w + (alpha - 1) c_S with the live
  input and rms(u).
- Sets (review risk c): one fixed set per model from Part A, not a per-problem set. Rank neurons (heads) by how many
  Part A donor items, facts and addition, put them in their top set; break ties by the summed score; keep the top 5
  percent of neurons (2 heads); freeze to data/partb_sets_<model>.json (sha256 in sources.json) before Part B. rand =
  one seeded draw of equal count, also frozen. Printed per alpha: ||c_S|| of top and of rand (mean over problems and
  positions), so a difference in write mass is visible beside the accuracy.
- Arms per problem: 7 alphas (alpha 1 excluded) x 2 branches x 3 sets = 42; one alpha-1 identity arm with every hook
  installed; the loop-count arm: k = 2, MLP whole write, 8 alphas (k = 2 alpha 1 is its own baseline; the k = 4 curve
  is arm "all"); RA addition: an all-position alpha-0 arm per branch (the write removed at every position, loops 2 to
  4), 2 arms, reported beside the SETTLE skip row with the setting difference stated (SETTLE removes the LoRA delta of
  a budget-trained adapter from one module type or layer block, at every position and loop, at a 32-token budget;
  notes/ra_settle_module_removal.md). 53 decoded arms.
- T1 for Part B: the prefill with the hooks installed computes r's edited writes in loops 2 to 4, so r's keys and
  values at those loops (cache index current_ut * L + layer) hold the edited state; each decode step edits its own
  token's writes before its keys and values in later layers are written; the cache is used unchanged otherwise.
  Identity: the alpha-1 arm with every hook installed reproduces the unedited steer_gen chains token by token (r00
  wave layout, batch width 16; and bw1_r00 on its 20 items at batch width 1), and the accuracy equals Table 2's.
  Not compared with run 1c (SDPA path, not eager).
- Readouts: accuracy (the Table 2 parser), share with no parsable answer, mean chain length; 90 percent cluster
  bootstrap. Prediction (brief): accuracy falls on both sides of alpha 1 for the all-neuron MLP arm, and top changes
  accuracy less than rand at the same alpha; no prediction on alpha above 1 at two loops.
- Shards (under 6 h each): by branch and set, 6 shards of 7 alphas x 391 problems, one loop-count shard, one shard for
  the two all-position arms and the identity arm. Cost at the brief's 3 s per decoded chain: 391 x 53 x 3 s = 17.3
  card-hours on 1.4B (2.3 h per branch-set shard, 1.7 h loop count, 0.3 h for the last shard), about double on 2.6B.
  Cut order of the schedule amendment applies (loop-count arm first, then the random sets).

## 11. Pilot, cost and order on the card

- Pilot (done, all gates passed, under the previous plan with 66 arms): the first 25 items of each dataset; report: the alpha tables,
  G1 to G5, dropped counts with reasons, the curves, the T2 print, the reduction gaps, seconds per arm and item, the
  full-run card-hours at that rate (pilot_gate.json field projected_full_hours), and the prune proposal.
- Seconds, measured by the pilot: 7.94 s per item at 76 forwards (0.104 s per forward, 0.105 s per arm and item).
  Full 1.4B under this plan: 85 forward-equivalents per item, about 8.9 s per item; 530 items plus 6 gate items,
  about 1.32 card-hours (G5 budget 1.5; projected by ra31.projected_hours from the pilot gate). [belief: the backward
  costs about 2 forwards; the full run's gate items measure it before the scored rows.]
  Full 2.6B: about 2x per forward on 48 layers, item count from its
  own filter (estimate 400 to 600), 1.8 to 2.7 card-hours plus the clean-filter scan (2 forwards per candidate).
- Order (schedule correction): RA-31 Part A pilot on 1.4B after the RA-29 pilot; full 1.4B when kept (about one
  hour, after RA-30 GSM8K and RA-29 AQuA/CSQA); 2.6B Part A on the card only if the card is free before T0+18h,
  else overflow with Part B; Part B overflow only.

## 12. Open risks (after the review)

1. Resolved by B3: the brief's top-versus-all clause is inverted; section 8 holds the corrected numeric reading.
2. Resolved by B1: G4 separates bias from bf16 noise; the alpha-2 dall - all gap is printed as the noise floor.
3. Resolved (review risk c): Part B uses a fixed set from Part A (section 10).
4. Resolved (review risk d): Part B uses the bundle copy (md5 3b6e3153), confirmed against r00 in Phase 3.
5. Open: Ouro-2.6B base must already be cached on the card at the pinned revision; otherwise 2.6B is not run.
6. Open: the top-5-percent neuron set is global over layers (6,758 of 135,168 on 1.4B), one mask for loops 2 to 4, as
   the brief's "summed over loops 2 to 4" reads; a per-loop or per-layer set is not run.

## 13. Part B engine for the cluster (B-EXT6, 2026-10-12; supersedes section 10 where they differ)

Part A (sections 1 to 9, 11, 12 and the Part A code) is unchanged by this section.

- Code: ra31_gen.py (GenWriteHooks, shards, run_shard, summarize), make_partb.py (data), data_partb/ (its own
  sources_partb.json; data/ and data/sources.json of Part A are untouched), run.sh steps partb_list, partb SHARD,
  partb_summarize, tests/test_partb.py.
- Rows: the a1_gsm8k job of the cluster round, 395 scored problems plus 22 sentinels, decoded as one job per arm in
  the r00 layout (G.main, batch width 16, horizon 512, prompt A). Readouts on the 395 scored rows: accuracy = the
  cells' trace_correct (the Table 2 parser), share with no parsable answer (trace_answer null), mean chain length
  (n_trace); 90 percent cluster bootstrap, cluster = problem. The brief's "391" is a different count; this job's
  r00 accuracy is 0.982 on 395.
- Hooks: GenWriteHooks subclasses the bundle's SteerHooks (tree/prod/steer_gen.py, md5 3b6e315346a8a7cf54048bfe9cf52ad3,
  checked at start). Forward hooks on post_attention_layernorm_2 (MLP write) and input_layernorm_2 (attention write)
  of every layer, and pre-hooks on down_proj and o_proj (the live input of the subset op). The loop is SteerHooks'
  current_ut + 1, the layer its layer index, the rows its batch keys, the positions its absolute positions (left
  padding handled): edited positions are plen - 1 and every later position (every generated and read-out token), or
  every position for the all-position arms. Rows carry no SteerHooks edits, so the parent never acts.
- Ops: whole write y = alpha w (float32, cast once; an exact scale op); subset y = w + (alpha - 1) c_S with the live
  input and full-vector rms (section 2).
- Sets (recorded deviation): frozen from the Part A pilot (out/pilot_1.4b, 50 donor items, facts and addition), not
  the full run, because the Part A full run ends after the push. Rank by the number of donor items whose top set
  holds the unit; ties by the lower flat index (the pilot records store the top sets only, so the summed score is not
  available; 803 neurons tie at the cut count 17). 6,758 neurons, 2 heads (layer 1, heads 10 and 14). rand: one
  seeded draw each ("ra31:partb:neurons", "ra31:partb:heads"). File data_partb/partb_sets_1.4b.json.
- Arms (53 planned, unchanged) plus one reference arm:
  - 42: k = 4, branch MLP or attention, set all / top / rand, alpha in {0, 0.25, 0.5, 0.75, 1.25, 1.5, 2}, loops 2 to 4;
  - 8: k = 2, MLP whole write, alpha in {0, 0.25, ..., 2} including 1 (the k = 2 baseline), loop 2;
  - 1: ident, alpha 1 with every hook installed (both branches, scale and both subsets);
  - 2: all-position alpha 0, MLP and attention, loops 2 to 4 (beside the SETTLE skip row; setting difference in
    section 10);
  - r00_rerun: the unedited run on the cluster card with plain SteerHooks, as r00 had.
- Identity (gate, exit 3): ident equals r00_rerun token by token on every row. Both are also compared with the
  stored r00 chains and the agreement is printed; it is not gated, because bit-exact greedy chains across GPU types
  (r00 ran on a different card and torch build) are not guaranteed. The pin check: the steer_gen md5 of the copy
  equals the bundle pin, and its VERSION equals model_meta.steer_gen of r00's meta ("k17gen-1"); r00's meta holds
  the version string and the modeling md5 (5a0c3755), not the steer_gen file md5. data_partb/r00_meta.json is an
  extract without host paths.
- T1 (cache): proven on the CPU fake under prod's cached, left-padded decode: the cached greedy chain under an edit
  equals the argmax of a no-cache forward of prompt plus chain with the same hooks; the alpha-1 arms of every op
  are bit-exact; from-last leaves earlier positions untouched and all-position does not.
- Shards (15, each one G.main job per arm with a DONE.json; a requeued job resumes and skips done arms): per branch
  and set, alphas {0, 0.25, 0.5, 0.75} ("lo") and {1.25, 1.5, 2} ("hi"); k2_lo, k2_hi; ident_allpos (r00_rerun,
  ident, the two all-position arms, in that order). Card-hours (ra31_gen.py list): the brief's 3 s per decoded
  chain taken as the single-card rate (r00 measured 1.43 s per row at batch width 16 on the faster host), an A100 at 0.6 x
  single card speed (the slow end), subset hooks 1.2 x, k = 2 at 0.55 x, 417 rows per arm: 1.27 to 2.78 h per shard,
  32.5 card-hours in total. Longer chains at alpha 0 or 2 lengthen a shard; the largest stays under 6 h up to 2.1 x
  the r00 length.
- Summarize: ra31_gen.py summarize merges every arm folder under the output root: results_partb.json,
  results_partb.csv (results_common columns), checks_partb.json, summary_partb.md (accuracy, no-answer share, mean
  chain length, mean ||c_S|| per subset arm, the identity results). Missing arms are reported as not run.

## 14. Model axis: Part A on Ouro-1.4B-Thinking and Ouro-2.6B-Thinking (B-EXT6b, 2026-10-12)

Model axis only: no new arms, no new predictions, no change to Part A on 1.4B base or 2.6B base, or to Part B.
- Models: ouro_1_4b_think (revision 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d, 24 layers) and ouro_2_6b_think
  (f1edd81e7ac41355db670500ceaf204e0f73af68, 48 layers), CLI keys 1.4b-think and 2.6b-think. Same pins as Part A
  otherwise (transformers 4.56.2, eager, bf16, prod md5); the revision is checked against ra31.REVISION_PINS (equal
  to tree/prod/tasks/data/model_revisions.json); exit 2 on a difference. Raw completion prompts, no chat template,
  as in the RA-23 branch-freeze runs on these models.
- Items: the existing selection code at run time on the run host (section 4, the 2.6B base path): facts = every
  k15_data.d1_candidates pair then the model's own clean filter; addition = d2_candidates, cap 4 per A-side
  operand value, same filter. Token ids are built inside the job with the model's own tokenizer; no token-id file is
  made in advance. Counts expected from the RA-23 branch-freeze cluster runs (same code, same revisions):
  1.4B-Thinking facts 267 of 295 (10 A-side, 18 B-side clean fails), addition 263; 2.6B-Thinking facts 266 of 295
  (16 and 13 fails), addition 263. The run reports its own counts in selection_<ds>.json.
- Jobs per model: a pilot (25 items per dataset, every gate) first, then the full run, which needs that model's
  passing pilot gate. G3 (the run 1c reproduction) is not applied (no run 1c reference on these models); G1, G2,
  G4 and G5 apply. G5 budgets: 3.0 card-hours (1.4B-Thinking), 5.5 (2.6B-Thinking), sized for an A100 at 0.6 x the
  single card's speed and under the 6 h wall.
- Card-hours [belief: the pilot rate of 1.4B base, 8.88 s per item at 85 forward-equivalents on the single card;
  2.6B at 2 x for 48 layers; 56 item runs per pilot including the repeated gate items; the clean-filter scan about 2
  forwards per candidate]:

| cell | items | pilot, single card | full, single card | pilot, A100 at 0.6 x | full, A100 at 0.6 x |
|---|---|---|---|---|---|
| 1.4B-Thinking | 530 | 0.16 h | 1.36 h | 0.27 h | 2.27 h |
| 2.6B-Thinking | 529 | 0.30 h | 2.71 h | 0.50 h | 4.52 h |

  At 1.0 x the A100 columns equal the single-card columns.

## 15. Part B breadth grid (EXT6c, 2026-10-12; PI decisions of the same day)

No new arms and no new predictions: the section 13 arm types on the Table 2 grid of two base models by four
benchmarks. Part A and the GSM8K 1.4B Part B (section 13, run in EXT6) are unchanged; the cells use new functions,
commands and data only (ra31_gen.py cell, cell_list, cell_summarize, freeze26, dep26; make_partb_cells.py;
data_partb/<cell>/; steer_gen_math.py and tree/data_steer/asdiv, tree/data_steer/mawps copied unchanged from the bundle).
- Cells: 1.4B SVAMP, ASDiv, MAWPS; 2.6B GSM8K, SVAMP, ASDiv, MAWPS (1.4B GSM8K is section 13). No Part B on the
  Thinking models. Pools: the cluster round's a1 (1.4B) and a2 (2.6B) generation jobs, prompt A, the r00 layout, at
  most 395 scored problems per cell (the 1.4B GSM8K cell's count; the first in items.json order) plus every
  sentinel: svamp_1.4b 244 + 8, asdiv_1.4b 395 of 1,036 + 22, gsm8k_2.6b 395 of 425 + 22, svamp_2.6b 248 + 4,
  asdiv_2.6b 395 of 1,059 + 22 (equal, idx and order, to the RA-30 builder's Table 2 list
  ra30/data/pool_asdiv_ouro_2_6b_base.json, checked and hashed by make_partb_cells.py; ra30 is read only).
  MAWPS (1.4B a1_mawps, 2.6B a2_mawps): items.json is read at run time from <OUT_ROOT>/A/steer_gen_math/a1_mawps
  (a2_mawps), the path pattern of gen_jobs.py, with --pool_root; its sha256 goes into the shard record; exit 2 if
  missing. ASDiv and MAWPS run through the bundle's steer_gen_math.install (md5 f76f3e722e364d2471148602debd1c1c).
- Arms per cell (23), MLP lines: the whole write (scale op), the top set and the random set (subset op) at alpha
  0.5, 0.75, 1.25, 1.5 (four loops, loops 2 to 4); the loop-count arm (two loops, edit at loop 2: the whole write at
  alpha 1, 1.25, 1.5, the top and random sets at 1.25 and 1.5; compared with four loops at alpha 1, the r00_rerun
  arm); the two all-position alpha-0 arms (MLP, attention; loops 2 to 4); r00_rerun (unedited, plain SteerHooks);
  ident (alpha 1 with every hook installed: MLP scale, top and random subsets, attention scale).
- Sets: 1.4B cells the frozen 1.4B set (section 13). 2.6B cells (all four) chain behind the 2.6B set: freeze26, a CPU
  job, applies the section 13 rule (count over donor items, ties by the lower index, 13,517 neurons of 48 x 5,632;
  random draw seeded "ra31:partb:neurons") to the EXT6 2.6B Part A pilot (<OUT_ROOT>/E6/ra31/a_2.6b/pilot_2.6b, group
  g21_e6ra31a, stage s1_pilot; exit 2 unless its pilot_gate.json passed) and writes
  <OUT_ROOT>/E6/ra31/partb_sets_2.6b.json; every 2.6B shard needs that file (--top_sets; exit 2 without; a 1.4B
  file is refused). The submit side: ra31_gen.py dep26 --out_root <OUT_ROOT> prints afterok:<id> of that pilot from
  <OUT_ROOT>/_chain/submitted.tsv, or done (pilot gate passed; "done set" when the set file exists too); exit 2 if
  neither. Recorded deviation (printed in each 2.6B summary): the 2.6B set comes from the 50-item 2.6B pilot, across
  units, not from a full Part A run.
- Gates as section 13: pins (steer_gen md5; data cells also the data hashes and the r00 version string; math tasks the
  steer_gen_math md5), the prompt length of every scored row equal to the pool's plen (exit 2: a tokenizer
  difference), ident against r00_rerun token by token (exit 3); agreement with the stored r00 printed where the pool
  has r00 chains.
- Shards (one cluster job each; amended the same day so that the 2.6B chain fits: no ordering, no "_all first"):
  the 22 measurement arms of a cell fall into four independent groups, all (the whole-write line), lc (r00_rerun, the
  four-loop alpha-1 baseline on every item; the whole-write loop-count arm; the two all-position alpha-0 arms), top
  and rand (each line with its loop-count arm). Every shard runs its own identity gate on its own items: gate_rerun
  (unedited, plain SteerHooks) and gate_ident (alpha 1 with every hook installed: MLP scale, top and random subsets,
  attention scale), token by token equal or exit 3; the lc shard compares gate_ident with its own r00_rerun instead
  of a second rerun. The rule was a full rerun unless that doubled the cost; it adds about 40 percent (37.9 to 51.2
  single-card hours over the grid), so the full own-items gate is used (GATE_ITEMS = None; an int switches to a fixed
  gate on the cell's first problems). A group whose shard exceeds 4.5 h at 0.6 x is split by items into n parts, part
  s = the decoded ids [s::n] (2.6B GSM8K: all and lc in 2, top and rand in 3). 1.4B shards start at once; 2.6B shards
  start after freeze26 (seconds, after the EXT6 2.6B pilot), all in parallel. Card-hours (ra31_gen.py cell_list; each
  cell's r00 rate on the faster host scaled to the single card as for GSM8K 1.4B; MAWPS at the same model's ASDiv rate
  and 395 + 22 rows [belief]; arm factors of section 13):

| shard | arms | single card | A100 at 0.6 x |
|---|---|---|---|
| svamp_1.4b_all | 4 | 1.12 h | 1.86 h |
| svamp_1.4b_lc | 6 | 1.05 h | 1.75 h |
| svamp_1.4b_top | 6 | 1.50 h | 2.50 h |
| svamp_1.4b_rand | 6 | 1.50 h | 2.50 h |
| asdiv_1.4b_all | 4 | 0.95 h | 1.58 h |
| asdiv_1.4b_lc | 6 | 0.90 h | 1.50 h |
| asdiv_1.4b_top | 6 | 1.28 h | 2.13 h |
| asdiv_1.4b_rand | 6 | 1.28 h | 2.13 h |
| mawps_1.4b_all | 4 | 0.95 h | 1.58 h |
| mawps_1.4b_lc | 6 | 0.90 h | 1.50 h |
| mawps_1.4b_top | 6 | 1.28 h | 2.13 h |
| mawps_1.4b_rand | 6 | 1.28 h | 2.13 h |
| gsm8k_2.6b_all_p1of2 | 4 | 2.20 h | 3.66 h |
| gsm8k_2.6b_all_p2of2 | 4 | 2.20 h | 3.66 h |
| gsm8k_2.6b_lc_p1of2 | 6 | 2.07 h | 3.46 h |
| gsm8k_2.6b_lc_p2of2 | 6 | 2.07 h | 3.46 h |
| gsm8k_2.6b_top_p1of3 | 6 | 1.96 h | 3.27 h |
| gsm8k_2.6b_top_p2of3 | 6 | 1.96 h | 3.27 h |
| gsm8k_2.6b_top_p3of3 | 6 | 1.96 h | 3.27 h |
| gsm8k_2.6b_rand_p1of3 | 6 | 1.96 h | 3.27 h |
| gsm8k_2.6b_rand_p2of3 | 6 | 1.96 h | 3.27 h |
| gsm8k_2.6b_rand_p3of3 | 6 | 1.96 h | 3.27 h |
| svamp_2.6b_all | 4 | 1.34 h | 2.23 h |
| svamp_2.6b_lc | 6 | 1.26 h | 2.11 h |
| svamp_2.6b_top | 6 | 1.80 h | 3.00 h |
| svamp_2.6b_rand | 6 | 1.80 h | 3.00 h |
| asdiv_2.6b_all | 4 | 1.16 h | 1.93 h |
| asdiv_2.6b_lc | 6 | 1.09 h | 1.82 h |
| asdiv_2.6b_top | 6 | 1.55 h | 2.59 h |
| asdiv_2.6b_rand | 6 | 1.55 h | 2.59 h |
| mawps_2.6b_all | 4 | 1.16 h | 1.93 h |
| mawps_2.6b_lc | 6 | 1.09 h | 1.82 h |
| mawps_2.6b_top | 6 | 1.55 h | 2.59 h |
| mawps_2.6b_rand | 6 | 1.55 h | 2.59 h |

  Per cell (single card / 0.6 x, hours): svamp_1.4b 5.17 / 8.61; asdiv_1.4b 4.41 / 7.34; mawps_1.4b 4.41 / 7.34;
  gsm8k_2.6b 20.30 / 33.86; svamp_2.6b 6.20 / 10.34; asdiv_2.6b 5.35 / 8.93; mawps_2.6b 5.35 / 8.93. Total 51.19 h
  on the single card, 85.35 h at 0.6 x. Longest chain: 2.6B, the EXT6 pilot, then freeze26 (seconds), then the
  longest shard (gsm8k_2.6b_all_p2of2), 3.66 h at 0.6 x.
- Merge: ra31_gen.py cell_summarize reads every shard folder of the cell (out/<shard>/<arm>), merges each arm's item
  parts and stops (exit 2) when one arm holds a problem twice; it reports the planned and found shards, every gate,
  and the merged r00_rerun's agreement with the stored r00 where the pool has r00 chains.
- Summary per cell: ra31_gen.py cell_summarize writes results_partb_<cell>.json and .csv, checks_partb_<cell>.json
  and summary_partb_<cell>.md (accuracy, no-answer share, mean chain length, mean ||c_S||, identity results, the set
  rule, deviation and tie count, the pool sha256 and, for 2.6B and MAWPS, the deviations).
