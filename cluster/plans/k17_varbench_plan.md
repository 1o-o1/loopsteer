# B-K17-VARBENCH plan: real-math placement with the copy-read target (v5, 2026-10-07 night)

v1 was written before any code or run. v2 applies PLAN.md amendments A1-A6 and the plan critic F1-F12 (sections
"Amendments applied" and "Critic log"). v3 applies amendment A7 and the code-review findings (section 14 "Review
log"); it changes endpoints, site sets, the item build and the V6 rules, so the debug rows of section 13 are
harness evidence only. v4 applies the second plan critic (F1-F8, "Critic log", "Plan critic 2"): A2-1 is scored
against the readout-proximity site, A2-3 leaves the tally, undo is cited only at e_carried-selected slots, a failed
answer gate is no longer called "no answer state", a parameter-free digit probe with its own positive control, V6
strata and PC values, a plan hash at launch, and D4 clusters by connected component. v5 applies plan critic 3
(F1-F8, "Critic log", "Plan critic 3") and re-freezes sections 4-10 (DV13): undo and window by influence in
first-token logits with CIs (V5, T6), a planted-digit power control at every probed site, the RND citation rule
(T3b), a host-migration rule, a per-family finite-difference gate on V6, a derived-use VarBench target (VBD, V8)
as the real-math computed dataset of A2-2, T6 scored outside the W subset, an ST1e stratum, and S1 stated as 4
per-model jobs. v5.1 and v5.2 (2026-10-08, DV17) fix the FD check's fp32 pass (TF32 off) and write the code's bf16
selector condition into the V6 gate in the T7 band form. v6 (2026-10-08, DV18) applies code review 3 (17 findings,
section 14 "Review log v6"): T3b cites O1 and uses RND only as a gate on lower bounds, b2 per D_KO, the dose clause
with a scaled band and paired items, the bf16 gate on 30 fresh items, number words in the prefix filter (items
rebuilt in `data_v6`), the PLAN G4 path-integrated gain row and G8 hashes, host roles keyed on the GPU name, the
random relay control drawn before the span, and a recomputed S1 estimate. Still before any scored full-run row.
Tags: [read] in code or files, [ran] computed on CPU or a GPU, [belief] untested.

Builder: Opus. Owned files: `loop-flow/src/k17_varbench.py` (new), this plan, remote
`out/k17_varbench/` (debug runs in `out/k17_varbench/debug/`) on fast-host and shared-host, pulled to
`loop-flow/artifacts_spark/k17_varbench/`. Imports, never modified: k15_smoke, k15_smoke2, k15_data,
k13_edit, k10_ifm, k16_path_ouro (helpers only).

## Amendments applied (PLAN.md A1-A7)

- A1: every cell in checks.json carries n_items, n_clusters, strata ST1 (source tokens 1 vs >= 2) and ST6
  (paraphrase mode), and the PC-final and PC-site values for its model. Cell keys are listed in section 9.
- A2: carry-gain predict-then-verify is arm V6, on VarBench, D1, D2, D4, on Ouro x4 and the three IFM models.
- A3: the brief's "first answer token after the prompt" read-out is replaced by the copy-read target (I2). The
  brief's stratum "answer first digit differs" maps to the copied number (ST2).
- A4: not this track (GEN).
- A5: no undo on the injection twin (re-injection makes it uninformative), same logic as A5 for raven models.
- A6: V6 primary endpoint = top-1 hit rate (with F3's eligibility and tie rules), answer sites only, verify at
  a = 0.25 and a = 1.
- PLAN 3b2 (paper chat decision, via critic F1): the "last loop" answer site is Lmid@K. Llast@K at the answer
  position is the readout input and is used only as the PC-final harness row.
- A7 (v3, review F2/MS5): the V6 site sets keep loops 1..K-1 and only L0@K and Lmid@K in the last loop, for the
  selector, the verify rows and the dev search (section 2). The selector claim is limited to separating carried
  from not-carried sites and ranking candidates for verification. Per dataset the false-negative rate is reported
  (T7). checks.json, results.json and summary.md carry host roles, not host names, user names or home paths.
- v4 (re-read A1-A8 on 2026-10-07 21:20): no amendment changed since v3 (A8 is PUBLIC only). A1 was not met for
  the V6 cells; plan critic 2 F6 adds ST1 (and ST2 on VarBench) strata and the PC-final value to every carry_gain
  record (section 9).
- v5 (re-read A1-A8 and the Hard rules on 2026-10-07 night): A1-A8 are unchanged since v4. They still supersede the
  brief: A3 replaces the read-out (DV1), A4 moves generation to GEN, A5 is applied to the injection twin (no undo).
  New since v4: the HOST RULE (Hard rules, the PI 22:35): fast-host may be used, every fast-host job must be resumable
  and killable, and fast-host is freed on the PI's word. Applied in section 7 (migration rule, critic 3 F3). A1 for
  the new VBD cells (n_items, n_clusters, ST1/ST1e strata, PC-final) is in section 9.
- v6 (PLAN.md "Amendments after GPT critic 3", G1-G8, binding since 2026-10-08; review C-F4). G1: this track does
  not define the paper's placement rule; the undo slots are chosen on the W subset and scored on M-only items, the
  dev-tuned site is fixed on dev items, and the per-item best verified site is the oracle used for regret, with its
  forwards counted (`cost_fwd_equiv.dev_search_forwards`). G2, G3, G5: X10 and X4, not this track. G4: new V6 row,
  the path-integrated gain G_PI (section 4 V6 v6). G6: not this track (no public models here). G7: the VarBench
  pool is the GSM8K test core pool of the pairs file (split "eval", 1198 rows, 522 problems) [read]; whether it
  equals SETTLE's 1,219-item eval subset is not verified by this track. G8: the launch record holds the sha256 of
  every data file and of the run config before any row; phase M refuses changed data files; summarize hashes every
  output file before it reads them (`output_sha256`). The plan hash (written hypotheses) is unchanged (section 9).

## 0. Recon facts

- RF1 [read] A3 read-out: teacher-force A's gold GSM8K solution up to the token before the first restatement of
  the edited number x, read the x' vs x preference there. Source site = number span in the prompt. Answer site
  = last teacher-forced position r.
- RF2 [ran] Core pool (text_exact, eval, outside_equal on four tokenizers): 1198 rows / 522 problems. The gold
  solution restates value_A in 1179 rows / 516 problems; 6 rows restate it first inside `<<...>>` (excluded).
  Joint encoding of prompt + " " + chain prefix + number keeps the prompt ids as a prefix and gives equal-length
  A/B sequences on Ouro and IFM for all 1179 rows.
- RF3 [ran] Ouro splits digits, so the first differing token can be the second digit (16 vs 12 reads after "1").
  IFM uses 3-digit chunks. Input length to the read: Ouro median 665 (max 938), IFM median 553.
- RF4 [ran] S200 (v1 rule): Ouro span 1 token 87 vs >= 2 tokens 113; IFM 196 vs 4. Digits 1/2/>=3: 87/85/28.
  Dev/test 105/95.
- RF5 [ran] A resampled VarBench render is degenerate for O2 on 350/1198 rows and stale on 878/1179.
- RF6 [ran] SETTLE GSM8K block = openai/gsm8k train rows 0-3; rows 4-7 give the alternative block R2.
- RF7 [read] Ouro x4 share tokenizer.json (md5 5473993d). 1.4B: 24 layers, 2.6B: 48, hidden 2048, 4 loops, no BOS.
  data_steer README: transformers 5.x changes Ouro prompt lengths on some rows, so Ouro ids come from
  tokenizer.json via the `tokenizers` library (equal to transformers 4.56.2).
- RF8 [read] Ouro: `OuroDecoderLayer` returns a tensor; `model.norm` runs once per loop and its output is the next
  loop's input and the readout input; eager attention returns post-softmax probabilities as `out[1]` (P8 holds).
  IFM: `IfmRunner.fwd` is `@torch.no_grad()`; gradients use `IfmRunner.fwd.__wrapped__`. IFM attention probs are
  not returned by the module (backend None), see F11.
- RF9 [ran] Laptop (RTX 4070 Laptop, 8 GB) with `.venv-ouro` (torch 2.11, transformers 4.56.2) runs Ouro-1.4B;
  anaconda's transformers 5.7 cannot load Ouro's remote code. fast-host and shared-host were unreachable (ssh timeout) at
  v2 time; their GPU state, the 2.6B-Thinking cache and the IFM tokenizer on shared-host are unverified.
- RF10 [ran] Prefix validity on S200 (critic F4 reproduction): no prefix calculator step takes an input derived
  from x. Two rows have x also elsewhere in the question (ambiguous copy source): gsm8k-616-initial_height,
  gsm8k-809-initial_candies.
- RF11 notes/audit_path_ouro.md now exists (re-read 2026-10-07 night, after the API outage). How it applies: F8
  dyadic D1 CIs are already in v3 (section 5). F1 and F7 (Y4 per-item R^2 -1.09 on D1; the answer position's own
  MLP and attention cancel about 95% of any carried change) mean G is a first-order ranking with small magnitudes,
  which is the A7 scope; P-A2 stays as frozen and is not reworded on this basis. F6 (only D2 carries an answer
  edit; D4/T2 is lost) is consistent with the P-A2 retrieval/computed split only for D2, so A2-2 and A2-3 on
  D4-T2 are expected to miss [belief]; the frozen rows are kept and scored as written.
- RF12 [ran, 2026-10-07 night, read-only ssh] Both hosts reachable again. No k17_varbench process on fast-host or
  shared-host. fast-host GPU0 155 MiB and GPU1 172 MiB used of 97.9 GB. Remote `out/k17_varbench/` holds `data`, `debug`,
  `logs` (plus `models` on fast-host); `data_v3` exists only on the laptop and must be pushed before the v3 host
  debug run. No full-run row exists.
- RF13 [ran, 2026-10-07 21:20, after the second API outage] No k17_varbench process on either host. fast-host GPU0
  155 MiB, GPU1 172 MiB used; shared-host ran a k17_parcae debug (6.6 GB) that ended before the v4 IFM debug launched.
  Remote code was v2.1 (md5 59b2c6f5); v4 code (md5 ca605783) and data_v3 pushed to both hosts, md5-checked.
- RF14 [ran, laptop CPU, v5] Derived-use yield on the 1198 core rows. The literal critic 3 F5 rule (a chain prefix
  that never restates x) keeps 0 rows with the calculator annotations and 5 without them, and those 5 are copies
  ("Chair = 3"). GSM8K gold steps restate their operands ("48/2 = <<48/2=24>>"). The own-value rule of I8 keeps 848
  rows / 341 problems (drops: unverified prose number 321, no derived step 20, ambiguous source 5, other 3).
- RF15 [ran, laptop CPU, v5] The v4 debug rows rescored with the v5 summarize (n = 3 per model, harness only): RND
  fails T3 at 2 Ouro-1.4B cells (answer Llast@1, Llast@3), 7 Ouro-2.6B cells (source Llast@1, Llast@3; answer
  Llast@1, L0@2, Llast@2, Llast@3, L0@4) and 6 Ouro-2.6B-Thinking cells (source and answer Llast@1-3). There O1 -
  RND is 0.68-0.81 on source cells and -0.19 to +0.33 on answer cells. On Ouro-1.4B Llast@3, influence removal is
  0.25 [0.07, 0.50] while necessity dep is 0.03. Both agree with critic 3 F1 and F2 on these 3 items.

## 1. Items

- I1 VarBench pool = RF2's 1179 rows minus annotation rows, minus rows flagged by the prefix-validity filter (I3:
  contradictory prefix, ambiguous source), each count reported. One pair per problem (v3, review F8): the lowest
  var_name among the valid rows (prod DESIGN 12.3, the rule B-K17-GEN arm (a) uses on the core pool). Each item
  stores GEN's choice (`gen_pair_id`, lowest var_name over the core pool); they agree on 500 of 513 problems and
  198 of S200 [ran], so E1 and E9a can be paired per item on that intersection. Problem order:
  sha256("k17vb-order:" + idx). S200 = first 200 problems. Items are rebuilt in `out/k17_varbench/data_v3/`; the v2
  build in `data/` stays for the debug provenance. Ouro models run every S200 item that passes the clean filter. IFM models run the first 60 that
  pass (cap), raised to 120 on a model only if the debug variance shows 60 cannot reach TOST (F6). Cross-model
  contrasts use the intersection of kept items.
- I2 Sequences, per tokenizer. P_A = SETTLE block + "Question: " + q_A + "\nAnswer:". Chain prefix C = gold[:c], c
  = first isolated match of str(value_A). full_A = enc(P_A + " " + C + str(x)), full_B likewise with B's
  prompt and x' (the chain text C is A's). t* = first index after the prompt where full_A and full_B differ, r =
  t* - 1. Continuations: restA = full_A[t*:], restB = full_B[t*:] (equal length, asserted). Row sequences
  (all of length T' = t* + len(rest) - 1):
  SA = full_A[:t*] + restA[:-1] (A prompt, x continued), SA' = full_A[:t*] + restB[:-1] (A prompt, x' continued),
  SB = full_B[:t*] + restB[:-1], SB' = full_B[:t*] + restA[:-1].
  Every edited row of an item runs on SA (first-token margin and the x log-probabilities). A row that needs the
  full-span endpoint runs SA' with the same edit (one extra forward, only when len(rest) >= 2: "cont" items).
  Causality leaves positions <= r unchanged, checked bit for bit in debug (`causality_ok`).
- I3 Build-time asserts (`--build_items`, laptop CPU): prompt ids equal the pair file's len_A and span; the joint
  encoding keeps the prompt prefix; full_A and full_B differ only inside the span before t*; restA[0] != restB[0];
  R2 reproduces the question + chain tokens after its block. Prefix validity (GPT critic, EXPERIMENTS 3c):
  "contradictory prefix" = a `<<expr=res>>` step in C whose inputs include x or a result derived from x (taint
  tracking); "ambiguous source" = x equals another number in the question outside the span. Both are dropped
  and counted. Load-time assert on each host: the model's tokenizer reproduces every kept item's ids.
  v6 (review C-F3): the I8 prose rule now also applies to every copy-read prefix C, with number words. After the
  contradictory-prefix check, C is dropped if it states x as a number word ("x_word_form") or if a number outside
  the `<<...>>` annotations, in digits or words, is neither a question number (digits or words) nor the result of an
  earlier annotation in C ("prose_unverified_number": it can be an unannotated value derived from x, e.g.
  gsm8k-357 "Three total people" = 90 / 30). Each row is counted under its first reason in `prefix_filter`.
  Rebuild [ran, laptop CPU] into `out/k17_varbench/data_v6/` (data_v3 kept for the debug provenance): 1164 -> 913
  valid rows, 513 -> 409 problems (x_word_form 15, prose_unverified_number 237, contradictory 1, ambiguous 9). S200
  shares 140 of 200 pairs with the v3 S200 (first 60: 39). S200 strata (Ouro): span 1 / >= 2 tokens 86 / 114,
  digits 1/2/>=3 86/90/24, cont 97, dev 106; IFM span 193 / 7.
- I4 Clean filter (batch 1, final readout, on SA and SB): argmax z_A[r] = tA and argmax z_B[r] = tB. Counts of
  scanned, A_fail, B_fail per model. No filter on the continuation (teacher-forced).
- I5 R2 render: train rows 4-7 as the 4-shot block, same question, same chain prefix, positions shifted by the
  constant offset. R2 clean pass (P4): argmax at r' is tA on R2-A and tB on R2-B; items failing it get no O2 rows
  and no presence validation.
- I6 Split and clusters. Split = sha256("k17vb-split:" + idx) first hex digit even -> dev. Cluster = GSM8K idx
  (one pair per problem, n_clusters = n_items).
- I7 A2 datasets (carry gain only; read = last prompt position).
  - D1: k15_data.d1_candidates + select (cap 2 per subject), quota n/4 per (split, rel). The pool has 49 countries
    and 33 cities, so fresh subjects outside the K16 items are not available: D1 is labelled "same pool as K16 D1"
    (F9b). Strata pos_len (subject tokens 1 vs >= 2) and subject_pos0 (Ouro city template starts with the
    subject). v3 split (review MS3/F6): split = sha256("k17vb-d1:" + rel + ":" + subject), even first hex digit ->
    dev, for the A subject; a pair is kept only if its B subject hashes to the same split, so dev and test share no
    subject on either side (the select() quota key "half" follows this split). Ouro tokenizer [ran]: 295 candidates,
    161 cross-split pairs dropped, 134 kept, 38 dev and 39 test subjects, 0 shared (the K15 parity split shared 70
    A-side clusters). With the cap, at most 69 pairs before the clean filter (dev capital 21, dev city 14, test
    capital 27, test city 7), so D1 is below the 80 target on every model and the count is reported. CIs: dyadic
    bootstrap over subjects (A and B side; audit_path_ouro F8).
  - D2: d2_candidates, fresh items: every problem used by any K16 run (Ouro or IFM, from the K16 artifacts) is
    excluded. Quota 40 per op. Cluster = A-side edited operand value (audit T8). Split = sha256 of the cluster key.
  - D4: data/d4/d4_<tok>.jsonl, rows with len(ids_A) = len(ids_B) and ids differing exactly at source_span
    (P10(2)). Content key (P10(5)) = sha256(kind | vals | index of src_var | indices of query_vars), for the A side
    and the B side of a row. v4 (plan critic 2 F8): cluster = connected component of the graph with content keys
    as nodes and rows as A-key - B-key edges, over every row of the file; split = sha256("k17vb-d4c:" +
    component). Fresh items: a component that holds any key of a K16 row (A or B side, or the v3 K16 content set)
    is excluded. Ouro tokenizer [ran]: 814 components (largest 7 keys, 15 rows), 98 K16 components and 290 rows
    excluded (v3: 220), 1411 candidates, dev/test shared content keys 0 (v3: 144 of 877 test items shared a
    B-side key with a dev A-side key). Quota 20 per (split, kind), one item per component.
  - Order inside every dataset: round-robin over (split, kind or rel) so any cut keeps splits and kinds balanced
    (F9d).
- I8 VBD, the derived-use VarBench target (v5, critic 3 F5, LEDGER FINAL FRAME D2). Same core rows and pairs as
  I1. In A's gold chain, take the first calculator step `<<expr=res>>` whose inputs include x. Shared prefix =
  the gold chain up to that step's `<<expr=`, with every isolated x rendered with each side's own value (A: x, B:
  x'). Each side's prefix is therefore self-consistent and holds no x-derived result ("no conflicting
  calculation"). Read r = the position before the first token where res_A and res_B differ. res_B = expr with x'
  substituted, evaluated by a whitelist arithmetic evaluator. Source = the prompt span plus every chain position
  of x. Drops (counted): no derived step, an earlier step result equal to x, x in another format, a prose number
  that is neither x nor a question number nor an earlier step result (possible unannotated derived value),
  ambiguous source, an annotation whose expression does not evaluate to its stated result, result unchanged,
  tokenizer asserts (joint encoding keeps the prompt and the prefix; differences before the result lie inside the
  source). One pair per problem (pick_pair), the I1 order and split, cluster = problem. Clean filter: top-1 at r is
  tA on A and tB on B. Endpoint: first-token e only (no continuation). Build: `--build_vbd` writes
  `items_vbd_<tok>.json` into data_v3 [ran: 341 problems on both tokenizers, 184 dev; chain source positions
  median 4 on Ouro, 2 on IFM]. Literal rule rejected: RF14.
  v6 (review C-F3): the prose and "x in another format" checks also read number words (shared `prose_flags`), and
  number words in the question count as question numbers. Rebuild [ran]: 848 rows, 351 problems on both tokenizers
  (v3: 848 rows, 341 problems; 27 rows leave, mostly x or a derived value written as a word, and 27 enter, whose
  prose numbers are question numbers written as words in the question).

## 2. Models, sites, hooks

| model | host | V1 sites (loop j = 1..4) | tensor | readout |
|---|---|---|---|---|
| ouro, ouro_think | fast-host GPU1 | L0@j = layers[0] out, Lmid@j = layers[12] out, Llast@j = model.norm out | raw residual; norm out = carried state | lm_head(norm out of loop j) at r..T'-1, float32 |
| ouro26, ouro26_think | fast-host GPU1 | layers[0], layers[24], model.norm | same | same |
| ifm_p1_ouro, ifm_p1_ouro_inj | shared-host xllm_venv | slot 28(j-1) + {0, 14, 27} | block output via `r._layer` | m.output(last slot output) at r..T'-1 |
| ifm_d112 | shared-host | layer 28(j-1) + {0, 14, 27} | block output | m.output |

- Readout (v3, review MS2/F2): logits are computed in float32 from the bf16 state, Ouro `F.linear(h.float(),
  W.float())`, IFM a float32 copy of `m.output` applied to the float32 state, both with TF32 off inside the call
  (load_ifm turns TF32 on; the previous value is restored). The v2 code cast bf16 logits to float, which rounded
  them to 1/16-1/8 logit and tied distinct site edits. Hook check (h) and PC-final / SZ exactness still hold
  because both sides use the same path; checks.json `readout` logs the dtype and the fp32-vs-bf16 head difference.
- Answer-site sets for V6 (F1, F2, A7). Ouro site (j, l): layer l output for l < L-1, the norm output for l = L-1.
  v3 (A7): loops 1..K-1 keep their layers; the last loop keeps only L0@K and Lmid@K. The readout-feeding tensor
  (Ouro norm out of loop K, IFM slot 111, d112 layer 111) and every other last-loop layer are excluded.
  - G_full (every layer of loops 1..K-1, plus L0@K, Lmid@K): 74 sites on Ouro-1.4B, 146 on Ouro-2.6B. Used on the
    two 1.4B models in the pilot.
  - G_red (5 depths x loops 1..K-1, plus L0@K, Lmid@K = 17): Ouro-1.4B layers {0, 6, 12, 18, norm}, Ouro-2.6B
    {0, 12, 24, 36, norm}, IFM and d112 slots 28(j-1) + {0, 7, 14, 21, 27} for j <= 3 and slots 84, 98. Used on
    ouro26 x2 and the IFM models for the selector and the dev search; also reported on the 1.4B models as a subset
    analysis, so every model has a G_red result. `--gset` overrides the set per model (section 10).
  - Uniform rule = Lmid@K (Ouro layers[L/2] in loop 4, IFM/d112 slot 98). Under A7 it is also the latest site in
    the set, so the readout-proximity baseline asked by review F2 equals the uniform rule (reported as
    `latest_site`, with `latest_equals_uniform`); within loops 1..K-1 the readout-proximity baseline is Llast@K-1.
- H1 Ouro layer hooks (every layer, per-layer call counter = loop j) and H2 norm hook: edits
  (k15_smoke2.apply_entries), records, zero grad leaves. H3 attention wrapper keyed on (current_ut + 1,
  layer_idx): additive finfo.min on mask[:, :, Q, keys] (Q = {r} direct, or every position after the span end
  relay); probabilities read from the module's `out[1]` (P8).
- H4 IFM: own `r._layer` wrapper keyed on the executed slot (edits, records, leaves, final hidden capture) and
  a `segments` knockout wrapper on CausalSoftmaxAttention (backend None asserted). Gradients through
  `IfmRunner.fwd.__wrapped__` under enable_grad. `loop_end_layers == num_layers` asserted (audit M2).
- Ouro loads with attn_implementation="eager". Thinking checkpoints get the base prompt (DV2). Model copies as
  v1 (1.4B-Thinking from fast-host `out/k16_path_ouro/models/`, 2.6B-Thinking from the HF cache or copied from the
  shared-host cache into `out/k17_varbench/models/` with md5 checks). No hub download.
- Hook checks per model before any scored row (critic B4), stored in checks.json: (a) call counts; (b) zero add at
  every site and zero grad leaves give bit-identical logits; (c) repeat run bit-identical; (d) an edit at (j, l)
  position p leaves every earlier call bit-identical and later calls bit-identical at positions < p; (e) knockout
  placement: Ouro, module probabilities on knocked keys exactly 0 for direct and relay, positions outside Q
  identical; IFM (F11), positions outside Q identical at every slot, the module output equals our recompute
  under the knockout mask, and functionally the full relay span knockout gives dep >= 0.5 while an equal-size
  random non-span knockout gives |dep| < 0.5 x dep_span (on the window subset); (f) the grad run's logits equal
  the no-grad logits; (g) causality: z at r from SA and SA' runs bit-identical; (h) PC-final, SZ, PC-site exact.
  v5 (critic 3 F1, F3): the checks are stored per host role with the code sha256 (`checks_by_role`) and rerun
  whenever the host role or the code hash differs from the last passing check. v6 (review C-F5): the key is (host
  role, GPU name), because the role label is set by hand.

## 3. Operators and controls

- O1 donor: H[s, P] <- lerp(H, H^B[s, P], a); a = 1 unless stated; donor = clean B run on SB at the same site.
- O2 context-robustness check (F8, relabelled): H[s, P] += v, v = H^{B,R2}[s, P'] - H^{A,R2}[s, P'], at Llast@1 and
  Llast@3 for both positions, only on items passing the R2 clean filter. Never cited as vector transfer. Diagnostic
  per cell: cos(v, own delta) and |v| / |own delta|.
- RND random same-norm: per position, unit Gaussian (seed = first 8 hex of sha256(pair_id|site|pos|p)) scaled to
  |H^B - H^A| at (s, p). v5 (critic 3 F2): RND runs at all 24 V1 cells on every family (DV4's 7 IFM cells are
  now the cut-Q5 fallback), because the citation rule T3b needs the RND cell at the cited site.
- PC-final: answer O1 at Llast@K. Harness: must equal clean B at r bit for bit on every item, else stop.
- PC-site (subset): O1 on ALL positions at each V1 site; exact on non-injection models (stop if not); on the
  injection twin exact from L0@4 on, before that reported as what re-injection restores.
- SZ: source O1 and RND at Llast@K. Harness: must equal clean A bit for bit, else stop.
- References: clean A, clean B (prompt swap = ceiling).

## 4. Arms

- V1 Placement (E1): O1 and RND at {source span, read position} x 12 sites (48 rows), O2 at 4 cells.
- V2 PC-site: 12 rows on the subset (first 40 kept Ouro items; IFM 30 on ifm_p1_ouro and ifm_d112, 20 on the
  injection twin, v4).
- V3 Presence (supporting, "causal": false): pair projection at r, s(h) = (h - (mA + mB)/2) . (mB - mA) /
  |mB - mA|^2 + 0.5 with mA, mB the clean A/B states at r at the site (n/a if |mB - mA| < 0.01 |mA|). Validation
  (F5): the R2-context A and B states must project near 0 and 1; per item the site passes if s(R2-B) - s(R2-A) >=
  0.5, and a cell's presence gate passes if >= 50% of its items pass. Recorded at every V1 answer site and at the
  undo candidate slots, for clean R2 runs and for source O1 at Llast@1 and answer O1 at Llast@1-3.
- V4 Read window (E6, P9), phase W on the subset: suffix knockouts of span reads from boundary b in {start (full
  knockout, positive control and ceiling), after Llast@1, after Lmid@2, after Llast@2, after Lmid@3, after
  Llast@3, after Lmid@4} (IFM/d112: every call, or calls after slot 27, 42, 55, 70, 83, 98), direct (query r) and
  relay (every query after the span end): 14 rows. Plus one equal-size random non-span relay knockout over all
  calls (F11 control). dep(b) = (m_KO - m_A) / (z_A(tA) - z_A(tB)) on clean A, first-token margin. dep = 1 is the
  tA / tB tie (m_KO = 0); clean B is dep = D / (-m_A) (v6, review A-F4/C-F7). v6 (review B-F1): the random keys are
  drawn from positions 1 .. min(span) - 1 only, so every relay query sees each key as it sees the span (the v5 pool
  1 .. r - 1 also held chain positions after the span, hidden only from the queries after them).
- V5 Undo (E3), two phases (F7). Candidate restore slots c in {Lmid@2, Llast@2, Lmid@3, Llast@3, Lmid@4} (IFM
  no-inj and d112: slots 42, 55, 70, 83, 98). v2.1: Lmid@2 and Lmid@3 were added after the debug run (section
  13), before any full-run row, because on Ouro-1.4B the span reads ended inside loop 2 and no loop-end candidate
  could meet rule (a). Source O1 at Llast@1 (d112 layer 27), then the clean A source state restored at c.
  - Phase W (subset): for each c, rows U_c (edit + restore), B + KO_rel(after c), and U_c + KO_rel(after c).
    e_carried(c) = [m(U_c + KO_c) - m(KO_c)] / [m(B + KO_c) - m(KO_c)] (items with denominator >= 2 only).
  - Frozen rule: c is inside the window iff (a) mean dep_rel(after c) >= 0.2 x mean dep_rel(full) (the span is
    still read after c), and (b) the edit was read before c: (b1) mean presence s_c(r) under the source edit >=
    0.3, or (b2) mean e_carried(c) >= 0.2 over >= 10 items. The code writes `window_<model>.json`; the slots are
    appended to this plan (section 13), the plan is re-hashed, then phase M runs U at the in-window slots on all
    items. A model with no in-window slot is reported "no in-window slot" (undo uninformative), with the subset rows.
  - v4 (plan critic 2 F3): each slot stores the clause that selected it (`selected_by` = "a+b2" or
    "a+b1_only"). A slot is informative for "undo reverts" only if it passes (a) and b2: at a b1-only slot nothing
    carried may survive (e_carried about 0), so a revert is forced with or without a rebuild. b1-only slots are
    run and reported as uninformative; the undo predictions score only informative slots. Per slot (subset):
    e_carried(c) [cluster CI] and e_U / e_carried(c) (first-token e_U, ratio of sums, cluster CI) = the share of
    the carried effect that the re-read overrides. IFM subset raised from 20 to 30 items on ifm_p1_ouro and
    ifm_d112 so that b2 can reach its 10-item minimum (the injection twin has no undo and stays at 20).
  - Reported: e_U, e_U / e_src(Llast@1), at every slot (subset) and at the selected slots (all items).
    Not run on ifm_p1_ouro_inj (A5). The optional "stays" control is not run (cap).
  - v5 rule (critic 3 F1; replaces the frozen v4 rule above, which is kept as a reported diagnostic). Everything in
    first-token logits m = z(tB) - z(tA) on the same W-subset items with D >= 2. Per item: tot = m(source O1 at
    Llast@1) - m_A; car(c) = m(U_c + KO_rel after c) - m(KO_rel after c), the part of the edit that reached non-span
    positions before c (under the relay knockout after c no later call reads the span, so the restore at c cannot
    act); U(c) = m(U_c) - m_A.
    (a) influence: removal(c) = 1 - sum car / sum tot >= REMOVE_MIN = 0.2, i.e. the post-c span reads carry at least
    0.2 of the source edit's dm. Necessity (v4 clause a: dep_rel(after c) on clean A) is reported next to it.
    (b1) presence s_c(r) >= 0.3 (unchanged). (b2) mean car / D >= 0.3 with 95% cluster CI lower bound > 0.15.
    (a) and b2 need >= MIN_CL = 10 items and >= 10 clusters (debug runs use 1, code path only).
    Selected = (a) and (b1 or b2); informative = (a) and b2 (unchanged logic).
  - v5 phase M (critic 3 F6): at every selected slot each phase-M item also runs UKO_c and KO_c (2 forwards per
    slot), so survival can be scored on items that did not select the slot. Survival s(c) = sum U / sum car (ratio
    of sums, 95% cluster CI), scored only on phase-M items outside the W subset; the W-subset value is reported as
    `selection_data_W`.
  - v6 rule for b2 (review C-F1, audit C8/P8; replaces the v5 b2 above). car is measured under the relay knockout
    after c, so its ceiling is D_KO(c) = m(BKO_c) - m(KO_c), not D. b2 = mean car / D_KO >= 0.3 with 95% cluster CI
    lower bound > 0.15, over items with D_KO >= 2 logits (the v4 e_carried form with the v5 CI), >= MIN_CL items and
    clusters of its own. car / D and D_KO / D are reported beside it per slot. Debug evidence that motivated it [ran,
    v5.2 debug, n = 2-3]: ouro Llast@2 car / D 0.32 while car / D_KO 0.94 (D_KO / D 0.35). Phase M adds the BKO_c
    row (1 forward per selected slot) so the ceiling is checked on the M-only items that score T6
    (`carried_over_DKO_scored`, `DKO_over_D_scored`). T6 (survival) is unchanged.
- V6 Carry gain (A2, A6, F1-F3). One clean-A forward and backward with zero leaves at r at every site of the
  model's set (G_full or G_red). g(j,l) = d m_final / d H_{j,l}[r] (first-token margin); G(j,l) = g . (H^B -
  H^A)_{j,l}[r]. Per item, O1 at the top-3, median-3 (sorted ranks n/2-1, n/2, n/2+1) and bottom-3 sites by G,
  at a = 1 and a = 0.25.
  - Dev search over the same set (F2): O1 at every site of the set at a = 1 on dev items. Dev-tuned site (v3,
    review MS7) = argmax of mean dm / D over dev items with D >= 2 (the reported metric; v2 used mean raw dm,
    which weights high-D items); it is used at both doses (the selector's ranking is also dose-free).
  - Baselines on test items, each at a = 1 and 0.25: dev-tuned site, uniform Lmid@K (= latest site in the set),
    random site (one site of the set per item, seeded by sha256(pair_id|"rnd_site")), and (v3, review MS2) a site
    drawn at random from the selector's own top 3 (its expected dm / D and expected hit). Rows deduplicated against
    V1 and the verify rows.
  - Dev items (v3, review MS2/F2/MS5): every site is verified at a = 1, so the A2-1 endpoints are read there:
    tolerance hit over the full set vs band chance, full-set regret, Spearman; the same within loops 1..K-1 with
    the Llast@K-1 readout-proximity baseline; and the A7 false-negative rate (T7). The selector is not fitted on
    dev items, so dev items are a fair test of it; only the dev-tuned baseline must be read on test items.
  - v4 (plan critic 2 F2, F1): the A2-1 hit endpoint is the paired difference selector hit - latest-site hit
    (same tolerance band, eligible dev items, cluster CI), over the full set (latest = Lmid@K) and within loops
    1..K-1 (latest = Llast@K-1). Band chance stays as a diagnostic. Diagnostic for critic F1 option (a): within
    loops 1..K-1, dm / D at the selector's top-1 minus dm / D at Llast@K-1 (dev, a = 1, all complete items).
    The claim "the selector adds information beyond readout proximity" is made only where the 95% CI lower bound
    of a hit difference is > 0 (`adds_beyond_latest`).
  - Cost (F2): `carry_gain.cost_fwd_equiv` = selector 1 forward + 1 backward (nominal 3, plus the measured
    backward/forward time ratio) vs dev search |set| x n_dev forwards.
  - Datasets: VarBench (Ouro all kept, IFM first 60), D1, D2, D4 (Ouro 80 each, IFM 40 each), VBD (v5: Ouro 80,
    IFM 30).
  - v5 gate (critic 3 F4): no V6 row is interpreted until B-K17-CGCHECK passes (its full run, not its debug) AND
    this family's FD row passes. FD row (`--fd_check`, 4 VarBench items per model, G_red sites): pass 1 loads the
    model in bf16 and computes G as the run does; pass 2 loads it in fp32, computes G in fp32 and the exact O1 dm(a)
    at a in {1e-3, 1e-2, 0.1}. Pass rule (frozen, same form as CGCHECK H11): at a = 1e-2, over (item, site) with
    |aG| >= 1e-4 logits, median |dm - aG| / |aG| <= 0.05 and sign agreement >= 0.95. Also reported: the bf16 vs
    fp32 G relative difference (median, p90) and whether the bf16 and fp32 top sites agree (`fd_<model>.json`,
    checks.json `carry_gain_gate`). Until the gate opens, every A2 prediction row is "not scored".
    v5.1 (2026-10-08): the code also required a third condition that the plan did not state, "bf16 top site equals
    the fp32 top site on >= 0.9 of the FD items and FN-class agreement >= 0.95". With 4 items one near-tie closes the
    gate (v5.1 debug: ouro 3/4). The third condition is now written here in the T7 form: on every FD item with D >= 2,
    the fp32 G at the bf16 top site is within 0.05 D of the fp32 maximum (>= 0.9 of items), and FN-class agreement
    >= 0.95 (`bf16_agree`, `top_in_fp32_band`; exact top-site equality stays reported). Gate = CGCHECK pass AND
    family FD pass AND this bf16 condition.
  - v6 gate (review A-F3, A-F6; replaces the two v5.1/v5.2 sentences above for the gate decision). (1) CGCHECK pass
    = the full-run checks.json only: verdict PASS, no failed or insufficient cell, and sha256 equal to
    `46a4e86580839de7793e291ef14ac681cd2ab33784e38732a23441d30149ee1f` (artifacts_spark/k17_cgcheck/checks.json; the
    debug file also says PASS, with insufficient cells D1/SRC and D2/SRC). (2) Family FD pass as v5. (3) bf16
    condition, frozen band rule, scored on fresh items (`--bf16_check`, `bf16_<model>.json`): the first 30 S200 items
    per model that pass the clean filter and are not FD items, the model's run set (G_full on the 1.4B models, 74
    sites; G_red elsewhere), one clean A/B pass and one forward + backward per item per precision (bf16 as the run;
    fp32 with TF32 off). Pass = >= 30 items with D >= 2, fp32 G at the bf16 argmax within 0.05 D of the fp32 maximum
    on >= 0.9 of items for the full-set argmax AND for the argmax inside loops 1..K-1 (the 'A2-1 ltK' selector), and
    FN-class agreement >= 0.95. The Clopper-Pearson 95% lower bound of each band rate is reported (30 / 30 gives
    0.884, so the point rule is what is frozen). The 4-item FD values stay reported (`fd_bf16_*_4_items`), not gating.
  - v6 G_PI row (PLAN G4, review C-F4). Path-integrated gain per site: G_PI(s) = mean over a in {0.25, 0.5, 0.75, 1}
    of g_s(a) . delta_s, where g_s(a) is the gradient of the first-token margin at the state with only site s set to
    lerp(H^A_s, H^B_s, a) at r (a 4-point mean of the gradient along the single-site path; AtP*'s QK fix and
    GradDrop are not applied). Cost: 4 forward + backward passes per site. To keep it affordable it runs on G_red
    sites (17; a subset of G_full), on dev items only, on VarBench, D2 and VBD, at most PI_N dev items per dataset
    (pilot 20, S1 40). Reported (not a frozen prediction), `carry_gain.<m>.<ds>.path_integrated`: on dev items with
    every G_red site verified at a = 1, G_PI hit, G hit on the same sites and items, the paired G_PI - G hit, G_PI
    hit - latest-site hit, regrets, Spearman, the same inside loops 1..K-1, and the cost in forward-equivalents
    (`cost_fwd_equiv.path_integrated_*`). Laptop debug [ran]: 68 passes, about 97-110 forward-equivalents per item on
    Ouro-1.4B, against 74 forwards for the full dev search.
- V7 Answer-cell dose and matched norm (F5), subset: answer O1 at a in {2, 4} at Llast@1-3, and one row per
  loop with direction unit(H^B - H^A)[r] scaled to the source O1 edit's summed |delta| over the span at the same
  site (9 rows).
- V8 VBD, derived use (v5, critic 3 F5; items I8). Ouro x4: V1 (O1 and RND at the source, which is the prompt
  span plus the chain positions of x, and at r, 12 sites, first-token e, 48 rows) plus PC-final and SZ exactness,
  plus V6 with the dev search. IFM x3: PC-final, SZ and V6 only (shared-host cap). VBD is the real-math computed dataset
  of A2-2. No W phase, undo, O2 or presence on VBD.

## 5. Metrics, strata, statistics (frozen)

- Endpoints (F4, EXPERIMENTS 3c). Full-span log-odds LO = sum_k log p(restB[k]) on SA' - sum_k log
  p(restA[k]) on SA, k from r; dLO = LO(edit) - LO(clean A); D_LO = LO(clean B on SB/SB') - LO(clean A);
  e_LO = dLO / D_LO where D_LO >= 2. On items without a continuation LO = first-token margin. Computed for the
  O1 V1 rows, PC-final, SZ, RND at L0@1 and Llast@1 (both positions) and the undo rows. First-token margin (all
  rows): m = z(tB) - z(tA) at r, dm, dz(A), dz(B), p(B), flip (top-1 = tB), e = dm / D (D >= 2).
  Primary endpoint by edited position (v3, review MS1/F1, frozen before the full run):
  - Source rows, undo rows, SZ: e_LO.
  - Answer-position rows (V1 answer cells, RND at the answer position, PC-final): first-token e. Reason: an edit at
    r changes only position r; the continuation positions r+1.. on SA and SA' still read A's prompt span, while
    D_LO is computed with B's prompt, so e_LO of an answer edit is capped at D / D_LO per item. The PC-final row
    shows it exactly: dLO(PC-final) = D, so e_LO(PC-final) = D / D_LO (debug: 0.24 on ouro26 gsm8k-1050, cell
    means 0.65-0.91 on Ouro while first-token e = 1.00). Answer cells also report e_LO_over_pc = dLO / D (= e_LO /
    e_LO(PC-final) per item) next to the PC-final e_LO ceiling. Cell, strata, TOST, answer classes and
    predictions all use this rule (`primary_f` in the code).
  V4, V6, V7 and presence use the first-token margin (the selector's target). Ouro rows also store per-loop
  readouts at r. |delta| per edited position and its sum (P1); per-item clean margins (P14).
- Answer-cell classes (F5, P7), per cell and dose, loops 1..K-1 only. Ouro, every answer site (L0, Lmid, Llast
  of loop j): e_own(j) = (m_edit(j) - m_A(j)) / D read at loop j's own trained exit, with the final D (for O1 at
  Llast@j this equals D_j / D). v2.1: the e_own gate now also covers Ouro L0/Lmid (the edited state passes
  through the rest of loop j to its exit), which is a stronger presence check than R2. "Not carried" only where
  mean e_own(j) >= 0.5 and final e / e_own(j) < 0.25. IFM/d112 (no trained per-loop exits): "not carried" only
  where the V3 presence gate passes and final e < 0.25. "Carried" where final e >= 0.5. Every other cell is
  "gate fails (absent or overwritten within loop j)" or "partial" (v4, plan critic 2 F5: e_own(j) is read after
  the rest of loop j has run, and a position's own MLP and attention cancel about 95% of a carried change within
  one loop (audit_path_ouro F7), so a failed gate cannot tell an absent answer state from an overwritten one).
  Classes at a = 1 (all items) and at a = 2, 4 and matched norm (subset). The R2 gate is weak (debug: it passed
  at 11 of 12 Ouro sites, because R2 changes only the distant 4-shot block, so it tests context-invariance of the
  A/B axis, not answer content).
  Digit probe (v4, plan critic 2 F5/F7, replaces the trained 10-class probe): on the stored clean A and B states
  at r (`states_<model>.pt`, every V1 site, both families). Label of a state = the digit of its own number at the
  first character where x and x' differ (items where both are digits). For each state, leave its cluster out,
  centre on the mean of the other clusters' states, and predict the closer (cosine) of the two class means for
  digit(x) and digit(x') (chance 0.5, no trained parameters). Accuracy per site = mean per-item correctness,
  cluster CI. Positive control (the known-present case): the probe at the readout-feeding state Llast@K, from
  which the clean model reads tA / tB. Labels: present (acc >= 0.75), absent (acc <= 0.60), else unclear; no
  label unless the PC accuracy is >= 0.80. Citation (`cite_as`): a "gate fails" cell is cited as "no answer state
  at site" only where the probe says absent, as "present, overwritten within loop j" where it says present; an
  IFM "not carried" cell is cited only where the probe says present. Note: for O1 at a = 1 the edited state equals
  clean B's state at the site, so probing the clean B state is probing the edited state.
  v5 power control (critic 3 F1, replaces the "no label unless PC >= 0.80" rule): at EVERY probed site, each state
  gets a fixed random unit code of its own digit (seed sha256("k17vb-plant:" + digit)), scaled to PLANT_F = 0.5 x
  its item's |h_A - h_B| at that site, and the same probe runs on the planted states. Labels: present (acc >=
  0.75); absent (acc <= 0.60) only where the planted control is detected (planted acc >= 0.80), else "no power at
  site"; otherwise unclear. The Llast@K accuracy is still reported (`pc_Llast@K`) but no longer gates labels.
- Citation rule T3b (v5, critic 3 F2, frozen before phase W). For every V1 cell where O1 and RND both run, checks.json
  holds the RND cell's T3 verdict, its dz(A) and dz(B) next to the O1 cell's, and the paired per-item difference
  e(O1) - e(RND) with a cluster CI (on e_LO where both rows have it, else first-token e). A class, threshold
  verdict or TOST null may cite the O1 value alone only where the RND cell at the same site and position passes T3.
  Otherwise it cites the paired difference and the site is labelled "nonspecific damage" (`rnd_pair.cite`; '*' in
  summary.md). A cell without an RND row is not citable. The same rule applies to answer classes (final e =
  O1 - RND per item at such sites), to TOST nulls (TOST on the cited series), to the VBD cells, and to the A2-2
  working-site fraction at V1 sites inside the set (`working_site_rnd_adjusted`).
- T3b v6 (review C-F2; replaces the v5 rule above before any full-run row). O1 at a = 1 sets the state to clean B's
  own state, so it has no random-damage component to subtract, and damage at r moves the margin toward the tie
  (positive e at 40 of the 42 failing debug cells), so O1 - RND biased cited values down (v5.2 debug, VBD answer
  Llast@1: O1 e +0.08 / +0.15 on ouro26 / ouro26_think, RND +0.72 / +0.78, cited difference -0.64 / -0.62). Rule:
  the cited value is always O1 (a
  cell without an RND row stays not citable). RND is a gate: where the RND cell at the same site and position fails
  T3, a lower-bound claim ("works", e >= threshold, class "carried", a working site) also needs the O1 - RND paired
  cluster CI lower bound > 0 (`rnd_pair.lower_gate`); else the row is missed with that note. Upper-bound, null (TOST
  on the O1 series) and "not carried" claims cite O1 alone, labelled "RND fails T3". Answer classes use O1 for both
  final e and e_own. A2-2: the upper-bound rows (retrieval, twins) use the O1 working-site fraction, the lower-bound
  rows (computed, the regime row's VBD side) the gated fraction (`working_site_fraction_dev_rnd_gated`).
- Strata (A1): ST1 source tokens 1 vs >= 2 (per tokenizer). ST2 first digit of x and x' shared vs differs. ST3
  digits 1 / 2 / >= 3. ST4 prompt-length tertile, cuts by rank on the S200 prompt lengths for both tokenizers (v3,
  review F5: v2 applied Ouro t* tertiles 651 / 681 to prompt length, giving 155 / 36 / 9; v3 S200 counts 67 / 66 /
  67 Ouro and 68 / 65 / 67 IFM [ran]). ST5 chain distance <= 10 / 11-40 / > 40 tokens. ST6 paraphrase mode (one:
  aligned). "cont" (continuation >= 2 tokens) yes / no. Every O1 cell reports ST1-ST5 and cont. D1: pos_len,
  subject_pos0. v5 (critic 3 F7): ST1e = number of source positions whose A and B tokens differ, 1 vs >= 2. On Ouro
  digits, a shared leading digit has identical causal context in A and B, so O1 and RND at that position are exact
  no-ops and ST1 (span length) mixes 1-edit and 2-edit items. ST1e is used for P-ST and for the injection-twin TOST
  (`tost_by_ST1e`); ST1 stays as span length. ST1e is also a carry-gain stratum and an A2 item stratum.
- CIs: 95% percentile bootstrap over clusters, ratio of sums, 1000 resamples, seed 0. All paired differences use
  the same cluster bootstrap on per-item differences (F9a): VarBench by problem, D1 dyadic over subjects (v3), D2
  by edited operand, D4 by content cluster. Every regret and FN CI uses the dataset clusters (v3, review MS8).
  Stratum differences (P-ST) use a cluster bootstrap of the difference of stratum means (v3, review MS4). Window:
  simultaneous 95% band over the 14 cells (max-statistic bootstrap, P9). V4 dep excludes items with -m_A < 2 (v3,
  review MS6) from dep, the band and rule (a); the excluded count is reported.
- Nulls (F6, EXPERIMENT_MATRIX S5): a cell the paper may cite as a null (injection-twin source loops 1-3, answer
  cells, undo reverts, RND) is "null by TOST" only if the 90% cluster-bootstrap CI of e lies inside [-0.10, 0.10];
  otherwise it is reported as "below <95% upper bound>". Expected 90% half-width = 1.645 x SD(e) / sqrt(n). Debug
  SDs over null-cited cells (n = 3 per model, rough): median 0.005-0.014, max 0.15-0.23. With the max SD: 0.023-0.027
  at n = 120 (Ouro) and 0.008-0.048 at n = 60 (IFM). So IFM stays at 60 (F6 rule not triggered).
- Cross-model comparisons only on the intersection of kept items and on strata every compared model covers (P3).
  v3 (review MS4/F4): `cross_model` holds, for base vs Thinking, 1.4B vs 2.6B, twin vs inj vs d112, every V1 O1
  cell on the intersection (n, strata counts) with each model's mean and the paired per-item difference with the
  problem-cluster CI. Pairs share a tokenizer, so each intersection item has the same strata in both models. P-ST
  is scored from these cells; P-O2 pairs O2 with O1 on the r2_pass items (first-token e on both).

## 6. Frozen thresholds

- T1 cell works: mean e >= 0.5 and flip >= 0.5. Fails: mean e <= 0.15 (cited as null only by TOST). Else partial.
- T2 answer classes as in section 5.
- T3 random control passes: |mean e| <= 0.05 and flip <= 0.02 (cited as null only by TOST).
- T4 PC-final, PC-site (where exact), SZ: bit-identical on every item, else stop and report first.
- T3b citation rule (v5): section 5.
- T6 undo (v5, critic 3 F1/F6; replaces the v4 e_U thresholds): reverts if the 95% cluster CI upper bound of the
  survival s(c) is <= 0.3; stays if its lower bound is >= 0.7; else partial. Scored only at slots selected by (a)
  and b2, and only on phase-M items outside the W subset. e_U (e_LO and first-token) is reported, not scored.
- T7 carry gain (v3, review MS2; frozen before the full run). Eligible item at dose a: D >= 2 and verified dm range
  (max - min over the verified sites) > 2 logits at a = 1, > 0.5 at a = 0.25; the eligible count is reported. Hit
  (tolerance rule) = dm at the top-predicted site >= max verified dm - 0.05 D. Chance = mean over eligible items
  of the fraction of verified sites inside that band (the hit rate of a site drawn at random from the same set).
  Where: A2-1 hit and regret are read on dev items over the full set (every site verified at a = 1, chance = band
  fraction, 1/|S| when one site is in the band); the dose clause compares the verified 9 at a = 0.25 and a = 1 on
  all items; test items add the baseline "random site in the selector's top 3" (expected hit and dm / D). The v2
  strict hit (strictly larger than all others) is kept as a diagnostic only: with saturated sites it scores
  rounding, and a selector that orders its top 3 at random scores 1/3 = 3x the 1/9 chance by construction. Co-primary:
  regret = (best verified dm - dm at the top-predicted site) / D (D >= 2). Spearman(G, dm) per item only on
  eligible items, with the count. False-negative rate (A7, review MS5): on dev items, over every site with
  |G| < 0.05 D, the fraction whose verified e >= 0.5 (carried), cluster CI; also reported: the fraction of carried
  sites with |G| < 0.05 D. v4 (plan critic 2 F2): the scored A2-1 hit endpoint is hit - latest-site hit (section 4
  V6); band chance is a diagnostic only. v5 (critic 3 F4): every T7 row is "not scored" until the V6 gate opens.
  v6 (review A-F1): the band at dose a is HIT_TOL x a x D (0.0125 D at a = 0.25), the same scaling as the
  eligibility threshold, because dm shrinks with the dose. The dose clause is scored on the items eligible at both
  doses, as the paired per-item difference hit(0.25) - hit(1) >= -0.10 with a cluster CI (`dose_pair`); hit - band
  chance at each dose is reported beside it.
- T8 (v5: counted in clusters) a model with fewer than 40 kept VarBench clusters is reported and not interpreted:
  every non-harness prediction row of that model (and every cross-model row that includes it) is "not scored".
- MIN_CL (v5): b2, clause (a) and the A2-1 loops < K row need >= 10 items AND >= 10 clusters (the S1 pairs pool
  has up to 6 pairs per problem, so items and clusters differ there).

## 7. Budget, order, cut rules

- Caps: fast-host GPU1 240 GPU-min (guard 230), shared-host 120 GPU-min (guard 112). GPU-min = process wall time incl.
  load, with co-tenants logged (nvidia-smi snapshot at launch and in every heartbeat). Before each launch: check
  nvidia-smi, free memory, and for `k17_ko240` / production verification processes on GPU1 (F12a). Launch a 2.6B
  process only with >= 40 GB free on GPU1; on shared-host only with >= 40 GB free.
- Debug first under `out/k17_varbench/debug/` (F12b): 3 VarBench + 2 per A2 dataset per model, all checks,
  per-row timing, backward/forward ratio. The projection replaces the [belief] timings below.
- Phase order per host: phase W (subset: V2, V4, V5-W, V7) for every model, append the slots to section 13,
  re-hash, then phase M (V1, V3, V5-M, V6).
- fast-host order (v5, critic 3 F3): ouro, ouro26, ouro_think, ouro26_think, one process per model at a time on
  GPU1 (v4 order: ouro, ouro_think, ouro26, ouro26_think).
  Measured in the debug run [ran, fast-host GPU1, no co-tenant]: forward at T about 700: 0.066-0.079 s (1.4B),
  0.131-0.159 s (2.6B); backward/forward 2.3-2.4 (VarBench), 2.5-3.2 (A2). Phase-M VarBench item: 1.4B 5.0 s (test),
  7.2-7.4 s (test, cont), 10.7-15.3 s (dev, G_full search); 2.6B 11.2-18.0 s. Phase-W item: 3.4-4.7 s (1.4B),
  7.0-9.4 s (2.6B). A2 item: 1.4B 0.76 s (test), 3.1 s (dev, G_full); 2.6B 1.2-1.4 s (test), 1.7 s (dev). Clean
  pass on VarBench: 3/6, 3/5, 3/5, 3/4 scanned (ouro, ouro_think, ouro26, ouro26_think), so about 100-150 of S200
  are kept per Ouro model, not 200.
  Projection (kept 120, dev half): 1.4B: M 19 min + W 3 + A2 8 + load and checks 1 = 31 min each; 2.6B: M 28 + W 6 +
  A2 6 + 2 = 42 min each. Total about 146 min (206 min if all 200 S200 items pass). No cut is needed.
  Cuts in order if a co-tenant slows GPU1 and the projection passes 230: R1 D4 A2 to 40 items on ouro26*. R2 subset 40 -> 20 on ouro26*. R3 RND
  at L0/Lmid answer cells dropped on ouro26*. R4 VarBench n 200 -> 120 on ouro26* (order is already random, so
  splits and strata stay balanced). R5 D1, D2 A2 to 40 on Thinking models. R6 VarBench dev search on the 1.4B
  models from G_full to G_red.
  Never cut: O1 at 24 cells with full-span, PC-final, SZ, RND at L0@1 and Llast@1, undo at selected slots, V6 on
  VarBench (G_red at least).
  v5 additions on fast-host: VBD (V1 48 rows + V6, n 80 per model), UKO_c and KO_c on every phase-M item at selected
  slots (2 forwards per slot), and the FD check (4 items, two loads per model). Their measured minutes are in
  section 13 (v5 debug). New first cut R0: VBD n 80 -> 40 on ouro26*; R0b: VBD V1 rows dropped on the Thinking
  models (V6 kept). Never cut: VBD V6 on ouro and ouro26, the FD check.
- shared-host order: ifm_p1_ouro, ifm_p1_ouro_inj, ifm_d112 in one xllm_venv process (shared with B-K17-PUBLIC,
  B-K17-CLOSERS, B-K17-PARCAE). Rows per item reduced (DV4): O1 24, RND at 7 cells (source L0@1, Llast@1, Llast@3,
  L0@4, Llast@4 (SZ), answer Llast@1, Llast@3), O2 4. Measured in the debug run [ran, shared-host, with B-K17-CLOSERS
  running]: forward 0.18-0.40 s at T about 540, backward/forward 2.0-4.0, phase-M item 10-34 s, phase-W item 7-21
  s, A2 item 1.1-3.0 s, load 21-94 s. Projection: about 25-30 min per model, 75-90 min for three, under the 112
  guard; v4 adds 10 W items on ifm_p1_ouro and ifm_d112 (about 2-4 min each). Cuts if a co-tenant pushes the projection past it: Q1 D4 A2 skipped. Q2 D1 A2 skipped. Q3 O2 only at Llast@1. Q4 VarBench
  n 60 -> 40 on inj and d112. Never cut: O1 24 cells, PC-final, SZ, RND at L0@1 and Llast@1, undo (no-inj, d112),
  V6 on VarBench and D2.
  v5 additions on shared-host: RND at all 24 cells (+17 forwards per item), VBD V6 (n 30), the FD check (4 items, fp32
  reload). Q1 (D4 A2 skipped on IFM) is applied before the run to make room (DV16). New cut Q5: RND back to the 7
  v4 cells, with the O1 cells at the other sites "not citable" (T3b). Never cut: VBD V6 on ifm_p1_ouro, the FD check.
- v6 additions (DV18). Per Ouro model on fast-host (v5.2 debug rates [ran], counts [belief]): the bf16 check (30
  items, two loads) about 2-4 min; BKO_c in phase M, 1 forward per selected slot per item, under 1 min; G_PI on 20
  dev items x 3 datasets x 68 passes: about 11 s per item on 1.4B and 22 s on 2.6B, so 11 and 22 min. Pilot
  projection with v6: 1.4B about 31 + 15 = 46 min, 2.6B about 42 + 27 = 69 min, plus the v5 VBD rows (80 items, about
  9 and 13 min), so Ouro x4 is about 275 min against the 230 guard. This does not fit: first cut R-PI (no G_PI on
  the Thinking models), then the existing R0-R6. On shared-host (IFM) G_PI costs about 51 s per dev item (v6 debug, 0.35 s
  per forward): Q-PI (PI_N 20 -> 5 per dataset, about 13 min per model) is applied before the IFM pilot; the IFM bf16
  check is about 5-10 min per model. Both caps need the coordinator's decision (open issue in the review log).
- Results are rewritten atomically after every item; a rerun skips finished pair_ids. Before any relaunch, check
  the remote pid and heartbeat. Stop with SIGTERM and an anchored pattern, e.g.
  `pkill -TERM -f '^[^ ]*python k17_varbench.py --models ouro26'`.
- v5 migration rule (critic 3 F3, HOST RULE). (1) Every fast-host job is one model in one process, resumable by
  pair_id and killable at any time. Order: ouro, ouro26, ouro_think, ouro26_think (base models first, Thinking
  models are cut first). (2) On the PI's "free fast-host": stop every k17_varbench process on fast-host with
  `pkill -TERM -f '^[^ ]*python k17_varbench\.py'` (after checking `pgrep -af '^[^ ]*python k17_varbench\.py'`
  lists only this track's jobs), confirm GPU1 is empty with nvidia-smi, pull `out/k17_varbench/full/`, and record
  the stop time in section 13. (3) One host role per results file: the code refuses to continue a `vb_<model>.json`
  written under another host role (`host_mix_check`), every item carries its host role, and the hook checks rerun
  whenever the role or the code hash changes. A model stopped mid-way on fast-host keeps its fast-host file as a
  partial result (reported, not merged) and restarts from scratch on shared-host in a separate out dir
  (`out/k17_varbench/full_shared/`), phase W then phase M on shared-host, so its slots are selected and scored on one host.
  Models already finished on fast-host stay. (4) Cross-model contrasts pair two models only when both ran on one
  host role (`cross_model.<pair>.cross_host`; P-ST rows of cross-host pairs are "not scored"). v6 (review C-F5):
  every launch record holds the GPU name; a results file refuses a launch on a different GPU (as for a different
  role), and the hook checks are keyed on (role, GPU name). (5) shared-host budget line
  for the move: needs the coordinator's approval, because the shared-host cap (120 GPU-min) already holds the IFM work.
  Request: ouro only, at n_vb 60, G_red, A2 D2 and VBD only, n 40 each. The GB10 timing per Ouro forward is
  measured in the v5 debug (section 13) and the request's minutes are computed from it there. Measured: 1.16 s per
  Ouro-1.4B forward on GB10 with three co-tenants, so the request is about 4.5 shared-host GPU-hours. It does not fit
  the shared-host cap; on "free fast-host" the default is to stop and keep the partial file, and to resume on fast-host or
  the cluster (S1), not on shared-host, unless the coordinator grants the hours.

## 8. Predictions (written before any run, [belief] unless noted)

Tags: [harness] = true by construction, kept out of the held/missed tally (F10). Endpoint (v3, section 5): source
cells e_LO, answer cells first-token e, at a = 1, mean over kept items. Scoring (v3, review F3):
`predictions_vs_outcomes` encodes every row below with its harness tag and endpoint, a row whose inputs did not
run is "not scored", and verdicts use the point estimate (CIs are in the cited cells). A2-1 hit and regret are
read on dev items over the full set (T7 v3); P-ST cross-model rows use the intersection cells.
v5 (critic 3, re-frozen before any full-run row, DV13): every threshold row on an O1 cell scores the cited value of
rule T3b (O1 where RND at the site passes T3, else O1 - RND); undo rows use T6 v5 on phase-M items outside the W
subset; A2 rows stay "not scored" until the V6 gate opens; T8 counts clusters. Rows changed in v5 carry "(v5)".
v6 (code review 3, DV18, before any full-run row): T3b v6 replaces the sentence above: every threshold row on an O1
cell scores O1, and a lower bound at a site whose RND cell fails T3 also needs the O1 - RND CI lower bound > 0. The
P-ST rows (strata and cross-model) are on O1 in both sides, with each side's RND verdict recorded. "Relay dep >=
direct dep" is no longer a harness row (below). The A2-1 dose clause is scored paired with the scaled band (T7 v6).

| model | source cells | answer cells | other |
|---|---|---|---|
| ouro x4 | L0/Lmid/Llast@1 >= 0.85 (flip >= 0.85), loops 2-3 >= 0.8, L0@4 >= 0.7, Lmid@4 0.2-0.8, Llast@4 = 0 [harness SZ] | Llast@1-3 "not carried" with e_own(2), e_own(3) >= 0.5; L0@1-3 and Lmid@1-3 "gate fails (absent or overwritten within loop j)" (v4 name), final e <= 0.10; L0@4 <= 0.2; Lmid@4 0.2-0.8; Llast@4 = 1 [harness PC] | undo reverts (T6 v5) at every informative selected slot ((a) and b2) |
| ifm_p1_ouro | loops 1-3 >= 0.8, L0@4 >= 0.7, Lmid@4 0.1-0.6, Llast@4 = 0 [harness] | loops 1-3 final e <= 0.10, class "gate fails" at L0/Lmid@1-2 (v4 name); L0@4 <= 0.15; Lmid@4 0.1-0.6; Llast@4 = 1 [harness] | undo reverts (T6 v5) at every informative selected slot |
| ifm_p1_ouro_inj (1-token spans) | loops 1-3 <= 0.10 (null by TOST on the ST1e = 1 stratum, v5), L0@4 >= 0.7 | as the no-inj twin | PC-site loops 1-3 0.3-0.95 (low confidence), = 1 from L0@4 |
| ifm_d112 (mapped) | layers 0-70 >= 0.8, 83-84 >= 0.5, 98 <= 0.4, 111 = 0 [harness] | layers <= 55 <= 0.10, 83 0.1-0.5, 84 0.2-0.7, 98 >= 0.5, 111 = 1 [harness] | undo at 55 (Llast@2) reverts (T6 v5), at 83 (Llast@3) s(c) <= 0.3 point estimate (v5; if selected by (a) and b2) |

- P-RND: every RND cell passes T3 and is null by TOST.
- P-O2 [harness-like, F8]: O2 within 0.15 of O1 at Llast@1 and Llast@3 (context check only; tallied as harness).
- P-ST: on Ouro, source Llast@1 e differs by <= 0.1 between ST1e strata (v5; v4: ST1) and between ST2 strata.
  Thinking within 0.15 of base and 2.6B within 0.15 of 1.4B at every V1 cell (v5: only pairs that ran on one host
  role are scored).
- P-W: full relay knockout dep >= 0.8 (control). Ouro: the last loop still reads the span, scored by influence
  (v5): removal(Llast@3) >= 0.3, with necessity dep_rel(after Llast@3) reported beside it (v4 scored the necessity
  value; critic 3 F1: on the v4 debug necessity is 0.03 while the source edit at Llast@3 still moves the read).
  Relay dep >= direct dep [harness, near-forced]. Selected undo slots include Llast@2 or Llast@3 on at least 3 of 4
  Ouro checkpoints.
  v6 (review A-F4, C-F7): "relay dep >= direct dep" is not forced. dep is signed and unbounded, dep > 1 only means
  the knockout moved the margin past the tA / tB tie, and past the tie the order of two guesses no longer measures
  span reading. The row is tallied [belief] and scored on per-item min(dep, 1) at every boundary
  (`window.dep_clipped_at_1`).
- P-VBD (v5, critic 3 F5; written before any VBD output was read: the v5 debug, n = 2 per model, was already
  running, [belief]): on Ouro x4, VBD source Llast@1 >= 0.8, and
  answer Llast@2 and Llast@3 final e >= 0.5 (the H2 reading: a computed value formed by loop 2 is carried). The
  audit (audit_path_ouro F6: D4/T2 answer edits are lost; only D2 carries) predicts a miss on the answer rows.
- P-A2 (V6), on each model's set; retrieval = VarBench, D1, D4-T1; computed = D2, D4-T2, VBD (v5).
  - A2-1 selector works in both regimes (v4, plan critic 2 F2; written before any full-run row): on dev items
    over the full set at a = 1, selector hit - latest-site (Lmid@K) hit >= -0.10 [belief: under A7 the latest site
    is the predicted best answer site on retrieval, so the selector is predicted to match readout proximity, not
    to beat it]; within loops 1..K-1, selector hit - Llast@K-1 hit >= -0.10 on every dataset with >= 10 eligible
    items (else not scored; retrieval is expected to have few eligible items because no answer site before loop
    K carries [belief]); mean regret <= 0.15 at a = 1; hit at a = 0.25 >= hit at a = 1 - 0.10 (verified 9; v6: band
    0.05 x a x D, paired on items eligible at both doses, T7 v6). The
    v3 clause "hit >= 3x band chance" is withdrawn: a rule that always picks the site nearest the readout passes
    it by construction. "Adds beyond readout proximity" is a claim made only from the CI (section 4 V6), not a
    prediction.
  - A2-2 regime contrast, working-site fraction (dev items, a = 1, fraction of set sites with mean e >= 0.5): on
    Ouro and d112, retrieval <= 0.20 and computed >= 0.35. On the IFM twins all datasets <= 0.20 (no carried
    answer before loop 4).
  - A2-2 is the only scored regime test (v4, plan critic 2 F1 option (b)).
  - v5 (critic 3 F5): computed = D2, D4-T2 and VBD (real math); A2-2 is reported per dataset as before, plus one
    row per Ouro model and d112, "A2-2 regime": VarBench (retrieval) <= 0.20 and VBD (computed) >= 0.35. A regime
    claim is made only from that row; D2 alone gives no regime claim. On the IFM twins VBD joins "every dataset
    <= 0.20".
  - A2-3 (v4: withdrawn from the tally, reported as a diagnostic). v3 text: selector minus uniform Lmid@K (paired
    mean dm / D, test items, a = 1), Ouro only, retrieval >= +0.20, computed <= +0.10. Re-derived from the V1
    rows above: on retrieval Lmid@4 (0.2-0.8) is the best answer site the selector can pick, so selector minus
    uniform <= about 0; on computed data Lmid@4 carries with loops 2-3, so the difference is about 0 as well. Both
    H* and its alternative predict about 0 in both regimes, so the row cannot discriminate. Reported as
    `test_dm_over_D.predicted_minus_uniform`, next to the option (a) diagnostic (section 4 V6).
  - A2-4 selector vs dev-tuned: |difference| <= 0.10 D on every dataset and model (no claim that the selector
    beats a tuned fixed site; its advantage is cost).
  - A2-5 eligible fraction (range > 2 at a = 1) >= 0.6 on every dataset [belief].

## 9. Verification and artifacts

- CPU: `python k17_varbench.py --selftest` (operators, RND norms, O2 offset map, bootstrap, TOST, Spearman, hit with
  ties and permutation chance, regret, window rule, knockout masks on a toy eager softmax, site sets, taint
  filter), `--build_items` (I3 asserts, writes `items_ouro.json`, `items_ifm.json`, `a2_ouro.json`,
  `a2_ifm.json`, `k16_exclusions.json`), `--cpu_check --models ouro` (hook checks a-h on a real prompt).
- GPU: debug run, phase W, slots appended, phase M, pull. Named artifacts: `k17_varbench/summary.md`,
  `k17_varbench/results.json`, `k17_varbench/checks.json`.
- Checks keys: `plan_sha256`, `endpoint` (v3: source rows / undo / SZ e_LO, answer rows first-token e, readout
  float32), `tokenization`, `prefix_filter` (rows dropped: contradictory, ambiguous), `hook_identity`, `readout`
  (dtype, fp32-vs-bf16 head difference), `edit_placement`, `knockout_placement`, `causality`, `items` (scanned,
  kept, A_fail, B_fail, n_clusters, strata ST1-ST5 and cont, e_excluded, pair_equals_gen, run config, A2 counts
  with the D1 split counts and shared-subject check), `pc_final` (exact, e, e_LO ceiling), `pc_site`,
  `structural_zero`, `placement` (model x position x site x operator: endpoint, primary, n_items, n_clusters, e_LO
  [CI], e [CI], e_LO_over_pc for answer cells, flip, dm, strata, pc values, tost; tost_by_ST1 on the injection
  twin's source cells), `strata_diff`, `cross_model`, `random_control`, `o2_context` (with the paired O2 - O1
  difference), `presence`, `answer_classes`, `window` (with dep_excluded_small_margin), `undo_slots`, `undo`,
  `carry_gain` (model x dataset x a: eligible, tolerance top1_hit, band chance, random_in_top3_hit, strict_hit,
  regret, spearman with n; test dm / D at predicted vs dev-tuned vs uniform vs latest site vs random site vs random
  in top 3 with paired cluster CIs; `dev_full_set_a1` with hit, chance, regret, loops_lt_K and false_negative;
  working_site_fraction, dev_tuned_matches_v3_rule, cost_fwd_equiv), `predictions_vs_outcomes` (every section 8
  row, harness tags, endpoints, "not scored" where nothing ran), `predictions_tally`, `model_copies` (repo id and
  home-relative copy path), `downloads` (expected none), `gpu_minutes` (by host role and mode), `co_tenants`
  (GPU memory numbers only).
- v4 keys (plan critic 2): `plan_sha256_at_launch` (per model, one record per launch: phase, plan_sha256,
  plan_frozen_sha256 = hash without section 13, plan_sha_arg), `plan_frozen_consistent`; `undo_slots` with
  `selected_by`, `informative` and `informative_in_M`; `undo.<slot>` with `selected_by`, `informative`,
  `cite_as_rebuild_evidence`, `e_carried_subset`, `e_U_first_token_subset`, `e_U_over_e_carried`;
  `answer_classes.<model>.probe` (n_items, n_clusters, sites with acc [CI], pc_Llast@K, pc_ok, labels) and per
  class `probe` and `cite_as`; `carry_gain` records with `strata` (ST1, ST2: top1_hit, regret,
  test_dm_over_D_predicted; dev full set: top1_hit, hit_minus_latest, regret), `pc_final`,
  `dev_full_set_a1.hit_minus_latest`, `loops_lt_K.{baseline_site, hit_minus_latest,
  dm_over_D_selector_minus_Llast_Km1}`, `adds_beyond_latest`; D4 build `n_components`,
  `k16_components_excluded`, `dev_test_shared_content_keys`.
- Plan hash (v4, plan critic 2 F4, P14). Order: (1) the coordinator writes this file's sha256 into LEDGER.md
  before phase W; (2) every full-run launch passes `--plan <this file> --plan_sha <that sha256>`; the code
  hashes the file at launch, refuses a mismatch, refuses `--phase all` for full runs, and stores the record in
  `vb_<model>.json` before any scored row; (3) after phase W the slots are appended to section 13 only, the
  coordinator records the second sha256 in the LEDGER, and phase M launches with it; the code refuses phase M if
  the plan outside section 13 differs from the phase-W launch (`plan_frozen_sha256`). The summarize-time hash is
  kept as `plan_sha256` but is not evidence of pre-registration.
- v5 code hash (critic 3 F1). Every launch record holds `code_sha256` of k17_varbench.py, the sha256 of the four
  imported modules, `code_version` and `host_role`. A full run refuses to start unless the code sha256 equals the
  "Launch code sha256" line below (read from the frozen part of this file), and phase M refuses a code hash that
  differs from its phase-W launch. Any code change therefore needs a new line here, a new plan hash in the LEDGER,
  and a rerun of the hook checks (they rerun on a new code hash by themselves). Debug runs record
  `code_matches_plan`. The line is written last, after the v5 debug, on the code that will be launched:
  Launch code sha256: `43808060ac3948b6c3c0cd91f0edc06bacfe4254f982ece05c0253dca9f2c0d1` (cluster copy cluster/pilot/k17_varbench.py, final:
  shard version f8d3d427 + cluster patches + --no_gradient_rank + --vbd_v1 (2026-10-11, default unchanged); replaces
  a161b00c of 2026-10-09, which replaced b61194d8)
- v5 checks keys: `code_version`, `host_roles` (per model: file roles, items per role, mixed), `t8`,
  `placement.<m>.<cell>.rnd_pair` (rnd_T3, rnd_primary, rnd_flip, rnd_dzA, rnd_dzB, diff_e, diff_e_LO,
  diff_primary, cite) with `cited`, `cited_label`, `cited_endpoint` and `tost_basis`; `tost_by_ST1e`;
  `strata_diff.<m>."src Llast@1 ST1e 1 - >=2"`; `undo_slots.<m>.slots.<c>` (removal, carried_over_D, survival_subset,
  n_items, n_clusters, presence, necessity_dep_rel_after, v4_rule, selected_by, informative); `undo.<m>.<c>`
  (survival, t6, n_items, n_clusters on M-only items, selection_data_W, cite_as_rebuild_evidence);
  `answer_classes.<m>.probe.planted` and labels with "no power at site"; `answer_classes.<m>.a=1.<site>.final_e_basis`;
  `placement_vbd.<m>` (pc_final, structural_zero, O1 and RND cells with rnd_pair); `items.<m>.vbd`;
  `carry_gain.<m>.VBD`; `carry_gain.<m>.<ds>.working_site_rnd_adjusted` and `loops_lt_K.n_clusters`;
  `carry_gain_gate.<m>` (cgcheck_pass, fd_pass, fd, bf16_vs_fp32_G, interpretable); `cross_model.<pair>.host_roles`
  and `cross_host`; `hook_identity.<m>.by_role`.
- v6 keys (DV18): `plan_sha256_at_launch.<m>[].{data_sha256, config, config_sha256, gpu_name}` (G8);
  `output_sha256` (every vb/fd/bf16/window/states file, hashed before analysis); `bf16_check.<m>` and
  `carry_gain_gate.<m>.{cgcheck, bf16_check, fd_bf16_vs_fp32_G_4_items}`; `placement.<m>.<cell>.{lower_gate,
  rnd_pair.lower_gate}`; `answer_classes.<m>.a=1.<site>.{rnd_fails_T3, carried_rnd_gate}`;
  `undo_slots.<m>.slots.<c>.{carried_over_DKO, carried_over_D, DKO_over_D, n_b2}` and `undo.<m>.<c>.{carried_over_DKO_
  scored, DKO_over_D_scored}`; `window.<m>.dep_clipped_at_1`; `carry_gain.<m>.<ds>.{dose_pair, path_integrated,
  working_site_fraction_dev_rnd_gated, working_site_rnd_gate_closed}`; `strata_diff.<m>.basis`;
  `cross_model.<pair>.cells.<k>.{basis, rnd_T3_A, rnd_T3_B}`; `items.<m>`'s `prefix_filter` with x_word_form and
  prose_unverified_number.

## 10. Slurm scale configuration (30 x 96 GB)

- Config S1 (the one to scale, v3): Ouro x4 x all valid copy-read pairs (1164 rows, 513 problems after the I3
  filters, every pair; cluster = problem) x the full row set of section 4 on every item, with G_full V6 and dev
  search on every model (A7 sets: 74 sites on 1.4B, 146 on 2.6B), full-span on every row, W subset 100 items,
  plus A2 on D1, D2 and D4 at a 300-item quota each. Reachable A2 n (review F9): D1 at most 69 pairs before the
  clean filter on the Ouro tokenizer (section 1 I7: the subject split and the cap of 2 per subject bind, test city
  has 7); D2 at most 509 candidates after the K16 exclusions (600 - 91, debug build [ran]); D4 1481 rows after
  the K16 exclusions, one item per content cluster [belief: 300 reachable]. Batch 1, eager attention, bf16 model,
  float32 readout. Exact commands, per model M in ouro, ouro_think, ouro26, ouro26_think (every flag exists in
  the v3 code, review F9):
  `python k17_varbench.py --models M --data DATA_V3 --out OUT --pool pairs --n_vb 2000 --gset full --n_sub 100
  --a2_n 300 --datasets D2,D1,D4 --phase W --role ouro_host --plan PLAN --plan_sha SHA_W`, then, after the slots
  are appended and the plan re-hashed, the same with `--phase M --plan_sha SHA_M` (v4, plan critic 2 F4; the code
  refuses a full run without both flags).
  v5 (critic 3 F8): S1 is 4 single-model Slurm jobs, one GPU each, nothing else. The `--shard k/n` slice and the
  merge step are not built and not part of the deliverable (DV15): the dev-tuned site needs every dev item and its
  test-item pass 2 runs inside the same process. v5 S1 adds `--datasets VBD,D2,D1,D4 --vbd_n 400` (VBD at every
  valid derived-use problem, 341 before the clean filter) and the FD check per model (`--fd_check`, one short job
  per model). Each model's W and M phases run back-to-back on one node, so the migration rule (section 7) holds on
  the cluster too. The v5 commands are the v4 ones above plus `--datasets VBD,D2,D1,D4 --vbd_n 400`.
- IFM on the cluster (F12c): no IFM checkpoint or xllm environment is known on the cluster (recon_mathcode F11),
  so IFM stays shared-host-only in S1. Moving it needs the three checkpoints (`ckpt/LoopedLM-P1-dense-*`) and the
  xllm-loop tree copied, and a cluster venv built from `spark_setup/build_xllm.sh`.
- Estimate (v3) from the debug timings (section 7) [ran for per-item cost and forward times, belief for the pass
  rate 0.6, so about 700 kept items per model]: Ouro-1.4B: 6.2 s per test item plus 74 x 0.07 s = 5.2 s G_full
  search on the dev half, 8.8 s per item x 700 = 1.7 h, W 100 x 4 s = 0.1 h, A2 (69 + 300 + 300 items) 0.4 h, so
  2.2 h each. Ouro-2.6B: 12 s + 146 x 0.137 s / 2 = 22 s per item x 700 = 4.3 h, W 100 x 8 s = 0.2 h, A2 1.4 h, so
  5.9 h each. Total about 16 fast-host GPU-hours (v2 estimate 19 h with the larger pre-A7 sets). At the cluster's
  measured 2-3x slower rate (recon_mathcode F16): 33-49 cluster GPU-hours; as 4 per-model jobs the wall time is set
  by the 2.6B job (12-18 h); with the shard step written it is about 1 h on 30 GPUs. IFM (shared-host-only): about 0.2 s
  per forward uncontended, 0.32 s with a co-tenant; all 513 problems x 3 models is about 6-9 shared-host GPU-hours. The
  fp32 readout adds one 2048 x 49k (Ouro) or d x 250k (IFM) fp32 matmul per readout position, below 1% of a
  forward [belief, not timed separately].
- Config S1 v6 (the one to scale; supersedes the v3 estimate above, review C-F6). Per Ouro model M, one Slurm job,
  one 96 GB GPU, run in this order on one node (the migration rule holds):
  (1) `python k17_varbench.py --bf16_check --models M --data DATA_V6 --out OUT --fd_dir OUT --role cluster` after
  (0) `--fd_check` with the same flags (FD 4 items, bf16 check 30 fresh items);
  (2) `python k17_varbench.py --models M --data DATA_V6 --out OUT --pool pairs --n_vb 2000 --gset full --n_sub 100
  --a2_n 300 --vbd_n 400 --pi_n 40 --datasets VBD,D2,D1,D4 --phase W --role cluster --plan PLAN --plan_sha SHA_W`;
  (3) the slots appended to section 13, the plan re-hashed, then the same command with `--phase M --plan_sha SHA_M`;
  (4) `--summarize --out OUT --plan PLAN --cgcheck CGCHECK_checks.json`. DATA_V6 = `out/k17_varbench/data_v6`
  (913 valid pairs, 409 problems, VBD 351 problems).
  Estimate from the v5.2 debug per-item seconds on fast-host GPU1 with a co-tenant on the GPU [ran: section 13 v5.2
  rows; per item 1.4B M dev 11.5-14.0 s, M test 6.3-8.9 s, W 4.1-4.9 s, VBD dev 9.5-10.1 s, A2 dev 2.6-2.8 s, A2 test
  0.8 s, one forward 0.07 s, one forward + backward 0.16 s; 2.6B M 12.7-18.0 s with G_red, W 8.2-9.7 s, VBD dev 11.1
  s, A2 1.3-1.8 s, forward 0.14 s, forward + backward 0.33 s]. Counts [belief]: clean pass 0.55 (debug 12 of 22), so
  about 500 kept pairs per model; VBD pass 0.8 of 351 = 280; A2 D1 60, D2 300, D4 300; half of every set dev.
  1.4B: M 250 x 12.7 + 250 x 7.6 s = 1.4 h; W 100 x 4.5 s = 0.13 h; VBD 140 x 9.8 + 140 x 4.3 s = 0.55 h; A2 660 x
  about 1.7 s = 0.31 h; G_PI 120 dev items x 68 x 0.16 s = 0.36 h; FD, bf16 check, loads and checks 0.2 h; total
  about 3.0 h. 2.6B with G_full (146 sites, so a dev item adds 129 x 0.14 s = 18 s over the G_red debug): M 250 x 34 +
  250 x 15 s = 3.4 h; W 100 x 9 s = 0.25 h; VBD 140 x 29 + 140 x 9 s = 1.5 h; A2 330 x 20 + 330 x 1.3 s = 1.9 h; G_PI
  120 x 68 x 0.33 s = 0.75 h; FD, bf16 check, loads 0.3 h; total about 8.1 h. Ouro x4: about 22 fast-host GPU-hours
  (v3 estimate 16; the difference is VBD, G_PI, the 2.6B G_full A2 search and the lower pass rate on more pairs). At
  the cluster's measured 2-3x slower rate (recon_mathcode F16): 44-66 cluster GPU-hours; the wall time is set by the
  2.6B jobs, 16-24 h.
  IFM (shared-host only, not on the cluster): S1 IFM = every valid problem (409) at pass 0.6 = about 245 kept per model, W 30,
  VBD 30, D1 and D2 40, G_PI PI_N 20. At the v6 debug rate (ifm_p1_ouro, 0.32-0.35 s per forward, three co-tenants
  [ran]): M item 33-37 s, W 23 s, VBD 13 s, G_PI 51 s per dev item; per model M 245 x 35 s = 2.4 h, W 0.2 h, VBD and
  A2 0.2 h, G_PI 60 x 51 s = 0.85 h, FD and bf16 check 0.3 h, about 4 h; three models about 12 shared-host GPU-hours. At the
  heavy co-tenant rate measured in the v5 debug (0.85-1.0 s per forward) it is 2.5-3x more, about 30-36 shared-host
  GPU-hours.

## 11. Critic log

- A1-A6: see "Amendments applied". A7 (v3): applied, see "Amendments applied", section 2 sets, T7 and section 14.
- P1 |delta| recorded, matched-norm answer row (V7). P2 n/a (no generation). P3 ST1-ST6, intersection rule. P4 O2
  never broadcast, R2 clean-pass filter. P5 VarBench spans never at position 0 (span index >= 500); D1
  subject_pos0 stratum. P6 n/a. P7 classes with the e_own and presence gates (section 5). P8 relay variant;
  Ouro probabilities from the module; IFM functional check. P9 n_items, n_clusters, max-statistic band, disjoint
  dev/test clusters. P10 D4 equal-length filter and content clusters. P11, P12 n/a (no D3, D5). P13 undo slots
  from the measured window (two phases). P14 per-item margins, plan hash, batch 1.
- critic_k17_paper K1 fixed by A3. K5 F6-F8 in V6 and T7. K6 presence as support only. K7 IFM 3-digit chunks:
  196/200 spans are one IFM token, so the inj twin tests the single-token case.
- F1 (blocker) applied: one answer-site set per model with only the readout-feeding tensor removed (G_full 95 on
  1.4B; G_red 19 elsewhere, see F2), uniform rule Lmid@K, Llast@K only as the PC row. P-A2 rewritten.
- F2 applied: the dev search runs over the same set as the selector (G_full on the 1.4B models, G_red on ouro26
  and IFM, the critic's reduced-grid option), random-site baseline added, cost key added, S1 estimate redone.
- F3 applied: eligibility (range > 2 / > 0.5), no tie credit, permutation chance, random site, regret co-primary,
  regime predictions on non-overlapping metrics (A2-2, A2-3); hit-rate predictions no longer claim a regime
  difference.
- F4 applied: full-span log-odds is primary (one extra forward per row on cont items, rows listed in section 5),
  first-token margin secondary; prefix-validity filter with counts; endpoint keys named.
- F5 applied: e_own normalised by the final D; "not carried" only with e_own >= 0.5 or a validated presence gate;
  "no answer state at site" otherwise; V7 doses and matched-norm row.
- F6 applied: TOST rule for null-cited cells, expected half-widths, IFM n raised to 120 only on debug evidence.
- F7 applied: two-phase window then undo, full knockout as ceiling, slot rule with relay and presence/carry
  conditions, e_U / e_src reported. The optional "stays" control is skipped for the cap.
- F8 applied (option 1): O2 relabelled a context-robustness check at Llast@1 and Llast@3, R2 clean-pass filter.
  Option 2 (cross-item vector) rejected: Ouro digits give 90 (tA, tB) pairs and about 100 dev items, so most test
  items would have no matching dev pair.
- F9 applied: cluster bootstrap for every paired difference; D1 labelled a same-item replication (pool too small
  for fresh subjects), D2 and D4 fresh items; D1 pos_len and subject_pos0 strata; round-robin item order.
- F10 applied: harness tags in section 8 and in predictions_vs_outcomes.
- F11 applied: IFM knockout placement by module-output equality plus the functional span vs random check.
- F12 applied: KO240 and production co-tenants checked before each launch and logged; debug dir moved to
  `out/k17_varbench/debug/`; IFM marked shared-host-only for S1; S1 estimate includes the dev search and full-span rows.

### Plan critic 2 (v4, 2026-10-07 night; all eight applied, none rejected)

- F1 (major, A2-3 cannot discriminate) applied, option (b): A2-2 is the only scored regime test; A2-3 is withdrawn
  from the tally with its re-derivation in section 8 and kept as a diagnostic. Option (a) is reported as a
  diagnostic (`loops_lt_K.dm_over_D_selector_minus_Llast_Km1`, dev items, all sites verified) because I cannot
  derive a regime prediction for it: on D2 the K16 answer edit carries from Llast@1 (0.81 D kept), so Llast@K-1
  is likely inside the band and the difference near 0 there too [belief]. Pre-run deviation DV12.
- F2 (major, A2-1 passes by readout proximity) applied: scored as hit - latest-site hit (paired, cluster CI) over
  the full set and within loops 1..K-1 (Llast@K-1 baseline, scored when >= 10 eligible). Band chance is a
  diagnostic. Predictions in section 8. Code: `carry_summary` `hit_minus_latest`, `predictions` rows "A2-1 hit"
  and "A2-1 ltK". Test: selftest `v4.carry_v4` (toy where the selector equals the latest site on one item and
  beats it on the other: full-set difference 0.0; loops < K difference 1.0).
- F3 (major, undo slot selected by presence alone) applied: `selected_by` per slot, undo cited and scored only at
  (a) and b2 slots, e_carried and e_U / e_carried with cluster CIs per slot, IFM W subset 20 -> 30 on the two undo
  models. Test: selftest `v4.window_clause`, `v4.boot_ratio`; summarize on the v2.1 debug rows labels every
  selected Ouro-2.6B slot "a+b1_only" (n = 3 < 10 for b2), so the undo row is "not scored".
- F4 (major, hash taken at summarize time) applied: `--plan` and `--plan_sha` at launch, hash records in each
  results file, frozen-part hash check before phase M, S1 commands updated (sections 9, 10). The coordinator
  writes the sha256 into the LEDGER (the builder does not touch the LEDGER). Test: selftest `v4.plan_hash`,
  `v4.plan_launch` (missing plan, wrong sha, phase all, phase M without a W record, and a plan edit outside
  section 13 are all refused; a section 13 edit passes).
- F5 (minor, failed gate called "no answer state") applied: class renamed "gate fails (absent or overwritten
  within loop j)"; the digit probe runs on the Ouro states too and decides `cite_as`. Predictions renamed.
- F6 (minor, V6 strata and PC missing) applied: ST1 and ST2 strata of hit, regret and test dm / D, and the
  model's PC-final value in every carry_gain record; ST1 added to the D1, D2 and D4 item strata. Test: selftest
  `v4.cg_st`, `v4.carry_v4` (strata keys present).
- F7 (minor, IFM probe without a working positive control) applied: parameter-free leave-one-cluster-out
  centroid probe over all kept items (chance 0.5), positive control at the readout-feeding state Llast@K
  (known present: the clean model reads tA / tB from it), labels only when the PC accuracy is >= 0.80. The
  critic's "PC-site rows" are not used as the probe PC: a PC-site row replaces every position, so its exactness
  says nothing about the digit at r; the readout-feeding state is the case where the digit is known to be
  linearly present. Test: selftest `v4.probe` (planted direction 1.00, noise 0.47) and `v4.probe_table`.
- F8 (minor, D4 B-side shares across splits) applied: connected components of shared A/B content keys, split by
  component, K16 exclusion by component (section 1 I7). Test: selftest `v4.d4_components`; CPU build on the Ouro
  tokenizer: 0 shared content keys across splits.

### Plan critic 3 (v5, 2026-10-07 night; all eight applied, F5 and F8 in a stated variant)

- F1 (major, plan and code diverged; v4 rules wrong on the debug evidence) applied. The v5 rules are now in the
  plan before any hash: clause (a) by influence with necessity reported (section 4 V5 v5), b2 with its CI and
  MIN_CL items and clusters, T6 by the survival CI (section 6), T8 in clusters, and the planted-digit power control
  at every probed site with "absent" only where the planted code is detected (section 5). Sections 4-10 are
  re-frozen (DV13). The launch record holds the code sha256; a full run refuses a code hash that differs from the
  plan's "Launch code sha256" line, and phase M refuses a code change since phase W (section 9). The v5 debug and
  hook checks ran on the code that is launched (section 13). Tests: selftest `v5.slot_stats`, `v5.t6`,
  `v5.probe_planted`, `v4.window_clause` (v5 rule), `v4.plan_launch` (6 refusals incl. a code edit),
  `v5.code_hash`.
- F2 (major, RND fails at Llast sites, no citation rule) applied. Rule T3b (section 5): paired O1 - RND
  difference with a cluster CI, dz(A) and dz(B) of both rows, O1 cited alone only where the RND cell passes T3,
  else the paired difference with the label "nonspecific damage"; applied to classes, TOST nulls, threshold rows,
  VBD cells and the A2-2 working-site fraction. RND now runs at all 24 cells on IFM too (cut Q5 restores the 7).
  The ouro26 debug rows were rescored under the rule (RF15). Tests: selftest `v5.rnd_pair` (RND 0.2 -> cite
  O1 - RND = 0.105; RND 0 -> cite O1; no RND -> not citable; class final e from the difference).
- F3 (major, no host-migration path) applied (section 7 migration rule): one model per process, base models
  first, the stop command, one host role per results file enforced in code, item host tags, hook checks rerun per
  role and code hash, a stopped model restarts from scratch on shared-host (W and M on one host), cross-host pairs not
  scored, and a shared-host budget request that needs the coordinator. Test: selftest `v5.host_mix`; the GB10 Ouro timing
  comes from the v5 debug on shared-host (section 13).
- F4 (major, V6 not gated on CGCHECK; bf16 G) applied (section 4 V6 gate): V6 rows are "not scored" until CGCHECK
  passes and the family FD row passes; `--fd_check` compares a G (fp32) with the exact fp32 dm at a = 1e-3, 1e-2,
  0.1 on 4 VarBench items per model at G_red sites and writes the bf16 vs fp32 G difference. Tests: selftest
  `v5.fd`; the GPU FD rows are in section 13.
- F5 (major, no derived-use target) applied in a variant (I8, V8). The literal rule (prefix that never restates
  x) yields 0 usable rows (RF14), because GSM8K gold steps restate their operands. The variant keeps the shared
  chain text, renders every x with each side's own value, and reads the first derived result: it is an operand
  edit, the prefix has no x-derived result, and the read is the first derived-result token. VBD runs V1 and V6 on
  Ouro and V6 on IFM, joins A2-2 as the real-math computed dataset, and A2-2 is reported per dataset with a single
  "A2-2 regime" row that needs VarBench and VBD; no regime claim from D2 alone. Tests: selftest `v5.vbd_text`,
  `v5.safe_eval`; CPU build 341 problems.
- F6 (minor, undo scored on its selection items) applied: T6 and s(c) are scored on phase-M items outside the W
  subset; phase M runs UKO_c and KO_c at the selected slots to have the denominator there; W values are reported as
  selection data. Test: summarize on the v4 debug (M-only set is empty there, so `t6` is null; W selection data
  present).
- F7 (minor, ST1 counts span tokens) applied: ST1e (differing source tokens) added, used for P-ST and the
  injection-twin TOST (`tost_by_ST1e`), ST1 kept as span length, ST1e also in the A2 and carry-gain strata. Test:
  selftest `v5.st1e`.
- F8 (minor, S1 needs a shard step that does not exist) applied by the critic's second option: S1 is stated as 4
  per-model jobs with a 12-18 h wall time (section 10); the shard step is not built (DV15). Reason: the dev-tuned
  pass needs all dev items in one process, a correct merge needs its own test harness, and the 4-job form already
  runs on the cluster.

## 12. Deviations from the brief

- DV1 Read-out = copy-read target (A3).
- DV2 Thinking checkpoints get the base raw-completion prompt (spans verified for the base assembly; K16 did the
  same).
- DV3 O2 is a context-robustness check (F8), not the brief's "aligned paraphrase from a second render" (RF5).
- DV4 IFM rows reduced to fit the 2 GPU-h shared-host cap.
- DV5 Read window by 5 suffix knockouts x 2 variants on a subset, not per-call knockouts.
- DV6 V6 on G_red (19 sites) for ouro26 x2 and IFM (critic F2's reduced-grid option); G_full on the 1.4B models.
- DV7 Annotation-restatement, no-restatement, contradictory-prefix and ambiguous-source rows excluded.
- DV8 PLAN's hard rule says fast-host GPU1 belongs to someone else; the K17 brief and the coordinator assign GPU1 to
  this track, and that later assignment is followed. Remote writes go to `out/k17_varbench/`.
- DV9 While both hosts were unreachable, the hook checks ran on the laptop GPU (RTX 4070, Ouro-1.4B, T 736, all
  pass). The debug runs then ran on the hosts. `--drop_block` (zero-shot copy-read) exists for laptop debugging
  only and was not used for any reported number.
- DV11 v3 (code review, before any full-run row): D1 falls below the 80-item quota (at most 69 under the subject
  split, section 1 I7); answer-position cells use first-token e as primary (section 5); A7 site sets (section 2).
- DV12 v4 (plan critic 2, before any full-run row): A2-1 rescored against the readout-proximity site, A2-3
  withdrawn from the tally, undo scored only at (a)+b2 slots, the class "no answer state at site" renamed and
  gated by a parameter-free probe, IFM W subset 30 on the undo models, D4 clusters by component, full runs
  refuse to start without the plan hash.
- DV13 v5 (plan critic 3, before any full-run row): sections 4-10 re-frozen with the v5 undo/window rule
  (influence clause a, b2 with CI, T6 survival CI, MIN_CL), the RND citation rule, the planted probe control, ST1e,
  T8 in clusters, the V6 gate and VBD. The re-freeze came after the v4 debug (n = 3, harness only) had been read;
  RF15 states what was read. The P-W "last loop reads the span" row moved from necessity to influence after that
  read, so it is marked (v5) and should be read as re-registered, not as pre-registered in v4.
- DV14 VBD uses the own-value prefix variant, not the literal "prefix does not restate x" rule (RF14).
- DV15 No `--shard` step; S1 = 4 per-model jobs (critic 3 F8).
- DV17 v5.1 (2026-10-08, before any full-run row): the FD check's fp32 pass turns TF32 off (matmul, cudnn,
  precision "highest") and restores it after. k10_ifm.load_ifm sets TF32 on, so in the v5 debug the IFM "fp32"
  body ran in TF32 and the FD rule failed on all three IFM models from rounding noise (Ouro, without TF32, passed).
  The rule itself is unchanged. The fd json records `tf32_fp32_pass` and `final_hidden_dtype_fp32`. v5.2: the
  bf16 selector condition of the V6 gate (in the code since v5, not in the plan) is written into section 4 V6 and
  changed from exact top-site equality to the T7 0.05 D band; the fd json stores per-item bf16 and fp32 G
  (`G_pairs`). Selftest `v5.bf16_agree`.
- DV16 D4 A2 is not run on the IFM models (pre-cut Q1) to make room on shared-host for RND at 24 cells, VBD and the FD
  check.
- DV18 v6 (code review 3, 2026-10-08, before any full-run row; frozen sections 4-10 changed, so a new plan hash and a
  new launch code sha256 are needed): T3b v6 (cite O1, RND gates lower bounds); b2 per D_KO with the BKO row in
  phase M; T7 dose band scaled by a and the dose clause paired; the V6 gate on the CGCHECK full-run file hash and on a
  30-item fresh bf16 check with the loops < K argmax; the P-W relay row tallied on clipped dep; the random relay
  keys before the span; number words in the prefix filters with items rebuilt in `data_v6` (S200 changes: 140 of
  200 pairs kept); the G4 path-integrated gain row; G8 data, config and output hashes; host roles keyed on the
  GPU name; S1 estimate recomputed. The v5.2 debug numbers in section 13 were read before this change (they
  motivated C-F1, C-F2 and A-F4), as stated per finding in section 14.
- DV10 v2.1 (after the debug run, before any full-run row): window boundaries after Lmid@2 and Lmid@3 and undo
  candidates Lmid@2 and Lmid@3 added (V4, V5); Ouro answer classes gated by e_own at the edited loop's own exit for
  every site; IFM classes keep the R2 gate but are not cited without the digit probe (section 5).

## 13. Run log (appended after phase W and the debug runs)

- 2026-10-07 evening, build: 1198 core rows -> 1164 valid rows / 513 problems (no restatement 17, restated first
  inside an annotation 6, contradictory prefix 1: gsm8k-641-raise_percentage, ambiguous source 10, tokenizer
  assert failures 0). S200 strata: Ouro span 1 / >= 2 tokens 87 / 113, IFM 196 / 4; digits 1/2/>=3 87/86/27; cont
  96 (Ouro), 4 (IFM); dev 106. Ouro's standalone block ends in one "\n\n" token that splits in context, so block
  ids are taken in context (574 tokens, IFM 471). K16 exclusions: 79 D2 problems, 65 D4 templates, 154 D4 content
  clusters.
- Hook checks a-h pass on all 7 models (laptop GPU for Ouro-1.4B; fast-host GPU1 for Ouro x4; shared-host for IFM x3):
  zero edits at every site, zero grad leaves, repeat, records, wrapper, causality at r all bit-identical; Ouro
  module probabilities on knocked keys exactly 0 (direct and relay); IFM module output equals the recompute under
  the mask (max abs 0.0). PC-final exact 3/3, SZ exact 6/6, PC-site exact 36/36 per Ouro model.
- Debug (3 VarBench items per model, not interpretable, harness only), Ouro x4: relay dep after Llast@1 / Lmid@2 /
  Llast@2 / Llast@3: ouro 0.71 / 0.58 / 0.18 / 0.01; ouro_think 0.71 / 0.71 / 0.45 / 0.21; ouro26 0.68 / 0.65 /
  0.47 / 0.22; ouro26_think 0.58 / 0.60 / 0.63 / 0.44. Full relay knockout: 0.99, 0.97, 0.70, 0.68; random
  equal-size knockout: -0.01 to 0.01. Slots selected by the rule (debug only, the full run re-selects on 40
  items): ouro [Lmid@2]; ouro_think and ouro26 [Lmid@2, Llast@2, Lmid@3, Llast@3]; ouro26_think all five.
- Debug, IFM x3 (same caveat): source e_LO at L0@1 / Llast@1 / Llast@2 / Llast@3 / L0@4 / Lmid@4: ifm_p1_ouro 1.00
  / 1.00 / 1.00 / 0.93 / 0.93 / 0.20; ifm_p1_ouro_inj 0.01 / 0.01 / 0.02 / 0.02 / 0.99 / 0.73; ifm_d112 1.00 / 0.96 /
  0.93 / 0.18 / 0.18 / 0.00. Answer Lmid@4: 0.79, 0.28, 0.84. Injection twin PC-site 0.01-0.06 at loops 1-3 and
  1.00 from L0@4. Relay dep after Llast@3: 0.84 (no-inj), 0.84 (inj), 0.19 (d112). Slots selected (debug): no-inj
  [Lmid@2, Llast@2, Lmid@3, Llast@3]; d112 [Llast@2, Lmid@3]; inj none (no undo, A5).
- Debug GPU minutes: fast-host GPU1 6.2 (three Ouro debug processes plus a W-then-M split test that confirmed the
  two-phase resume), shared-host 14.4 (two IFM debug processes; the first used the v2 candidate set and is archived in
  `debug/run1/`, as is the first Ouro-1.4B debug). The laptop GPU ran the hook checks (2.7 min) and a debug run
  that was stopped unfinished once the hosts came back (no results used).
- 2026-10-07 night, v3 (review fixes, section 14). Items rebuilt into `out/k17_varbench/data_v3/` (laptop CPU):
  same 1164 valid rows / 513 problems, pair = lowest var_name (S200 equals GEN's pair on 198), ST4 by rank (Ouro
  67 / 66 / 67), cont 97 (Ouro). fast-host and shared-host unreachable (ssh timeout, no DNS), so the v3 debug ran on the
  laptop GPU (RTX 4070 8 GB, Ouro-1.4B, offline hub cache). With the 4-shot block the job spilled 2.3 GB into
  shared system memory and stalled on the first W item; it was stopped after about 9 minutes (no rows used). The
  rerun used `--drop_block` (zero-shot, laptop debug only, DV9): 3 VarBench + 2 per A2 dataset, 825 forwards,
  2.2 GPU-min [ran]. Hook checks a-h all pass with the float32 readout (dtype float32, fp32-vs-bf16 head max 0.062,
  TF32 off after the call); PC-final exact 3/3 (e 1.00, e_LO ceiling 0.80), SZ exact 6/6, causality 66/66 rows;
  G_full = 74 sites with latest site = uniform 4.12; D1 split: 295 candidates, 161 dropped, 0 shared subjects,
  dyadic CI; no host, user or home string in checks.json, results.json, summary.md. Not run on GPU in v3: the
  IFM float32 head (`out32`, needs shared-host) and the 2.6B sets; both must pass the hook checks in the next debug run
  on the hosts before phase W.
- 2026-10-07 21:35-21:50, v4 debug on the hosts [ran] (code md5 ca605783 on the hosts for the run; the final v4
  file, md5 becd89a8, differs only in a summary.md line and is pushed to both hosts). Out
  `out/k17_varbench/debug/v4/`, pulled to `artifacts_spark/k17_varbench/debug_v4/` (merged summary in `merged/`).
  3 VarBench + 2 per A2 dataset per model, `--phase all --debug`. Hook checks a-h all_ok on all 7 models with the
  float32 head (fp32 vs bf16 head max 0.055-0.073; IFM `out32` loaded, TF32 restored to the loader's value);
  edit placement and knockout placement (direct, relay) ok on all 7; PC-final exact 3/3, SZ 6/6, PC-site 36/36 on
  every non-injection model (injection twin: 0/3 at loops 1-3, 3/3 from L0@4, as designed); causality 66/66 and
  99/99 rows on Ouro (IFM items have no continuation); grad run bit-identical; selector backward/forward
  2.28-2.60. v4 D4 build on both tokenizers: 814 components, 98 K16 components excluded, 0 shared content keys.
  Debug-only numbers (n = 3, not interpretable): source e_LO L0@1 / Llast@1 / Llast@3 / Lmid@4: ouro 1.00 / 0.86 /
  0.58 / 0.51, ouro26 1.00 / 0.92 / 0.80 / 0.71, ifm_p1_ouro 0.99 / 1.00 / 0.92 / 0.21, inj 0.01 / 0.01 / 0.02 /
  0.80, d112 1.00 / 0.99 / 0.16 / 0.00. RND at source Llast@1: ouro26 0.25 [0.08, 0.39], ouro26_think 0.17 (2 of
  3 items at 0.27-0.46 D, no flips; seen in the v2.1 debug too), all other RND cells |e| <= 0.02. No undo slot is
  informative in debug (b2 needs 10 items); every selected slot is "a+b1_only". e_U / e_carried at the Ouro
  slots 0.03-0.45. Selector: dev hit = latest-site hit on every model (1-2 eligible items). The probe could not
  run (3 items < 4); it is covered by the selftest only. Harness row "relay >= direct" missed on the injection
  twin and d112 by 0.01-0.12 at some boundaries. GPU minutes: fast-host GPU1 4.8, shared-host 6.7 (totals with earlier
  debug: 11.0 and 21.1).
- 2026-10-07 22:40-23:50, v5 build and debug [ran]. CPU: `--selftest` ok (v4 and v5 blocks, predictions coverage
  572 rows, all "not scored" on empty input); `--build_vbd` 341 problems; summarize on the v4 debug copies (RF15).
  Laptop GPU (zero-shot `--drop_block`, harness only): debug of ouro with VBD and D2, 2.3 GPU-min, and the FD check
  (pass; bf16 vs fp32 G median rel 0.019, top site 4/4). fast-host GPU1, code md5 e4a139cc, `debug/v5/`, n = 3
  VarBench + 2 VBD + 2 per A2 dataset per model, co-tenant at 72% utilisation on GPU1: hook checks all_ok, edit
  and knockout placement ok, PC-final 3/3, SZ 6/6, PC-site 36/36, causality 66/66 and 99/99 rows on all four Ouro
  models; VBD tokenization 0 mismatches, VBD clean pass 2/2, VBD PC-final 2/2 and SZ 4/4 on every model. FD check
  (fp32, a = 1e-2, 54-60 scored site-items per model): median rel error 0.0036-0.0088, sign agreement 1.00, pass on
  all four; bf16 vs fp32 G median rel 0.04-0.11, p90 0.39-2.77, top site equal on 15 of 16 items. v5 slot rule
  (min_n 1, debug only): influence removal at Llast@3 0.25 (ouro) to 0.68 (ouro26_think) while necessity is
  0.03-0.51. RND fails T3 at the VBD source Llast@1 cell on all four models. Per-item seconds with the co-tenant:
  phase M 9-23, phase W 6-10, VBD 11-17, A2 0.8-4.5. Debug numbers are harness evidence only.
- 2026-10-08 00:00-00:25, v5 IFM debug and v5.1 [ran]. Code provenance of the v5 runs: fast-host Ouro debug md5
  e4a139cc; IFM debug ifm_p1_ouro and inj sha256 67a52230, d112 and the IFM FD check sha256 b4b75326 (the file
  changed on the hosts at 23:27 during the IFM run). IFM v5 debug (shared-host, n = 3, harness only): hook checks all_ok on
  all three. IFM FD check (v5): fail on all three (a = 1e-2 median rel err 1.40 / 1.56 / 0.13, sign 0.77 / 0.56 /
  0.90), Ouro pass. Cause found: TF32 on in the IFM fp32 pass (DV17). v5.1 code (sha256 4377ede2, md5 5b762fe1,
  pushed to both hosts, md5-checked): FD on ifm_p1_ouro passes (a = 1e-2: median rel err 0.0058, p90 0.028, sign
  1.00, n 47; a = 1e-3: 0.0018; final hidden fp32, TF32 off). bf16 vs fp32 G on ifm_p1_ouro: median rel 0.42, p90
  12.8, |diff| / D median 0.0005, p90 0.0025, top site 4/4, FN class agree 1.00.
  GB10 Ouro timing (shared-host, ouro, v5 code b4b75326, three co-tenants: k17_parcae full, two k17_public debugs): hook
  checks all_ok; W item 58 fwd in 67.5-67.9 s, M item (dev, G_full) 163 fwd in 188.6 s, so 1.16 s per forward,
  16x the uncontended fast-host rate. Stopped by SIGTERM (pid) after the first M item. Migration request (section 7
  (5): ouro, n_vb 60, G_red, W 40, D2 and VBD 40 each): about 40 x 68 s + 60 x 130 s + 80 x 70 s = 4.5 GPU-h at
  this load, not 120 min. Moving ouro to shared-host does not fit any shared-host cap; the fallback is to pause, not migrate.
  IFM rate in the v5 debug with the same co-tenants: 0.85-1.0 s per forward (v4: 0.18-0.40), M item 50-84 s, W
  item 26-53 s; the shared-host IFM projection (75-90 min) is therefore about 3-4x low under this load.
  v5.1 (sha256 4377ede2) [ran, VPN down 00:26-01:09 and 01:35-02:10, remote jobs continued]: fast-host GPU1
  `debug/v5_1/`, one process per model, then FD per model; shared-host `debug/v5_1/` IFM hook checks and FD on all three.
  FD at a = 1e-2 (median rel err / sign): ouro 0.0045 / 1.00, ouro_think 0.0055 / 1.00, ouro26 0.0036 / 1.00,
  ouro26_think 0.0088 / 1.00, ifm_p1_ouro 0.0058 / 1.00, ifm_p1_ouro_inj 0.0045 / 1.00, ifm_d112 0.0028 / 1.00: pass
  on all seven. bf16 top site equal to the fp32 top site on 4/4 items everywhere except ouro (3/4), which closed the
  code's unstated exact-equality condition; v5.2 replaces it (DV17, section 4 V6).
- 2026-10-08 01:30-02:10, v5.2 = launch code (sha256 0e25471e, md5 8d6c6d8e, pushed to both hosts, md5-checked)
  [ran]. CPU: `--selftest` ok (new key `v5.bf16_agree`; predictions coverage 572 rows). fast-host GPU1 `debug/v5_2/`,
  one process per model, n = 3 VarBench + 2 VBD + 2 per A2 dataset, phase all, then FD per model; every launch
  record holds code sha256 0e25471e. Hook checks all_ok, PC-final 3/3, SZ 6/6, VBD PC-final 2/2 and SZ 4/4 on all
  four Ouro models. FD pass on all four (same numbers as v5.1); bf16 top site inside the fp32 0.05 D band 4/4 on all
  four (exact top-site equality ouro 3/4); FN-class agreement 0.97-1.00; bf16 selector condition met on all four.
  shared-host `debug/v5_2/`: hook checks (`--cpu_check --device cuda`) all_ok on the three IFM models, tokenization 0
  mismatches. Not rerun on v5.2 to save shared-host minutes: the IFM phase-all debug (last run on 67a52230 / b4b75326)
  and the IFM FD (v5.1; the FD part of the code is unchanged in v5.2, and exact top-site equality 4/4 implies the
  band condition). The full run reruns the hook checks at launch by itself (new code hash per role).
  Summarize on the merged v5.2 debug (Ouro v5.2 + IFM v5 rows + IFM v5.1 FD, `artifacts_spark/k17_varbench/
  debug_v52/merged/`): harness rows 28 held, 3 missed, every tallied row "not scored" (debug, CGCHECK not
  passed). The 3 misses are "relay dep >= direct dep at every boundary" on the three IFM models (v5 IFM debug rows,
  relay / direct): full knockout ifm_p1_ouro 1.20 / 1.22, inj 0.89 / 1.00, d112 0.93 / 1.21. The v4 debug showed
  0.01-0.12 gaps on inj and d112; the d112 full-knockout gap (0.28) is larger. Both values exceed 1 on some
  models. [v6 correction, review A-F4/C-F7: dep > 1 means the knockout moved the margin past the tA / tB tie (tB
  now beats tA), not past clean B, which is dep = D / (-m_A), about 2.2 on these items; the row is not forced, see
  section 8 v6.] n = 3, not interpreted; to be read on the phase-W subset. Debug-only VBD numbers (n = 2, harness
  evidence only): RND fails T3 at source Llast@1 on all four Ouro models (v5 cited value O1 - RND 0.68-0.80). [v6
  correction, review B-F2: the answer Llast@1 values -0.64 (ouro26) and -0.62 (ouro26_think) are the O1 - RND paired
  differences, not O1 e. O1 e at that cell is +0.08 and +0.15 (per item +0.06 to +0.17), so the donor edit moves the
  read slightly toward B; RND e is +0.72 and +0.78 and fails T3. Under T3b v6 the cited value is the O1 e.]
  GPU minutes (process wall time): fast-host GPU1 v5.1 14.1 + v5.2 12.1; shared-host v5 IFM debug 23.1 + v5 FD 20.7 + GB10
  Ouro debug about 8 + v5.1 18.0 + v5.2 3.5 = about 73. shared-host total with the v2-v4 debug (21.1): about 94 of the
  120 GPU-min cap, before any IFM full-run row.
  HOST RULE breach (builder error): the v5.1 and v5.2 Ouro debug and FD jobs ran on fast-host GPU1 between about
  00:05 and 01:45 CDT, while PLAN.md Hard rules (the PI 04:20 UTC = 23:20 CDT) put fast-host off-limits until he
  releases it. The builder had not re-read PLAN.md after the restart. Another user's process ran on GPU1 at the
  time (6.5 GB). All jobs ended by themselves; at 02:15 no process of this track runs on fast-host. Every later GPU job
  of this track runs on shared-host (or second-shared-host) until the LEDGER line that releases fast-host. The v5.2 Ouro debug rows are
  valid harness evidence but came from the forbidden host.
- 2026-10-08 02:20-02:55, v6 (code review 3, DV18, section 14 v6) [ran]. CPU: `--selftest` ok (new `v6` keys; v5
  `rnd_pair` and `slot_stats` updated to T3b v6 and BKO); `--build_items` into `data_v6` (913 rows, 409 problems; VBD
  351 problems); `--summarize` on a copy of the v5.2 merged debug (harness 24 held, 0 missed after the relay row
  left the harness set; RND fails T3 at 42 debug cells, RND e positive at 40). Laptop GPU (RTX 4070, `--drop_block`,
  Ouro-1.4B, `laptop/debug_v6/`, harness only): W 1 + M 2 items with BKO rows at the selected slots, G_PI on one dev
  item each of VarBench, VBD and D1 (68 passes, 97-110 forward-equivalents), the bf16 check (3 fresh items, G_full
  74 sites, FD items excluded), summarize with the CGCHECK full file (gate open) and the debug file (gate closed: 2
  insufficient cells, sha256 mismatch); hook checks all_ok keyed "laptop|NVIDIA GeForce RTX 4070 Laptop GPU"; 4.0
  GPU-min. shared-host (GB10, three co-tenants, code sha256 1dba6a6a pushed and md5-checked with `data_v6`,
  `debug/v6/`, pulled to `artifacts_spark/k17_varbench/debug_v6/shared-host/`): ifm_p1_ouro phase all, n_vb 2, VBD, pi_n 1:
  hook checks all_ok keyed "ifm_host|NVIDIA GB10", PC-final 2/2, SZ 4/4, VBD PC-final 2/2 and SZ 2/2, BKO rows at
  the selected slots on the M-only item, random relay keys before the span, functional knockout check ok (span 1.08,
  random -0.01), G_PI on one VarBench and one VBD dev item (51-52 s each); bf16 check on 2 fresh items (fp32 pass in
  fp32, TF32 off): band 2/2, loops < K 2/2, FN class 1.00. 5.6 GPU-min; shared-host total for this track about 100 of the
  120 GPU-min cap. No fast-host use (HOST RULE).
- 2026-10-08 07:55 UTC (02:55 CDT), resume after the VPN came back [ran: ssh, nvidia-smi, pgrep]: no k17_varbench
  process on shared-host, fast-host or second-shared-host, and no `full/` dir on either host, so nothing ran during the outage. fast-host
  GPU0 and GPU1 are empty, but the HOST RULE is not lifted (no LEDGER release line). shared-host GB10 at 93% with three
  co-tenants (production verify_gpu, two k17_control jobs). Full run not launched: (1) fast-host not released;
  (2) the plan sha256 is not in the LEDGER (section 9, step 1); (3) about 20 of the 120 shared-host GPU-min are left,
  and the IFM pilot needs several hours at this load; (4) R-PI and Q-PI need the coordinator. No GPU minutes used.
- 2026-10-08 full run launch (coordinator decisions, LEDGER 03:10): plan sha256 34ef3dc3 at phase W, code sha256
  1dba6a6a, data_v6 sha256 equal on laptop, shared-host and the Ouro host. Ouro-1.4B and Ouro-2.6B base on a GB10 Spark
  shared with another user's vLLM job (role `ouro_gb10`, `full/`); ifm_p1_ouro on shared-host (role `ifm_host`, `full/`,
  `--pi_n 5` (Q-PI), `--datasets VBD,D2,D1` (DV16)). Thinking models wait for fast-host (R-PI: no G_PI on them).
  FD check pass on ouro (a = 1e-2 median rel err 0.0054, sign 1.00), ouro26 and ifm_p1_ouro (0.0058, 1.00). bf16
  check (30 fresh items): ouro band 30/30, loops < K 30/30, FN class 0.985, pass; ifm_p1_ouro 30/30, 30/30, 0.980,
  pass.
- Phase W slots for the full run (appended after phase W, before phase M):
  - ouro (40 W items, 1 excluded from dep for small margin): selected [Llast@2, Lmid@3, Llast@3, Lmid@4], all
    "a+b2" and informative. removal / carried_over_DKO: Lmid@2 0.96 / 0.22 (not selected), Llast@2 0.67 / 0.72,
    Lmid@3 0.54 / 0.80, Llast@3 0.30 / 0.90, Lmid@4 0.23 / 0.91. Relay dep full 0.99, after Llast@3 0.29, random
    relay -0.001.
  - ifm_p1_ouro (30 W items, 0 excluded): selected [Lmid@2, Llast@2, Lmid@3, Llast@3]; informative ("a+b2") only
    Llast@3; Lmid@2, Llast@2, Lmid@3 are "a+b1_only" (b2 items with D_KO >= 2: 0, 1, 7, below MIN_CL). removal /
    carried_over_DKO: Lmid@2 0.99 / n/a, Llast@2 0.99 / 0.83 (n 1), Lmid@3 0.91 / 0.95 (n 7), Llast@3 0.87 / 0.97
    (n 11), Lmid@4 0.05 / 1.00 (not selected). Relay dep full 0.91, after Llast@3 0.81, after Lmid@4 0.03, random
    relay 0.0006.
  - ouro26 (40 W items, 1 excluded from dep): selected [Lmid@2, Llast@2, Lmid@3, Llast@3, Lmid@4], all "a+b2" and
    informative. removal / carried_over_DKO: Lmid@2 0.95 / 0.31 (CI lower 0.19, n 16), Llast@2 0.79 / 0.59, Lmid@3
    0.65 / 0.73, Llast@3 0.58 / 0.84, Lmid@4 0.39 / 0.88. Relay dep full 0.98, after Llast@3 0.50, random relay
    -0.0002.
  - Rates on the shared GB10 (two of our processes plus the co-tenant): ouro phase M about 2.2 min per VarBench item
    (0.93 s per forward), ouro26 phase W 1.8 s per forward. Projection with the pilot sizes: ouro about 13 h,
    ouro26 about 19 h. Pre-registered cuts (section 7) applied to ouro26 phase M only: R0 (VBD n 80 -> 40) and R4
    (VarBench n 200 -> 120, `--n_vb 120`). R1 (D4 A2 only) and R3 (RND at L0/Lmid answer cells) have no flag in
    the launch code and are not applied; R2 is moot (phase W already ran 40 items). ouro is not cut.
  - Math specificity readout (coordinator, RA-1 item 3; written before any row): new file
    `loop-flow/src/k17_vbspec.py` (imports k17_varbench, never modifies it), Ouro-1.4B base, on the
    data_v6 VBD items whose gold prefix holds an independent calculator step (no input derived from x, the vbd_text
    taint rule) before the first dependent step: 138 of 351 selected problems (213 have none; 1 restates x before the
    independent read). Read p_ind = the token before the first token of the independent result inside '<<expr=' (gold
    token g) and r = the VBD read. Clean filter: top-1 g at p_ind on A and B, tA / tB at r. Rows: span O1 and RND at
    Llast@1 (source = prompt span + chain restatements of x); last-token (r) O1 and RND at Llast@2 and Llast@3 (an
    edit at r cannot reach p_ind by causality; checked bit for bit, so its specificity is 1 by construction).
    Specificity = top-1 at p_ind stays g; target = top-1 at r is tB; both with the RND rows, n_items, n_clusters.
    No search, no strength matching. Prediction (coordinator's): the span edit moves every dependent step and no
    independent step (specificity about 1, target high); the last-token edit moves only what follows it. Output
    `out/k17_varbench/spec/` on shared-host (GB10, role `ouro_spec`), pulled to `artifacts_spark/k17_varbench/spec/`.
  - Math specificity readout result (ouro, 138 items / 138 clusters, all pass the clean filter, 28 GPU-min on the
    GB10): span O1 at Llast@1 specificity 1.00, target 0.39 [0.31, 0.47] (e 0.36); last-token O1 at Llast@2
    specificity 1.00 (causal, bit-identical 138/138), target 0.96 [0.92, 0.99] (e 0.91); at Llast@3 1.00 / 1.00
    (e 0.97). RND rows: specificity 1.00, target 0.00 (e 0.02-0.08).
  - 2026-10-08 12:52 UTC, host move (coordinator: the PI released both fast-host cards; ouro26 to card 1). ouro26 was
    stopped by SIGTERM on the GB10 after 46 of 120 phase-M items. Migration rule (section 7 (3)) applied: the GB10
    file `full/vb_ouro26.json` stays as a partial result (reported, not merged), and ouro26 restarts from scratch on
    fast-host card 1 in `full_rtx/` (role `ouro_host`): FD check, bf16 check, phase W, slots, phase M, so its slots
    are selected and scored on one host. On fast-host the pilot sizes fit, so the R0 and R4 cuts are not applied
    there (n_vb 200, vbd_n 80). The coordinator asked to resume from the saved files; the frozen migration rule was
    kept instead because resuming would score GB10-selected slots on fast-host items and mix host roles in one file.
  - ouro26 on fast-host card 1 (`full_rtx/`, phase W plan sha256 be0f4b77): FD pass (a = 1e-2 median rel err 0.0043,
    sign 1.00); bf16 check band 30/30, loops < K 30/30, FN class 0.977, pass. Phase W slots (40 items, 1 excluded):
    selected [Lmid@2, Llast@2, Lmid@3, Llast@3, Lmid@4], all "a+b2" and informative; removal / carried_over_DKO
    Lmid@2 0.95 / 0.31 (CI lower 0.20, n 16), Llast@2 0.79 / 0.60, Lmid@3 0.65 / 0.73, Llast@3 0.58 / 0.84, Lmid@4
    0.39 / 0.88; relay dep full 0.99, after Llast@3 0.50, random relay 0.002. Same slots as the GB10 phase W.
  - Base runs landed 2026-10-08 15:03 UTC: ouro (GB10, phase M plan sha256 b3902243) and ouro26 (fast-host card 1,
    phase M plan sha256 0854f04d). Thinking models (base-vs-Thinking ablation, R-PI: `--pi_n 0`) on fast-host,
    `full_rtx/`, role `ouro_host`, phase W plan sha256 0854f04d: ouro_think on card 1, ouro26_think on card 0.
  - ouro_think: FD pass (a = 1e-2 median rel err 0.0060, sign 1.00); bf16 check band 29/30, loops < K 29/30, FN class
    0.967, pass. Phase W slots (40 items, 1 excluded): selected [Llast@2, Lmid@3, Llast@3, Lmid@4], all "a+b2" and
    informative; removal / carried_over_DKO Lmid@2 0.96 / 0.22 (not selected), Llast@2 0.75 / 0.69, Lmid@3 0.67 /
    0.73, Llast@3 0.41 / 0.89, Lmid@4 0.35 / 0.90; relay dep full 1.01, after Llast@3 0.40, random relay 0.0005.
  - ouro_think phase M launched with plan sha256 05153c74. ouro26_think: the first phase-W launch was refused (plan
    sha 0854f04d no longer matched after the ouro_think slots were appended; no row written) and was relaunched
    with 05153c74. FD pass (a = 1e-2 median rel err 0.0052, sign 1.00); bf16 check band 30/30, loops < K 30/30, FN
    class 0.975, pass. Phase W slots (40 items, 1 excluded): selected [Lmid@2, Llast@2, Lmid@3, Llast@3, Lmid@4];
    informative ("a+b2") Llast@2, Lmid@3, Llast@3, Lmid@4; Lmid@2 "a+b1_only" (carried_over_DKO 0.20, n 13).
    removal / carried_over_DKO Llast@2 0.84 / 0.61, Lmid@3 0.75 / 0.71, Llast@3 0.64 / 0.84, Lmid@4 0.53 / 0.87;
    relay dep full 0.93, after Llast@3 0.55, random relay 0.0005.
  - ouro26_think phase M launched with plan sha256 16ef642d. Both Thinking models landed 2026-10-08 17:34 UTC. All
    five full-run models pulled to `artifacts_spark/k17_varbench/full/` (vb file sha256 checked against the remote);
    the GB10 ouro26 partial file to `artifacts_spark/k17_varbench/partial_gb10_ouro26/`. No k17_varbench process
    remains on any host. Summarize run on the laptop CPU over `full/`.
  - Ouro-2.6B phase W launched with plan sha256 b3902243 (the file after the ouro slots were appended; the frozen
    part, sha256 759b6bc0, equals the 34ef3dc3 launch).
- 2026-10-08 22:40 UTC, RA-16 item 5 (written before any row): GSM8K copy on ifm_p1_ouro_inj, then ifm_d112, on
  the shared machine (role `ifm_host`, `full/`), the same design as the ifm_p1_ouro run: launch code sha256
  1dba6a6a (unchanged), data_v6, pool s200, the model's design n (N_VB 60, N_SUB 20 inj / 30 d112, VBD 30, A2 D2 and
  D1 40), `--pi_n 5` (Q-PI), `--datasets VBD,D2,D1` (DV16), FD check and bf16 check first. No n cut (Q4 not
  applied). Predictions: the section 8 rows for these two models, unchanged. Order per model: FD, bf16, phase W,
  slots appended here, phase M. Launched 22:42 UTC with plan sha256 65e46d92 (code 1dba6a6a).
- 2026-10-08 22:43 UTC, RA-16 item 2 (written before any row): the gradient-free null rankers of X8 [CB-F4],
  NORM-top and IDX-latest, on the GSM8K copy (VarBench) items of ouro, ouro26, ouro_think, ouro26_think (`full/`).
  New file `loop-flow/src/k17_vbrank.py` (imports k17_varbench, never modifies it, so the launch code
  sha256 1dba6a6a of the IFM runs stays valid). Forwards: clean A and clean B per phase-M item that has G (batch 1,
  bf16 model, float32 readout), recording the state at r at every G site and V1 site; NORM(s) = ||H^B_s[r] -
  H^A_s[r]|| in float32, the same difference the carry gain dots with g_s. No intervention forward: every dm comes
  from the stored O1 rows. IDX(s) = executed order (loop, layer), so IDX-top is the latest site. Rankers are scored
  within item with `hit_metrics` (tolerance hit, band 0.05 x a x D, eligibility THR[a]; strict hit as diagnostic;
  regret on items with D >= 2) on (i) the 9 verified sites at a = 1 and a = 0.25, all items; (ii) the verified
  sites in loops 1..K-1 (items with >= 3 such sites); (iii) dev items, full G set at a = 1, and its loops 1..K-1
  part. Paired per item: CG - IDX and CG - NORM (hit on eligible items, regret on D >= 2), 95% problem-cluster
  bootstrap (k17_varbench.boot). Checks: clean top-1 tA / tB and clean margins reproduce the stored values; NORM
  at the V1 sites agrees with states_<m>.pt (fp16); CG hit and regret on (i) and (iii) reproduce checks.json.
  ouro was scored on the GB10 (role ouro_gb10): its norms come from the fast host and are labelled cross-host.
  Prediction (existing, X8 pass rule; none new): CG hit above IDX and NORM hit (paired CI lower bound > 0) and CG
  regret below IDX-latest and NORM-top regret (paired CI upper bound < 0), on S and on S_<K; point directions
  reported too. Runs on the fast host card 0 after the selective-control Ouro-2.6B job ends.
  - ifm_p1_ouro_inj (RA-16 item 5): FD pass (a = 1e-2 median rel err 0.0045, sign 1.00); bf16 check band 30/30,
    loops < K 30/30, FN class 1.00, pass. Phase W (20 items, 0 excluded): no undo slots (A5); relay dep full 0.85,
    after Llast@3 0.80, after Lmid@4 0.43, random relay -0.0008. Phase M launched with the phase-W plan sha256
    65e46d92 (nothing to append) and landed 23:14 UTC (1680 s, 8788 forwards).
  - Rankers (RA-16 item 2) ran 23:15-23:19 UTC on the fast host card 0 (role `ouro_host_rank`, code sha256
    69d6e54a, hash line in `rank/hashline.txt` written before the first row); results in
    `artifacts_spark/k17_varbench/full/rank/`.
  - ifm_d112 (RA-16 item 5): FD pass (a = 1e-2 median rel err 0.0023, sign 1.00); bf16 check band 30/30, loops < K
    30/30, FN class 0.998, pass. Phase W slots (30 items, 1 excluded from dep): selected [Lmid@2, Llast@2, Lmid@3];
    informative ("a+b2") Llast@2 and Lmid@3; Lmid@2 "a+b1_only" (n_b2 3). removal / carried_over_DKO: Lmid@2 0.97 /
    0.37 (n 3), Llast@2 0.94 / 0.60 (n 11), Lmid@3 0.92 / 0.68 (n 14), Llast@3 0.03 / 0.98 (not selected), Lmid@4
    0.00 / 0.99 (not selected). Relay dep full 0.79, after Lmid@3 0.71, after Llast@3 0.03, random relay -0.001.
    Phase M launches with the plan sha256 after this entry.
- 2026-10-09, cluster bundle (RA-20 Amendment 2): the launch code line in section 9 now names the cluster copy
  cluster/pilot/k17_varbench.py (sha256 b61194d81af961de457a9417311d16a13c2d592baae8a28dee16f3052e44c494; the shard version f8d3d427 with paths from the
  environment, pinned Ouro revisions and no host or home strings; no numeric change). Plan sha256 before this
  change 032ea3c70280652ae97dfe53f3b667326a8dc43cb6093d6b56e2ba354f5dd33d (without section 13: 759b6bc0f4835b8403e8ca6219506daf91b900e6793692d4986e6587c9f8361c). After the line change, before this entry:
  7b88e1831d711afb2b4a44cf84bac0c520d4e7e1eaf770f055371d820c9c7f7e (without section 13: 633418a01118a2d96642d578a87358fe9b2edd4d25fc618c85fcba4209ae6bb2). The sha256 of the file with this entry is the value
  passed as VB_PLAN_SHA; the code chat reports it.
- 2026-10-09, cluster bundle, final: host names in this plan replaced by the role labels shared-host, fast-host and second-shared-host (60, 56 and 2 places; one remote folder name likewise) and drive prefixes removed (4 places); no other text changed. The launch code line in section 9 names the final cluster copy (sha256 a161b00c06238fc00653853306488318ae4e42a09d57564b56c579ee18714468; it adds the default-off --no_gradient_rank flag, which the job list turns on). Plan sha256 before these changes 097a0fc2e80b31d81c22745757cd77148322d07a23479024bc56daf1dd675cb5 (without section 13: 633418a01118a2d96642d578a87358fe9b2edd4d25fc618c85fcba4209ae6bb2); after them, before this entry, 809853c84287aea7dffc60e46f8781d878ab5d18a89515164351c8531c5f360f (without section 13: 35b379a5f7998f95feea51ab542b76dbc5456c6a2eccbbf2be7304a675487710). The sha256 of the file with this entry is VB_PLAN_SHA; the code chat reports it.

## 14. Review log (v3, code review before the full run)

Each finding of the code review, what changed, and the test that covers it. Code: `k17_varbench.py` v3. CPU
tests: `--selftest` (all keys below) and `--summarize` re-run on copies of the pulled debug artifacts. GPU: the
v3 debug job (section 13). Findings with the same id from two reviewers are listed together.

- MS1, F1, F1' (answer endpoint capped below 1): fixed. Answer-position rows use first-token e as primary in
  cell(), strata_cells(), TOST, answer classes, cross-model cells and predictions; e_LO stays primary for source
  rows, undo and SZ (`primary_f`). Answer cells add e_LO_over_pc = dLO / D and the PC-final e_LO ceiling. Section 5
  amended. Test: summarize on the debug rows gives PC-final e = 1.00 on all 7 models with e_LO_over_pc = 1.00,
  where v2 printed e_LO 0.65 [0.24, 1.0] (ouro26) and 0.74 (ouro26_think). No GPU rerun was needed for this part.
- MS2, F2 (bf16 readout, saturated sites, self-referential hit): fixed. (a) Readout in float32 with TF32 off
  (`readout32`, IFM `out32`); hook check logs `readout`. (b) T7 v3 tolerance hit (0.05 D) with band chance;
  A2-1 hit and regret on dev items over the full set; test baseline "random site in the selector's top 3"; strict
  hit kept as a diagnostic. Tests: selftest `fp32_readout` (fp32 error 1e-5 vs bf16 0.25 against float64,
  gradient flows, TF32 restored), `hit` (tie inside the band scores a hit, chance 2/9, top-3 expected hit 2/3,
  D < 2 not eligible). GPU: v3 debug hook checks pass with the fp32 head (section 13).
- MS3, F6 (D1 dev/test share clusters): fixed. Subject-level split, cross-split pairs dropped, select() quota key
  follows the split, dyadic bootstrap over subjects for D1 CIs, shared-subject count in checks.json. Test:
  selftest `d1_split` (0 shared subjects) and `boot2`; on the Ouro tokenizer 0 shared subjects vs 70 shared
  A-side clusters under the K15 split [ran]. Cost: D1 at most 69 pairs (DV11).
- MS4, F4 (no intersection cells, P-ST, P-O2, dose clause not computed): fixed. `cross_model` paired cells,
  `strata_diff` (cluster bootstrap of the difference of stratum means), O2 paired with O1 on r2_pass items, A2-1
  dose clause, all scored in predictions. Test: selftest `group_diff`; summarize on debug produces 4 Ouro and 3
  IFM pairs with 24 cells each.
- MS5, F7 (A7 FN rate missing; host names and home paths in checks.json): fixed. FN rate with |G| < 0.05 D and
  carried = e >= 0.5 on dev items; gpu_minutes keyed by role and mode; co_tenants hold memory numbers only;
  model_copies hold repo ids and home-relative paths; `scrub` removes host and user names from checks.json,
  results.json and summary.md; nvidia-smi records keep the script name only. Test: selftest `scrub`; summarize on
  debug: no host, user or home string in any of the three artifacts (grep [ran]).
- MS6 (dep without a small-denominator guard): fixed. Items with -m_A < 2 are excluded from dep, the band and
  rule (a); counts reported (`dep_excluded_small_margin`). Test: selftest `dep_guard`.
- MS7 (dev-tuned site by raw dm): fixed. Mean dm / D over dev items with D >= 2. Test: selftest `dev_tuned`
  (the review's 0.9 D / 0.1 D vs 0.6 D example picks the 0.6 D site). Summarize reports
  `dev_tuned_matches_v3_rule` for runs made before v3.
- MS8 (regret CIs over item index): fixed. Full-set regret, G_red subset regret and FN CIs use dataset clusters.
- F3 (bool strata counted as 0): fixed. Counts compare str(value); subject_pos0 is stored as "yes" / "no".
- F2' (A7 not applied; selector top-1 = latest site): fixed. A7 sets (section 2); the readout-proximity baseline
  is reported (`latest_site`, equal to uniform Lmid@K under A7) and hit and Spearman are reported within loops
  1..K-1 with the Llast@K-1 baseline. Test: selftest `sets` (74 / 17 / 146 sites; loop-K layers {0, Lb/2}).
  P-A2 is unchanged in wording; its dev-tuned and uniform baselines now sit in the same A7 set.
- F3' (predictions partly scored): fixed. Every section 8 row is encoded with its harness tag and endpoint;
  missing inputs give "not scored"; the d112 L0@3 and Lmid@3 answer rows are gone (only layers <= 55 are frozen).
  Test: selftest `predictions_coverage` (491 rows: 73 per Ouro model, 63 twin, 73 inj, 58 d112, 5 cross-model and
  global, all "not scored" on empty input).
- F5 (ST4 cuts from t* applied to prompt length; ST4 not per cell): fixed. Rank cuts on prompt length for both
  tokenizers (Ouro S200 67 / 66 / 67), ST4 in every O1 cell's strata. Test: selftest `rank_cuts`; rebuild [ran].
- F8 (pair choice differs from GEN): fixed. Lowest var_name among valid rows; `gen_pair_id` stored; agreement
  198 / 200 on S200 and 500 / 513 overall [ran]. Test: selftest `pick_pair`.
- F9 (S1 not runnable with the flags): fixed. `--gset`, `--n_sub`, `--a2_n`, `--role` added and recorded in
  `config`; S1 commands, reachable D1 n and the GPU-hour estimate rewritten from these flags (section 10). Test:
  selftest `per_model`.
- Rejected: none. Not done: the `--shard` slice and merge step (S1 still runs as one job per model; section 10).

### Review log v6 (code review 3 on launch code 0e25471e, 2026-10-08)

Ids: A = first reviewer (F1-F6), B = second (F1-F2), C = third (F1-F7). Code: `k17_varbench.py` v6. CPU tests:
`--selftest` (all v4, v5 and new `v6` keys pass [ran]); `--build_items` into `data_v6` [ran]; `--summarize` on a
copy of the v5.2 merged debug [ran]. GPU: laptop debug (RTX 4070, `--drop_block`, Ouro-1.4B: W, M with BKO and G_PI,
VBD, D2, D1, bf16 check) and a shared-host IFM debug (section 13 v6).
- C-F2 (major, T3b subtraction biased) fixed: T3b v6 (section 5). Decides A-F2 and A-F5. Rejected part: "an
  upper-bound, null or 'not carried' claim at a site where RND fails T3 is not citable". Reason: O1 has no random
  component, and nonspecific damage mostly moves the margin toward the tie (v5.2 debug [ran]: RND e is positive at
  40 of the 42 cells that fail T3, up to +0.78; the other two are -0.05 and -0.07), so it makes an upper bound
  harder, not easier, to meet; marking those cells not citable would drop the Ouro answer rows for no bias reason.
  They cite O1 with the label "RND fails T3", and the RND mean and the O1 - RND CI stay in the cell, so a reader can
  see the two cells where damage points the other way. The in-distribution control (third-
  number donor) is not built (cost; the gate already blocks a nonspecific "works" claim). Test: selftest
  `v5.rnd_pair` (cited O1 0.305 where RND fails, gate open on O1 - RND 0.105), `v6.t3b_gate` (O1 0.9 with the gate
  closed: lower-bound row missed; gate open: held; upper-bound row held on O1). Rescored v5.2 debug: VBD answer
  Llast@1 cites O1 +0.08 / +0.15 (ouro26 / ouro26_think), VBD source Llast@1 O1 0.94 / 0.93 with the gate open (O1 -
  RND 0.77 / 0.80, CI > 0).
- A-F2 (major, P-ST on raw O1) resolved by C-F2: under T3b v6 the cited value is O1, so the strata and cross-model
  differences are on O1 in both models (the subtraction it asked for is the rule C-F2 removed). Each cell records
  its basis and both sides' RND T3 verdicts (`strata_diff.basis`, `cross_model.cells.<k>.rnd_T3_A/B`).
- A-F5 (minor, mixed basis in answer_class) fixed by its second option: final e and e_own are both O1; the RND
  verdict is a label, and "carried" at a failing site needs the gate. Test: selftest `v5.rnd_pair` (class final e
  0.305, basis "O1").
- A-F1 (major, dose band) fixed: band HIT_TOL x a x D, dose clause paired on items eligible at both doses with a
  cluster CI, hit - chance per dose reported (T7 v6). Test: selftest `v6.dose_band` (the reviewer's case: miss at a
  = 1, hit at a = 0.25 under the v5 band, miss under v6), `v6.dose_pair`.
- A-F3 (major, bf16 gate weak) fixed: `--bf16_check` on 30 fresh items, the run set, full-set and loops < K argmax,
  CP bounds reported; the gate counts >= 30 items with D >= 2 regardless of the file's `--bf16_n` (section 4 V6 v6).
  Test: selftest `v6.bf16_v6` (loops < K band fails while the full set passes; CP(4/4) 0.398, CP(30/30) 0.884),
  `v6.bf16_v6_pass`; laptop bf16 check on Ouro-1.4B, 3 items, G_full [ran].
- A-F4 and C-F7 (minor, relay >= direct not forced; dep > 1 misread) fixed: row tallied on min(dep, 1); section 13
  wording corrected. Test: selftest `v6.relay_not_harness`. Rescored v5.2 debug: with clipping, relay < direct
  still holds on ifm_p1_ouro_inj at every boundary up to Llast@3 (full 0.88 vs 0.94) and on d112 at the full
  knockout (0.93 vs 1.00), so on that debug the row would miss on two IFM models for reasons other than saturation
  [ran, n = 3, not interpreted].
- A-F6 (minor, CGCHECK debug file opens the gate) fixed: `cgcheck_gate` needs PASS, no failed or insufficient cell
  and the full-run sha256. Test: selftest `v6.cg_gate`; summarize with the debug file gives cgcheck_pass false.
- B-F1 (minor, random relay keys after the span) fixed: pool 1 .. min(span) - 1 (`rand_ko_keys`, pool rule stored
  in the row). Test: selftest `v6.rand_keys` (gsm8k-188 keys 60 and 258, both before the span at 591).
- B-F2 (minor, report misreads -0.64) fixed: section 13 v5.2 text corrected; the report repeats the correction.
- C-F1 (major, b2 ceiling) fixed: b2 per D_KO with guard 2, D_KO / D reported, BKO_c in phase M. Test: selftest
  `v6.b2_dko` (car / D 0.15 fails, car / D_KO 0.60 passes; D_KO 1.5 below the guard leaves b2 no items),
  `v5.slot_stats` (BKO added). Rescored v5.2 debug ouro: Llast@2 car / D 0.32, car / D_KO 0.94.
- C-F3 (major, prefix filter misses words) fixed: `prose_flags` with number words on copy-read and VBD prefixes;
  items rebuilt (section 1). Test: selftest `v6.prose` (gsm8k-357 and gsm8k-595 cases, a clean prefix passes).
- C-F4 (major, G4 and G8 missing) fixed: G_PI row (section 4 V6 v6) and G8 hashes (section 9 v6), G1-G8 map in
  "Amendments applied". Test: selftest `v6.pi_gain` (toy m = |h|^2: G_PI = 2.75 exactly), `v6.pi_summary`,
  `v6.g8`; laptop debug: G_PI on VarBench, VBD and D1 dev items [ran].
- C-F5 (minor, host role manual) fixed: GPU name in every launch, refusal on a GPU change, checks keyed on (role,
  GPU). Test: selftest `v6.host_gpu`.
- C-F6 (minor, stale estimate) fixed: section 10 "Config S1 v6".
- Open (coordinator): the v6 pilot projection exceeds the fast-host guard (section 7 v6); the IFM pilot needs Q-PI; the
  LEDGER should record the CGCHECK checks.json sha256 (46a4e865...) next to its plan hash; a new plan sha256 (after
  the section 9 "Launch code sha256" line was set to v6) is needed before phase W.
