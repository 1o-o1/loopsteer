# B-K17-PUBLIC plan: Huginn-0125, McLeish r32 and its same-data non-recurrent Llama (v5)

v1 written before any code or GPU run (recon, 2026-10-07 21:22). v2 = the state of `k17_public.py` at resume (code of
22:15 applied a first critic round, "critic 1", but the plan text was not updated before the stop at 22:25; v3 writes
those changes into the text, marked "v2"). v3 applies the PLAN.md amendments A1-A8 again (section "Amendments
applied") and the second adversarial critic, "critic 2", findings F1-F13 (section 15). Nothing in this file has been
run at full scale. v4 (2026-10-08) applies review 3 (findings F1-F12, section 16 'Review log'), including PLAN.md
G4 (path-integrated gain) and G8 (hashing before execution). v5 (2026-10-08, before the shared machine run) adds only the shared machine
run scope S0 (13.3) and DV16; predictions and rules are unchanged from v4. Brief: PLAN.md B-K17-PUBLIC with K17 amendments A1-A8, audit `notes/audit_pilot_code.md` section 5
(P1-P14, binding), `notes/audit_path_ouro.md`, and the orchestrator's addition: smcleish/Llama-3.2-non-recurrent-posttrained
as the non-recurrent reference for McLeish r32, mapped by executed depth. Tags: [read] at the cited file:line,
[ran] executed (CPU or remote), [belief] not executed, [harness] true by construction (kept out of the held/missed
tally).

Source keys: `HUG:` and `MCL:` = the modeling files cited in `notes/k17_raven_facts.md` (Huginn bb6621b6 copy,
McLeish a5f6f126); `RAV:` = `E:/Research/loopedcot/prod/models/raven.py`; `K10H:` = `E:/Research/loop-flow/src/k10_hf.py`;
`KPO:` = `E:/Research/loop-flow/src/k16_path_ouro.py`; `PAPER:` = arXiv 2511.07384 (McLeish et al., "Teaching
Pretrained Language Models to Think Deeper with Retrofitted Recurrence"); `CODE:` = `E:/Research/loop-flow/src/k17_public.py`.

## Amendments applied (PLAN.md "K17 amendments for the paper", re-read for v3)

- A1. Every cell in checks.json carries n_items and n_clusters (`cell()`), strata by source token count (`pos_len`
  1 vs >= 2) and paraphrase mode (O2 rows only from aligned forms, so one mode), relation, and its positive-control
  values (PC-final and SZ per (model, dataset); SC at the same site for raven, section 6). Keys the paper may cite:
  `e1_cells`, `e5_cells`, `e5_rules`, `sc_cells`, `answer_classes`, `e8`, `pn3_paired`, `k16_vs_k32`, `fd_check`,
  `tiny_floor`, `cross_model_D1_capitals`.
- A2 / A6 / A7. E8 = carry-gain site selection on Huginn, McLeish and NR (D1, D2, D4): G over every answer site of S,
  O1 verify at top-3 / median-3 / bottom-3 at a = 1 and 0.25, a full search over S+ on search items, top-1 hit as the
  primary quantity, scored as selector hit minus latest-site hit (critic 2 F1, section 9 T7). S excludes the
  readout-feeding site (raven post-coda; NR layer 15) and, on raven, the coda-gated Llast@K (kept in S+ as secondary;
  DV5). A7 false-negative rate per dataset. E8 is interpreted only on a model whose FD check passes (critic 2 F2).
- A3. VarBench arm uses the copy-read target (VARBENCH v3/v4 I1-I6: teacher-forced gold solution up to the token
  before the first restatement of x, read dz(x') - dz(x)), source = the number span, answer = the last teacher-forced
  position r = t* - 1; clean filter top-1 = x on A and x' on B; full core pool, one pair per problem.
- A4. Not applicable (GEN only).
- A5. Huginn and McLeish run only E1, E5 (j = K-1 included) and E8. No knockout, window or undo arm. NR is labelled
  "same-data non-recurrent reference, not a depth control" (checks.json `nr_map.label`). BOS and space-digit merging
  are checked per tokenizer by `--build_items` (section 1.4, build_meta.json `tokenizers`).
- A8. Section 1.3 with file:line sources; executed-index maps asserted in every forward (hook check k).

## 0. Scope

Core (A5, never cut): E1 placement (source vs answer, by loop), E5 input protocols including j = K-1, E8 carry gain
with verify. Models: Huginn-0125 at K = 32 (K = 16 and K = 8 reduced, same weights), McLeish r32 at K = 32,
Ouro-1.4B-Thinking as anchor (E1 only, same item texts where they pass), and NR with E1 and E8 at every layer.
Datasets: D1, CounterFact aligned subset (CF), D2, D4 T1/T2, D5, VarBench copy-read (VB). Cap 8 GPU-hours on the shared machine.

## 1. Recon facts

### 1.1 Checkpoints on the shared machine [ran, sha256 first 16 hex]

| key | repo @ revision | files checked |
|---|---|---|
| huginn | tomg-group-umd/huginn-0125 @ bb6621b6 | tokenizer.json 9cc201a5061b70ab, config.json e9fe79df06a783ca, raven_modeling_minimal.py a1d447da93c605a6 |
| mcleish | smcleish/Recurrent-Llama-3.2-train-recurrence-32 @ a5f6f126 | tokenizer.json 6b9e4e7fb171f92f, config.json 19b8171b615eaff5, raven_modeling_minimal.py 4f7a86e9b4ce9e73 |
| llama_nr | smcleish/Llama-3.2-non-recurrent-posttrained @ 004e96ac | tokenizer.json 6b9e4e7fb171f92f, config.json d0df35983c4f20bd (no README, no generation code) |
| ouro_think | ByteDance/Ouro-1.4B-Thinking @ 3aaa2224 | tokenizer.json fcb808fe5e7642f5, config.json 44e42ec2ed97f49f |

McLeish and NR share the tokenizer file byte for byte, so every McLeish item is an NR item with identical ids.
No download is needed.

### 1.2 The non-recurrent Llama (NR) and the "same data" claim

- Config [ran]: LlamaForCausalLM, 16 layers, hidden 2048, 32 heads, 8 KV heads, head_dim 64, intermediate 8192, rope
  llama3, vocab 128256, untied, bos 128000, stored fp32. McLeish r32: 4/6/4, n_embd 2048, `injection_type: linear`,
  mean_recurrence 32, `sampling_scheme: poisson-lognormal-filling`, test_time_noise 0 [ran].
- Data [ran, HF metadata]: both list `datasets: nvidia/Nemotron-CC-Math-v1` and `arxiv:2511.07384`; NR lists
  `base_model: meta-llama/Llama-3.2-1B`. PAPER: the non-recurrent baselines post-train the original model on the same
  data in the same order [read]. That the released NR checkpoint is exactly that baseline is [belief] (no README).
- NR is matched on data and initialization, not on architecture or compute (it keeps layers 4-5, has no adapter and
  runs 16 layers where McLeish executes 200). Label: "same-data non-recurrent reference", never "depth control".
- Layer origin [read, PAPER 4.3.2, Table 9]: McLeish prelude = Llama layers 0-3, recurrent block = 6-11, coda = 12-15.

### 1.3 Executed-index maps, norms, readout, initial state (A8)

All from `notes/k17_raven_facts.md` F2-F7 [read there at the file:line given]; asserted at runtime.

| model | executed index of (loop j, layer l), l 0-based | loop-end norm | readout from the loop-end state | initial state |
|---|---|---|---|---|
| Huginn | prelude 0-1; core 2 + 4(j-1) + l (HUG:675-679, 775-777); loop end 4j+1; coda slots -1, -2 (HUG:695-698) | every core block ends in `norm_4` (HUG:533-536); after the last loop ln_f(x) (HUG:760) | ln_f, coda[0..1], ln_f, lm_head (HUG:695-710) | trunc-normal std 0.0087 x emb_scale 72.66 (HUG:802-811) |
| McLeish | prelude 0-3; core 4 + 6(j-1) + l (MCL:661-668, 763-765); loop end 6j+3; coda slots -1..-4 | none (MCL:466-469); raw x returned (MCL:749) | coda[0..3], ln_f, lm_head; logits bf16 (MCL:684-699) | std 0.0087 x 1.0 (MCL:884-893) |
| NR | layer L = L, 0..15 | none between layers; final `model.norm` | norm, lm_head | none |
| Ouro-1.4B-Thinking | 24(j-1) + l, K = 4 | `model.norm` output at each loop end is the carried state (KPO:6) | lm_head on it; trained exit at every loop | none |

- Hooks read `int(args[2])` at call time and never keep the tensor (McLeish increments it in place, MCL:761-765).
- Depth-j readout: neither raven model passes the step index into the blocks, so the readout of the clean loop-j state
  through coda and head equals the model run at `num_steps = j` [ran: hook check (g), bit for bit, tiny and real].
  Both models were trained with sampled depth (mean 32), which rarely samples j <= 2; whether depth-j outputs are
  answers is measured per item by D_j (section 9 T2, critic 2 F4), not assumed.
- Gradients: an int `num_steps` runs every loop under `torch.no_grad()` (HUG:742-753, MCL:731-742); E8 passes
  `num_steps = torch.tensor([0, K])` (MCL:728-729); hook check (e) asserts its logits equal the int-K run.
- Never call `predict_from_latents`, `iterate_one_step`, `embed_inputs` or `compile_mask` (raven facts F5, F9).
- Contraction evidence [ran, k10 updates, GSM8K answer position]: McLeish relative update 52 at pass 1, 0.0096 at pass
  16, plateau 0.0089 after 20; Huginn (K = 16) 1.84, 0.57 at pass 5, 0.136 at pass 16.

### 1.4 Tokenizers and BOS [ran, laptop tokenizers, build_meta.json]

- Huginn: no BOS (prod convention, measured better on GSM8K). The space merges with the FIRST digit only: " 45" =
  ["Ġ4", "5"]. A two-digit answer is two tokens; D2/D5 first-token margins are leading-digit margins (P12), the
  remaining digits pass a teacher-forced full-answer filter (v2). "Paris" at position 0 has no space.
- Llama (McLeish, NR): BOS 128000. The space before a number is its own token 220; numbers up to 999 are one token.
- Ouro: no BOS (K16 convention).
- D1 candidate cells after the build (relation x pos_len) [ran, build_meta.json]: Huginn capital 1-token 161,
  capital >= 2 5, city >= 2 79 (all position 0, dropped at runtime), city_mid 1-token 43, city_mid >= 2 26.
  Llama capital 1-token 230, city 1-token 3, city >= 2 86, city_mid 1-token 150. Ouro capital 230, city 1 + 64
  (all position 0, dropped), city_mid 130. On Llama every >= 2 subject is in the city template; on Huginn the >= 2
  subjects are city_mid or capital. So pos_len and template are confounded on Llama (critic 2 F8).

## 2. Decisions

- DC1 Depth. Huginn K = 32, K = 16 and K = 8 (K = 8 only for PB8); McLeish K = 32; Ouro K = 4. Every raven forward
  passes `num_steps` explicitly and asserts K adapter calls and K calls per core block. Loop set J = {1, 2, K/4, K/2,
  K-1, K}: K = 32 {1, 2, 8, 16, 31, 32}; K = 16 {1, 2, 4, 8, 15, 16}; K = 8 {1, 4, 6, 7, 8}; Ouro {1, 2, 3, 4}.
- DC2 Precision. bf16 weights on every model for the scored rows, float32 readout (post-ln_f state at r through
  `lm_head.weight.float()`), TF32 off, batch 1. The carry-gain gradient is checked against finite differences with
  fp32 weights on the same items (FD check, section 7 E8a; critic 2 F2), and the bf16 vs fp32 rank agreement of G is
  reported. The bf16 noise floor for small answer edits is measured by the tiny-edit row (critic 2 F13).
- DC3 BOS, frozen. Huginn no BOS; McLeish and NR BOS; Ouro no BOS. On no-BOS models any item whose edited span
  (main or paraphrase form) includes position 0 is dropped and counted (P5). v2: D1 adds the template
  "The city of {} is located in the country of" (`city_mid`) so city subjects survive on no-BOS models.
- DC4 Initial state (raven). Per item one draw on a CPU `torch.Generator` seeded `int(sha256(pair_id)[:8], 16) +
  seed_offset`, `trunc_normal_(std, -3 std, 3 std)` at the longest row length of the item, times `emb_scale`,
  rounded to bf16 (also for the fp32 FD check, so both dtypes start from the same values), sliced per row. A, B and
  every edited row share it. Hook check (j): std within 2% and the prefix property of the CPU draw [ran, tiny].
  3-seed sensitivity: seed_offset {0, 1, 2} on the D1 clean scan and on source O1 at Llast@16, both raven models.
- DC5 Endpoints. m = z(tB) - z(tA) at the read position r. D-sets: tA, tB = first differing answer tokens (shared
  answer prefix ids are appended to the prompt; Llama: after the shared space). Items whose remaining answer digits
  are not the greedy continuation on A or B are dropped (v2 full-answer filter). VarBench: source rows primary e_LO
  (full-span log-odds over the number's tokens), answer rows primary first-token e (VARBENCH v3 rule). e = dm / D
  only where D >= 2.
- DC6 Executed-depth map for NR. Prelude end: NR layer 3 output <-> McLeish prelude output e. Core loop end j: NR layer
  N(j) = 3 + ceil(8 j / K) (K = 32: j = 1 -> 4, 2 -> 4, 8 -> 5, 16 -> 7, 31 -> 11, 32 -> 11). Coda input: NR layer 11
  <-> McLeish Llast@32. NR runs every layer, so any map can be read from the sweep. v3 (critic 2 F6): the PN3 (b) and
  (c) comparison layer is min(N(j), lower 95% bound of NR's read window L* measured on dev items), section 10 PN3.
- DC7 Code. `k17_public.py` is self-contained: the pure helpers of `k17_parcae` are copied verbatim (md5 70d1d754 at
  copy time; `--selftest` asserts parity with k17_parcae and k17_varbench where importable). It imports `k15_data`
  (8a742446), `k16_data` (b4a70c6b), `k13_edit` and `k17_varbench` (copy-read text helpers) only in `--build_items` on
  the laptop; their md5s are written to build_meta.json and to every run meta. The md5 of k17_varbench changed after
  the build (becd89a8 at build time, 5c517bcd now); the built candidate files are fixed and hashed, so this does not
  change any item.

## 3. Hooks (exact tensors)

Installed once per runner, gated by a per-forward context; call counters reset each forward; with no context they do
nothing. Several runners on the same weights (Huginn K = 32, 16, 8) each act only on their own context.

Raven (Huginn, McLeish):
- R1 core block output: forward hook on `transformer.core_block[l]`; the pre-hook asserts `int(args[2])` equals the
  1.3 formula; call n -> loop j = n. Sites L0@j (l = 0), Lmid@j (l = C/2: Huginn 2, McLeish 3), Llast@j (l = C-1).
  Applies edits, records, and adds zero grad leaves at r for E8.
- R2 post-injection: forward hook on `transformer.adapter`, call j: post_inj(j).
- R3 injected input: pre-hook on `transformer.adapter`, call j: args[0] = cat([x, e]); returns a clone whose e half
  carries the edited rows at the listed positions (B's e, e_A + n (e_B - e_A), or e_A + random).
- R4 prelude output e: recorded from R3 at call 1.
- R5 coda: forward hook on `transformer.coda[c]`; coda[Cd-1] output = post-coda state (readout-feeding site).
- R6 readout: forward hook on `transformer.ln_f`, last call (Huginn calls it twice), output at r -> float32 lm_head.
- R7 depth-j readout: the forward's own path from a loop-end state (ln_f on Huginn, coda, ln_f, head).

NR: forward hook on `model.model.layers[L]` (L = 0..15); readout = `model.model.norm` output at r -> float32 lm_head.
Ouro: layer hooks l < 23 and the `model.norm` hook (l = 23, carried state) per loop; readout per loop = float32 head on
the norm output (trained exits); eager attention.

Hook checks before any scored row (checks.json `hook_identity`, per runner, first kept D1 item): (a) call counts in
every forward; (b) zero add at every site and every e call, plus zero grad leaves, bit-identical; (c) repeat
bit-identical; (d) an edit at loop j position p leaves earlier calls and later calls at positions < p bit-identical;
(e) the `[0, K]` grad run equals the int-K run, and every leaf receives a nonzero gradient; (f) modeling-file sha256;
(g) depth-j readout equals the model at num_steps = j, j in {1, K/2, K}; (h) IN-single(j): e at other calls equals
clean A's e, at call j B's e at the source and A's elsewhere; (i) B's e at all positions at every call gives clean B
bit for bit; (j) initial-state std and prefix property; (k) executed index asserted; (l) SC at Lmid@j equals SC at
Llast@j bit for bit (both replace every position of the loop-j state path). Status [ran]: all pass on tiny random
models of both raven modeling files and a tiny Llama (the shared machine CPU); real-model results in section 13 run log.

## 4. Sites

| model | E1 sites per loop j in J | count | readout-feeding (PC-final) |
|---|---|---|---|
| Huginn K=32, McLeish | post_inj(j), Lmid@j, Llast@j; plus L0@1, L0@32 | 20 | post-coda at r |
| Huginn K=16 | post_inj(j), Llast@j | 12 | post-coda at r |
| Huginn K=8 | PB8 rows only: Llast@7, Llast@6, Llast@4 (source) | 3 | post-coda at r |
| NR | layers 0..15 | 16 | layer 15 at r |
| Ouro-Thinking | L0@j, Lmid@j, Llast@j, j = 1..4 | 12 | Llast@4 at r |
| VB (raven) | post_inj and Llast at j in {1, K/2, K-1, K}, Lmid@K | 9 | post-coda at r |

Positions: source = the differing span (D1/CF subject, D2/D5 operand, D4 value token, VB number span); answer = r.
Last-loop answer cell (A7): raven Lmid@K; Ouro Lmid@4; NR layer 14. Llast@K on raven feeds only the coda; it is an E1
cell labelled "coda-gated", never the last-loop answer cell or a member of S.
PB8 extra rows (critic 2 F10): Huginn source O1 at Llast@(K-r), r in {1, 2, 4, 8}, at K = 32 and 16 on D2 and D4.

## 5. Operators

- O1 donor: H[s, P] <- lerp(H, H^B[s, P], a), a = 1 unless stated (exact copy); donor = clean B at the same site
  with the same initial state.
- O2 "cross-template same-pair transfer" (critic 2 F12): H[s, P] += H^{B,para}[s, P'] - H^{A,para}[s, P'], from the
  first paraphrase form whose source span has the same length and position-0 class as the main prompt and whose A
  and B pass the clean filter (P4). No broadcast fallback. The vector is built from the evaluated item's own entity
  pair, so these rows test template invariance of the pair difference, not transfer of a vector to new items; no
  transfer claim is made from them. Diagnostics cos(v, own delta) and |v| / |own delta| per row (P7).
- RND: per position a unit Gaussian (seed = sha256(pair_id|site|pos|p)) scaled to |H^B - H^A| at (s, p). v3: at every
  E1 site and both positions (critic 2 F5).
- IN edits (raven, R3): B's e rows at listed positions and calls; IN-single(j)xn = e_A + n (e_B - e_A) at call j only
  (critic 2 F3); RND-input = e_A + a random vector of norm |e_B - e_A| per position at calls 1..K.
- Tiny O1: a = 1e-3 at the answer-class sites (critic 2 F13); the fraction of state elements that the bf16 lerp
  actually changes is logged with each row. v4 (review 3 F7): diagnostic only. The bf16 lerp at a = 1e-3 rounds most
  elements away (changed fraction 0.001-0.22 in the debug), so it cannot fail and does not gate anything. The a = 0.25
  noise is measured directly in the FD check (7 E8a).
- |delta| per edited position and sums over calls are logged (P1): |H^B - H^A| summed over the span at every site,
  |e_B - e_A| and |W_e (e_B - e_A)| per call (source positions and the positions after the source), and
  |W_s (H^B - H^A)| at Llast@j (the push of a state edit into the next loop).

## 6. Controls

- PC-final [harness]: answer O1 at the readout-feeding site at r equals clean B bit for bit on every item, else stop.
- PC-site(s) [harness]: O1 at ALL positions at site s, plus on raven B's e at every later adapter call; must equal
  clean B bit for bit at every E1 site (first 20 kept items per dataset per model; Huginn first 10). It fails only on
  a hook error.
- SC(s), state ceiling, informative (raven): O1 at ALL positions at site s, e unchanged. v3 (critic 2 F7): at EVERY
  E1 site with j < K (K = 32: post_inj, Lmid, Llast at j in {1, 2, 8, 16, 31} plus L0@1, 16 rows; K = 16: 10 rows;
  VB: 6 rows). SC at post_inj(K) and Llast@K are [harness] (nothing reads e after the last call). Reading rule
  (frozen, every E1 site with j < K): a source null (mean e <= 0.15 or TOST null) is attributed to "recurrence forgets
  state" where SC(s) <= 0.15, to "source position" where SC(s) >= 0.5, else "unattributed".
- Informative positive controls on non-injection models: NR source O1 at layer 3, Ouro source O1 at Llast@1.
- RND same-norm at every E1 site, both positions (all models). Validity (critic 2 F5, EXPERIMENT_MATRIX 5.4): flip
  <= 0.02. Its e is reported as the floor of the cell, not gated. RND-input (E5 floor): same rule.
- SZ [harness]: source O1 and RND on the readout-feeding state at the source positions; r logits equal clean A bit for
  bit. VarBench also a causality check (continuation positions never change r).
- References: clean A, clean B; per-item clean margins and the gap tB - tA logged (P14).

## 7. Arms

- E1 placement (all models). O1 at every site x {source, answer}; RND at every site x {source, answer}; answer O1 at
  a = 0.25 at the class sites (raven Llast@j for j in J and Lmid@K; NR all 16 layers; Ouro all 12); tiny O1 at
  a = 1e-3 at the same sites on the first 20 kept items (critic 2 F13). O2 at 8 cells (raven source Llast@1,
  Llast@K-1, Llast@K, post_inj(K); answer Llast@1, Llast@K-1, Lmid@K, Llast@K; NR layers 3, 7, 11, 14 at both
  positions; Ouro Llast@1, Llast@3, Lmid@4) on items with a passing aligned form.
- E5 input protocols (raven, source positions):
  - IN-single(j): B's e at call j only, j in {1, 2, 4, 8, 16, 24, 28, 30, 31, 32} (K = 16: {1, 4, 8, 15, 16}). Label
    "argument only" (P13).
  - IN-persist(j): B's e at calls j..K, j in {1, 2, 3, 4, 8, 9, 16, 17, 24, 28, 30, 31} (K = 16: {1, 2, 4, 8, 15}).
    j + 1 values (3, 9, 17) exist for the SRCIN decomposition (v2). VB: IN-single at j in {K-1, K}, IN-persist at
    j in {1, K/2, K-1} (v4, review 3 F8: K-1 added so the frozen E5 rule and the absolutely matched pair at j = K-1
    can be read on VB; P13).
  - IN-single(j)xn, absolute matching (critic 2 F3): one call at j carrying e_A + n (e_B - e_A), n = K - j + 1, so the
    summed |W_e delta e| equals IN-persist(j): j in {K/2, K-1} (n = 17, 2) and j = 1 (n = 32, labelled off-manifold).
    Relative matching (same per-call strength) = IN-single(j) vs IN-persist(j). Both are scored (PB4).
  - IN-persist-down(1): B's e at every position after the source at every call.
  - RND-input-persist(1): the input-edit floor (critic 2 F5).
  - SRCIN(j) (v2): source O1 at Llast@j plus B's e at calls j+1..K, j in {1, 2, 8, 16, 31}; state_own = SRCIN(j) -
    IN-persist(j+1) is the part of a source state edit that survives on top of the refreshed input.
  - State before vs after the last injection: Llast@(K-1) vs post_inj(K), source, from E1, with both summed deltas.
  - SC rows (section 6). Every E5 cell reports the summed |W_e delta e| over its calls (P1).
  - Frozen E5 reading rule (critic 2 F7): an IN-single(j) null (e <= 0.15 or TOST null) reads "erased by refresh" only
    where IN-persist(j) >= 0.5 on the same items; otherwise "no working input edit at j". IN-persist "works" uses the
    5.4 rule against RND-input.
- E8a FD check (critic 2 F2; LEDGER FINAL FRAME D3), per model (Huginn, McLeish, NR), before E8 is read: fp32 weights,
  10 D1 + 10 D2 items (the kept items of the main run first, then candidates in build order), site classes raven
  {Llast@1, Llast@K/2, Llast@K-1, Lmid@K}, NR {layer 4, layer 7, layer 11, layer 14} (their DC6 images), O1 at r at
  a in {1e-6, 1e-3, 1e-2}. Llast@K-1 added after the v3 debug (DV9): on McLeish, G and FD at Llast@1 and Llast@K/2
  are both below the floor on every debug item, so those classes are "insufficient" and Lmid@K alone does not test
  the backward through a recurrence step. Each class also reports `n_G_below_FD_above` (zero G, FD above the floor:
  a detached-leaf sign); it is a diagnostic, not part of the frozen rule. CGCHECK frozen rule (k17_cgcheck_plan H8, H11): floor F = max(Z, N99), Z = zero-edit repeat |dm|, N99 = 99th
  percentile of |dm(1e-6) - 1e-6 G|; a cell is above floor iff |a G| > F; a class passes at a = 1e-2 iff median
  relative error <= 0.05 and sign agreement >= 0.95 on >= 10 above-floor cells (D1 and D2 pooled per class; per
  dataset reported; < 10 cells = "insufficient"). Verdict PASS iff every counted class passes. The same items are then
  rerun in bf16 (model cast in place). v4 (review 3 F6): the bf16 vs fp32 agreement of G is reported as Spearman over
  the above-floor sites of S (|1e-2 G_fp32| > F) and as top-1 and top-3 agreement over all of S (E8 takes the argmax
  over S); the all-S Spearman is a diagnostic, because below the floor G is 1e-13..1e-5 and its rank is noise that E8
  never selects (debug: McLeish all-S 0.79-0.98, above floor 0.993-0.9995, top-1 and top-3 equal on 4 of 4 items).
  v4 (review 3 F7): answer O1 at a = 0.25 at the class sites (raven Llast@j, j in J, j < K) in fp32 and in bf16 on
  the same items; noise = |e_bf16 - e_fp32| (each precision's own D, D >= 2 in both) per site (`a025_noise`).
  Without PASS, E8 on that model is reported and not interpreted.
- E8 carry gain (Huginn, McLeish, NR; A2, A6, A7). S at r: raven all core-block outputs except Llast@K (Huginn 127,
  McLeish 191); NR layers 0..14 (15). S+ = S plus Llast@K (raven, secondary). One clean-A forward and backward with
  zero leaves at every site of S+: G(s) = grad_{H_s[r]} m . (H^B - H^A)_s[r]. Verify: O1 at the top-3, median-3 and
  bottom-3 sites of S (and of S+, union) and one random site of S at a = 1 and 0.25. Full search (critic 2 F1, F9):
  O1 at a = 1 at every site of S+ on search items = dev and cross items, up to 60 per dataset on McLeish and NR and 25
  on Huginn (all of S, not a site subset; v1's 36-site Huginn subset is withdrawn). Dev-tuned site = argmax of mean
  dm / D over dev items with a full search; scored on test items at both doses with the uniform rule (last-loop answer
  cell), a random site, and a random site of the selector's top 3. Ouro E8 is run by B-K17-VARBENCH.
- v4 (review 3 F1, PLAN G4): path-integrated gain PIG next to G. PIG(s) = mean over a in {0.25, 0.5, 0.75, 1} of
  grad_a(s) . (H^B - H^A)_s[r], where grad_a is the gradient at the answer leaf of site s in a run whose source-span
  token embeddings are lerp(emb_A, emb_B, a); every site follows the input (EAP-IG, input path). Four extra forward
  and backward passes per E8 item, the same leaves as G. Per-dose terms are stored (`Ga`). Checks per item: the a = 1
  run equals clean B bit for bit (`pig_a1_equals_B`); A and B ids differ only at the source span. The per-site path
  (a lerp at site s only, the rest free) would need 4 x |S| passes (764 on McLeish), more than the full search it
  should replace; and setting every site at r to its lerp in one pass blocks the gradient through the later sites
  at r ('set' overwrites r; at a = 1 every leaf but the last gets zero gradient). PIG's top-3 sites of S are added
  to the verify rows at both doses. Scored next to G: full-set top-1 hit, PIG - G hit and PIG hit - latest-site hit
  (paired cluster CIs, a = 1, search items); dm / D at each estimator's top-1 site on all items at both doses.
- v4 (review 3 F4): E8 pilot gate on raven models. The first min(10, cap) search items in kept order are always
  fully searched (the pilot). The later search items are fully searched only if (eligible rate over S at a = 1 on
  the pilot) x (search items under the cap) >= 20; otherwise they get the verify rows only and the (model, dataset)
  E8 cell is descriptive (`e8_gate`, `descriptive_by_gate`). NR has no gate (its full search is 15 forwards). The
  rule is fixed here from the debug data (eligibility over S: McLeish D1 2/2, D2 1/2, D4 0/2; Huginn D1 2/2, D2
  2/2), before any scored row (G8).
- Presence (supporting, "causal": false): pair projection at every raven loop end and NR layer, validated per site
  by paraphrase separation (as Parcae v2).
- Huginn K = 16 (reduced): E1, E5 and PB8 rows on D2 and D4, on items kept at K = 32 that also pass at K = 16 (v2:
  D2, D4; Huginn D1 at K = 32 is capital/city_mid only). Huginn K = 8: harness rows and PB8 rows on the same items.

## 8. Items, filters, strata

- Targets per model and dataset (kept, after D >= 2): McLeish and NR (joint filter, identical items) D1 80, CF 60, D2
  80, D4 60, D5 60, VB 60. Huginn K = 32: D1 60, CF 40, D2 60, D4 40, D5 40, VB 40. Ouro-Thinking: D1 80, D2 80, D4 60,
  with McLeish's kept pair_ids scanned first. DV2. v4 (review 3 F2): the Selector caps bound what any target can
  reach, even at 100% clean-filter yield (`--count_caps` on the candidate files): D1 Huginn 73, Llama 108, Ouro 78;
  D2 65 on every tokenizer (78 operand units, cap 2), so the D2 targets of 80 and 60 are unreachable as written and
  D2 is capped at 65; D4 1,036; D5 86; CF Huginn 896 (after the position-0 drop), Llama 3,261; VB 512 / 513.
- Clean filter (batch 1): top-1 tA on A, top-1 tB on B, D >= 2, and the full-answer filter (DC5). McLeish and NR:
  kept only if both pass (paired); solo pass counts reported. Report scanned, A_fail, B_fail, D_small,
  full_answer_fail, pos0_dropped, kept. A dataset with fewer than 40 kept items is reported and not interpreted (T8).
- D1 (v2, v3): templates capital, city (subject first) and city_mid (subject mid-sentence), pairs within a
  relation with different answers and an equal-length contiguous difference; at most 2 uses per (template, subject)
  on either side. v3 (critic 2 F8): the candidate order is round-robin over relation x pos_len cells, so kept items
  are balanced over the cells the tokenizer supplies; the kept relation x pos_len table is reported per model. Cross-
  model D1 comparisons use capitals only (`cross_model_D1_capitals`, intersection of capital pair_ids). The capital
  ceiling is 49 pairs per model (49 capital subjects, cap 2 per side); city_mid adds up to 33 (DV6).
- CF: `k15_data.cf_candidates` on FULL-A's cf_records (relations P17, P103, P364, P641, P176), round-robin over
  relations, aligned paraphrases only (P4); at least 4 relations kept or the shortfall reported.
- D2: `k15_data.d2_raw`; quota round-robin over the edited operand; cap 2 per operand value on either side.
- D4: `k16_data.d4_items` T1 and T2 with the generalised encoder (Huginn space-digit merging); rows whose ids differ
  exactly at the source span (P10(2)); one item per content key on either side; round-robin over (split, kind). v4
  (review 3 F5): the v3 code capped only the A-side key (345 of 1,351 kept items shared a content unit with an earlier
  item). The Selector now caps both keys (`cl`), cap 1, at runtime; the candidate files are unchanged. Upper bound
  1,036 items on every tokenizer (A-side rule: 1,351).
- D5: `k16_data.d5_items`; one item per problem on either side (audit C16); copy-confound and demo-problem items
  dropped (P12). D5 is not an E8 dataset.
- VarBench: VARBENCH I1-I6 per tokenizer, one pair per problem (lowest var_name), k17vb-order; cluster = GSM8K idx.
- Clusters (v2, confirmed for critic 2 F11): dyadic over the A-side and B-side entity for D1 and CF (subjects), D2
  (edited operand values), D4 (content keys) and D5 (problems); one-way over the problem for VB. The CI bootstrap
  resamples units with weight c[A] x c[B] (audit_path_ouro F8).
- Splits (critic 2 F9): a unit is dev if the first hex digit of sha256(prefix + unit) is even, else test; an item is
  dev (test) only if every unit hashes dev (test), else "cross". Expected fractions: dyadic items 1/4 dev, 1/4 test,
  1/2 cross; VB 1/2 dev, 1/2 test. Measured counts per (model, dataset) in checks.json `clusters`. Dev/test unit
  overlap is 0 by construction (asserted). Use: full search on dev + cross items; dev-tuned site fitted on dev, scored
  on test; PN3 (b), (c) scored on non-dev items.
- Strata (A1, P3): pos_len, relation or kind, paraphrase mode, "edit includes position 0" (dropped on no-BOS models,
  counted), endpoint kind (full answer vs leading digit), VB ST2, ST3, cont. Every cell reports n_items, n_clusters.
  Cross-model comparisons only on the intersection of kept items and on shared strata.

## 9. Metrics, statistics, frozen thresholds

- Per row: dm, dz(A), dz(B), flip (top-1 = tB), e = dm / D (D >= 2), e_LO (VB), depth-j margins (raven, Ouro),
  presence, |delta| sums (P1).
- CIs: 95% percentile cluster bootstrap, 1000 resamples, seed 0, ratio of sums, dyadic where the cluster is a pair.
  Paired differences use the same bootstrap on per-item differences. Nulls: "null by TOST" only if the 90% cluster CI
  lies inside [-0.10, 0.10]; otherwise "below <95% upper bound>".
- NR read window: L* = last layer with mean source e >= 0.5, with a cluster-bootstrap interval (200 resamples), on all
  items and on dev items (critic 2 F6).
- T1 (v3, critic 2 F5 = EXPERIMENT_MATRIX 5.4): a cell "works" if mean(e - e_RND) >= 0.5, flip >= 0.5 and the 95%
  cluster-CI lower bound of e - e_RND >= 0.3, with e_RND from the same site and position; "fails" if e is null by
  TOST; else "partial". The raw mean e is reported next to it. VB source rows use e_LO.
- T2 answer classes (raven and Ouro, loops < K, thresholds scaled by a). v3 (critic 2 F4): e_own(j) = dm_own(j) / D_j,
  with D_j = the depth-j clean gap (m_B(j) - m_A(j) at the loop-j readout) and the guard D_j >= 2; items below the
  guard are "not testable at site". "Carried" if mean final e >= 0.5a (all items). Otherwise, on gated items only and
  only with >= 10 gated items: "not carried" where mean e_own >= 0.5a and final e < 0.25a, "no answer state at site"
  where e_own < 0.5a, "partial" otherwise; fewer than 10 gated items -> "not testable at site". Raven relabel (v2):
  a non-carried class becomes "state forgotten" where SC at the same site <= 0.15 and "not read (SC < 0.5)" where SC
  < 0.5. Every class cell reports n_items, n_gated, mean D_j / D. At a = 0.25 a class other than "carried" is scored
  only where the measured a = 0.25 bf16 noise (mean |e_bf16 - e_fp32| at that site on the FD items, 7 E8a; v4,
  review 3 F7) is below half the "not carried" cut (0.5 x 0.0625 = 0.03125); otherwise "not scored". Huginn K = 16
  class sites have no FD rows, so their a = 0.25 classes are "not scored". NR: final e only (no trained exits), no
  class.
- T3 RND validity (v3, critic 2 F5): flip <= 0.02; e reported as the floor.
- T4 PC-final, PC-site, SZ, SC at post_inj(K) and Llast@K: bit-identical on every item, else stop.
- T7 carry gain (v3, critic 2 F1 = VARBENCH v4 T7). Eligible item at dose a: D >= 2 and verified dm range > 2 (a = 1)
  or > 0.5 (a = 0.25). Hit = dm at the top-predicted site >= max verified dm - 0.05 D. Latest-site hit = the same for
  the fixed site nearest the readout: over S the latest site in execution order (Huginn Lmid@32 = c32.2, McLeish
  c32.4, NR layer 14), within loops 1..K-1 Llast@K-1 (raven). Scored A2-1 endpoint: selector hit - latest-site hit, a
  paired cluster CI, on eligible search items over the full set at a = 1; the same within loops 1..K-1 (scored with >=
  10 eligible items). "Adds beyond readout proximity" is a claim only where the CI lower bound is > 0. Band chance,
  strict hit and top-3 random hit are diagnostics. Co-primary: regret = (best verified dm - dm at the top site) / D.
  Spearman per eligible item. A7 false-negative rate on search items: among sites with |G| < 0.05 D, the fraction with
  verified e >= 0.5 (cluster CI), and the fraction of carried sites with |G| < 0.05 D. Dose clause on the verified 9
  of all items. A model's E8 is interpreted only with FD PASS, >= 20 eligible search items per dataset and a pilot
  gate that did not stop the search (v4). PIG is scored with the same endpoints (v4, 7 E8).
- T8 fewer than 40 kept items: reported, not interpreted.

## 10. Predictions (written before any full run; [belief] unless tagged [harness])

Mechanism reading. Both raven models were trained from a random initial state at sampled depth, so their output
must not depend on the state they start from: the recurrence is pulled to a state fixed by e. A state edit at loop j
is a new starting state for K - j more loops and should be forgotten when the map has converged (McLeish by j = 16),
while edits to e, or in the last loop, reach the output. Huginn converges more slowly, so it should keep more of a
mid-run edit than McLeish. NR and Ouro have no re-injection, so source edits persist until the read window ends.

- PB1 (E1 source, McLeish). Llast@j for j <= 16: e <= 0.10 on D1, D2, D4, VB (null by TOST on D1 and D2). Llast@31:
  e <= 0.30. post_inj(32): e in [0.3, 0.9] on D1. Llast@32 (coda-gated): e in [0.1, 0.7].
- PB2 (E1 source, Huginn K = 32). Llast@1 and Llast@2: e <= 0.15; Llast@16: e <= 0.30; Llast@31: e in [0.2, 0.8] on
  D1, D2, D4; post_inj(32) >= 0.4 on D1. Huginn keeps more than McLeish at Llast@16 on D2 and D4 (difference >= 0.05).
- PB3 (E1 answer, raven; v3 restated on gated items). Answer O1 at Llast@j for j <= 16: final e <= 0.10 (all items).
  On D1, at j in {2, 8, 16} where >= 10 items pass D_j >= 2, the class is "not carried" or "state forgotten". Lmid@K:
  e in [0.3, 0.9]. Llast@K (coda-gated): e >= 0.6. McLeish D2 answer Llast@j, j <= 16: e <= 0.15. [belief: on
  McLeish D_j at j <= 2 is near 0 on most items, so those cells will be "not testable at site".]
- PB4 (E5). Relative matching: IN-single(j) for j <= 30: e <= 0.10; IN-single(31) <= 0.25; IN-single(32) >= 0.3 on D1.
  IN-persist(j) >= 0.7 for j <= 16 on D1 and D2; |IN-persist(1) - IN-persist(16)| <= 0.15; IN-single(32) <=
  IN-persist(31) <= IN-persist(1). IN-persist-down(1) >= 0.5 on D1. Absolute matching (critic 2 F3): IN-single(16)x17
  <= 0.15 and IN-persist(16) - IN-single(16)x17 >= 0.5 on D1 and D2 (the same summed input push works only when
  repeated); IN-single(31)x2 <= IN-persist(31). Reading rule: the IN-single(j) nulls for j <= 16 read "erased by
  refresh". IN-single(1)x32 is reported only (off-manifold).
- PB5 (before vs after the last injection). Source e(post_inj(K)) - e(Llast@(K-1)) >= 0.3 on D1 (McLeish, Huginn) and
  D2 (Huginn), with both summed deltas reported next to it.
- PB6 (state ceiling). McLeish: SC(Llast@1), SC(Llast@16) <= 0.10, and SC at post_inj(j) and Lmid@j <= 0.15 for j <=
  16 (v3). Huginn: SC(Llast@1) <= 0.15. SC(Llast@K-1) in [0.1, 0.7]. SC(post_inj(K)) = SC(Llast@K) = 1 [harness]. So
  the PB1 nulls are attributed to forgetting, not to the source position.
- PB7 (seed). McLeish: over 3 seeds the D1 pass count changes by <= 1 item and source Llast@16 by <= 0.02. Huginn: <= 3
  items, <= 0.05. v4 (review 3 F9): the offset-0 scanned pair_ids are re-run with no Selector and statuses compared
  item by item; the item-level symmetric difference is scored with the same bounds next to the count change.
- PB8 (v3, critic 2 F10). Erasure depends on the loops remaining after the edit. For r in {1, 2, 4, 8}, source O1 at
  Llast@(K-r) agrees within 0.10 (paired, same items) between K = 16 and 32, K = 8 and 16, K = 8 and 32 on D2 and D4;
  a pair is scored only when both cells have e >= 0.2. The depth fractions (for example r = 4: 0.5, 0.75, 0.875) differ,
  so agreement at equal r with unequal fractions separates the two readings. IN-single(K-1) agrees within 0.15 between
  K = 16 and 32.
- PN1 (NR source, all items). e >= 0.8 and "works" (5.4) at layers 0..7 on D1; L* in [9, 13] on D1 and in [5, 11] on
  D2; e <= 0.15 at layer 14.
- PN2 (NR answer). e <= 0.15 at layers <= 7; e >= 0.5 at layers 12..14; e >= 0.8 at layer 14 on D1.
- PN3 (paired McLeish vs NR, same items). (a) prelude end: NR source layer 3 vs McLeish IN-persist(1): both >= 0.7,
  paired difference within 0.15. (b) v3 (critic 2 F6): NR source at layer b = min(7, lower 95% bound of L*_dev) minus
  McLeish source Llast@16 >= 0.5 on D1 and D2, on non-dev items (the headline contrast); the difference at every NR
  layer l <= L*_dev is reported as a curve. (c) NR source at layer c = min(4, L*_dev lower bound) minus McLeish source
  Llast@1 >= 0.5 (non-dev). (d) NR answer layer 11 and McLeish answer Llast@32 both >= 0.5. (b) and (c) are scored
  with >= 20 non-dev items.
- PD (SRCIN decomposition, McLeish D1 and D2). At j = 16: SC <= 0.10, source O1 <= 0.10, IN-persist(17) >= 0.7,
  |state_own| <= 0.10. At j = 31: state_own in [0.05, 0.6].
- PO1 (Ouro-Thinking anchor). Source O1 Llast@1 >= 0.8 and "works" on D1 and D4; answer O1 Llast@1 final <= 0.10 with
  class "not carried" (gated) on D1; D2 answer Llast@1 >= 0.5.
- PE8 (v3, critic 2 F1; per model and dataset, a = 1 unless stated). A2-1: selector hit - latest-site hit >= -0.10 over
  S on search items [belief: the latest site is usually inside the band on retrieval, so the selector is predicted to
  match readout proximity, not to beat it]; within loops 1..K-1, selector hit - Llast@K-1 hit >= -0.10 where >= 10
  items are eligible. Mean regret <= 0.15. A7 false-negative rate <= 0.10. hit9(0.25) >= hit9(1) - 0.15.
  |selector - dev-tuned site| <= 0.10 in dm / D (test). Descriptive: top-predicted site in the last loop on >= 80% of
  raven items. "Adds beyond readout proximity" is not predicted; it is read from the CI.
- PF (FD check). McLeish, NR and Huginn PASS. v4 (review 3 F6): median Spearman(G_bf16, G_fp32) over above-floor
  sites >= 0.9 and top-1 agreement on >= 90% of items (replaces the all-S Spearman, which is now a diagnostic).
- PT (v4, review 3 F7; replaces the tiny-floor prediction, which could not fail). The a = 0.25 bf16 noise is below
  0.03125 at every raven class site [belief: bf16 noise in e is about 0.01].
- PG4 (v4, review 3 F1; written before any PIG row ran) [belief]. PIG top-1 hit - G top-1 hit >= 0 on the full set
  at a = 1 per (model, dataset), scored only where the cell is interpreted. Specific belief: on Huginn D2, where G is
  negative at the last-loop sites but their a = 1 edits are positive (debug item D2|op1|36 + 11: G -2.47 at c32.0,
  e(1) +0.142), PIG ranks a last-loop site first more often than G.
- PC (controls). RND valid (flip <= 0.02) at every cell; PC-final, PC-site, SZ, SC harness bit-exact [harness].
- PY (yield). Kept >= 40 for McLeish/NR on D1, D2, D4 and VB; Huginn D1 >= 40 (capitals + city_mid); CF < 40 on
  every model; Huginn D5 < 40.

## 11. Budget, order, cut rules, GPU etiquette

- Rows per item (v3) [ran: counted in the debug run, section 13]. McLeish D-set item about 230 rows plus the full
  search (192) on search items; Huginn about 200 plus 128; NR about 110 plus 15; Ouro about 60.
- Cost: replaced by the debug timings in section 13 (run log) before the full run is launched.
- Cut order if the projection exceeds 7.0 h: (1) Huginn K = 16 and K = 8 on D4, (2) Ouro D4, (3) O2 rows, (4) Huginn
  CF, (5) Huginn D5 40 -> 30, (6) McLeish/NR CF and D5 60 -> 40, (7) Huginn VB 40 -> 30, (8) Huginn full search 25 ->
  15 per dataset (then Huginn E8 is descriptive, < 20 eligible). v4 (review 3 F12): every rule can now be set from
  the command line: targets below 40 need `--descriptive` (recorded in the pre-run hashes and reported as
  descriptive), `--full_search_cap huginn=15` is rule (8), `--full_search_cap mcleish.D4=0` drops one search,
  `--pc_site_n` sets the PC-site subset. Never cut: hook checks, PC-final, PC-site, SZ, SC, E1
  O1 and RND at every site, E5 single, persistent and absolute-matched rows including j = K-1, the FD check, the
  McLeish/NR pairs on D1, D2, D4 and VB, E8 on D1, D2 and D4 for McLeish and NR (D4 protected from cut 6, critic 2 F9).
- Order: CPU self-tests and `--build_items` (laptop; done) -> tiny-model selftest on the shared machine CPU -> debug (2 items per
  dataset per model) and hook checks -> FD check debug -> clean scans and seeds -> McLeish + NR (one process, shared
  items) -> FD check McLeish/NR -> Huginn K = 32, then K = 16 and K = 8 -> FD check Huginn -> Ouro anchor -> summarize.
  v4 (G8): before the first scored row, `--hash_inputs` prints the plan sha256, code md5, candidate-file sha256 and
  the run configuration; the coordinator writes that line into the LEDGER; every `--run` writes the same record to
  `inputs_<tag>.json` and the run meta before the models load, and a scored (non-debug) run refuses to start without
  `--plan`.
- Etiquette: the shared machine is shared with the IFM part of B-K17-VARBENCH (and Parcae/closers jobs). Before every launch run
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and `free -g`; launch only with >= 30 GB available.
  At most two k17_public processes at once. Stop only with SIGTERM to an anchored pattern:
  `pkill -TERM -f '^\.\./\.\./\.venv/bin/python k17_public\.py'` (the processes are started from `src/` with that
  relative interpreter path). If the network drops, jobs continue; check `out/k17_public/` before rerunning.
  Results are resumable by pair_id.

## 12. Verification and artifacts

- CPU: `python k17_public.py --selftest` (helpers, parity with k17_parcae and k17_varbench, bootstrap, TOST, site maps,
  executed-index formulas, DC6 map, selection rules, splits, v3: SC counts, latest site, hit - latest toy, e_own guard,
  5.4 works rule, FD pass rule); `--selftest_model` (tiny random raven model from each cached modeling file and a tiny
  Llama: hook checks a-l, controls, every arm incl. IN-single(j)xn (n = 1 equals IN-single bit for bit), RND-input, SC
  rows, PB8 rows, the K = 8 runner, the FD path in fp32 then bf16, the dev-tuned pass, summarize).
- v4 (review 3): `--selftest` adds the D4 both-side cap, PIG summary, arch_of(huginn8), CLI overrides, search plan
  and pilot gate, the scan deadline flag, VB IN-persist(K-1), above-floor rank agreement and the a = 0.25 noise rule;
  `--selftest_model` adds PIG at a = 0 equal to G bit for bit and at a = 1 equal to clean B, PIG top-3 verify rows, a
  driver test (sequential run vs stage scan + rows 0/2 and 1/2 + merge: same kept list, full-search set, gate and
  rows bit for bit; deadline flag; rows refused on an incomplete scan) on a K = 16 raven runner and on the tiny Llama,
  a res_huginn8 file in the summarize directory, and an FD file read by summarize.
- GPU: debug into `out/k17_public/debug/v3/` (v4 rows: `debug/v4/`), then the full run into `out/k17_public/full/`.
- Artifacts: remote `<home>/loop_flow/out/k17_public/` (`res_<model>_<dataset>.json`, `meta_<group>.json`,
  `fd_<group>.json`, `checks.json`, `summary.md`), pulled to `E:/Research/loop-flow/artifacts_spark/k17_public/`.
  Candidate files: built on the laptop in `E:/Research/loop-flow/out/k17_public/data/` and copied to remote
  `out/k17_public/data/` (build_meta.json holds their sha256; DV7).
- checks.json keys (no host names): `plan_sha256`, `code_md5`, `reused_md5`, `checkpoints`, `load_audit`,
  `hook_identity`, `exec_index_map`, `nr_map`, `endpoint`, `items`, `clusters`, `split_fraction`, `pc_final`, `sz`,
  `sc_harness`, `pc_site`, `sc_cells`, `e1_cells` (with `src_vs_RND`, `ans_vs_RND`, `SC_same_site`, `src_attribution`,
  `O2_label`), `e5_cells`, `e5_rules`, `decomposition`, `answer_classes`, `tiny_floor`, `presence_validation`,
  `nr_window`, `pn3_paired`, `k16_vs_k32`, `cross_model_D1_capitals`, `seed_sensitivity`, `fd_check`, `e8` (per model,
  dataset, set, dose: `full_set_search` with `hit_minus_latest`, `loops_lt_K`, `regret`, `fn_rate_search`; `verified9`;
  `test`; `dev_tuned_site`; `fd_verdict`; `interpreted`), `o2_label`, `predictions_vs_outcomes`, `gpu_minutes`, `timing`.
  v4 adds: `pre_run` (G8 hashes per run), `a025_noise` (F7), `fd_check.*.rank_agreement_bf16_vs_fp32` (F6),
  `e8.*.pig` (F1: `full_set_search.pig_minus_G_hit`, `pig_hit_minus_latest`, `top1_dmD_all_items`), `e8.*.e8_gate` and
  `descriptive_by_gate` (F4), `items.*.flag`, `scan_complete`, `rows_missing`, `search_plan` and
  `datasets_short_by_budget` (F10), `tiny_floor_note` (F7). `tiny_floor` is a diagnostic.

## 13. Run log and Slurm scale configuration

### 13.1 Run log

- 2026-10-07 22:13 (v2 builder, before the stop): candidates built on the laptop and copied to the shared machine `out/k17_public/data/`.
  No GPU row was run.
- 2026-10-07 23:04-23:32 [ran] `--selftest` (laptop) ok; `--selftest_model` on the shared machine CPU ok (tiny McLeish, Huginn,
  Llama; hook checks a-l, controls, every v3 row type, FD path).
- 2026-10-07 23:17-10-08 00:40 [ran] debug v3 (`--debug 2 --no_seeds`, 2 items per dataset per model) into
  `out/k17_public/debug/v3/`: McLeish + NR on D1, CF, D2, D4, D5, VB (35.3 min); Huginn K = 32 on D1, D2, D4, then
  K = 16 and K = 8 on D2, D4 (Huginn CF was dropped from the debug relaunch to save time; its scan ran: 2 kept of 11).
  Hook checks a-k all true on the real McLeish, Huginn K = 32/16/8 and NR (log `v3_*.log`). PC-final e = 1.00 and flip
  1.0, SZ exact, SC at the readout-feeding site exact, on every debug item. The shared machine was shared with Parcae, VARBENCH IFM
  and closers jobs (GPU at 95 percent), so the timings below are contended.
- Debug numbers (2 items, descriptive only). Raven source O1 e at Llast@j, j <= 16: 0.00 on every dataset and both
  models; SC (all positions, full state of B) at Llast@j, j <= 16: 0.00; at Llast@31: Huginn 0.51 (D1) / 0.37 (D2),
  McLeish 0.05-0.23. IN-persist(j) e: McLeish 0.94-1.02 at every j (D1, D2, D4); Huginn 0.98-1.02 for j <= 24,
  0.97-0.99 at j = 28, 0.86-1.02 at j = 30 and 0.67-0.97 at j = 31 (D1 0.717 / 0.669 at j = 31, 0.913 / 0.862 at
  j = 30; D2 op1 0.872 / 0.877 at j = 30 / 31) [ran: res_huginn_D1/D2.json; v3 wrote "0.95-1.05 at every j", which
  was wrong for Huginn, review 3 F11]. IN-single(j) 0.00 for j <= 16. NR source e >= 0.9 at layers
  0-8 on every dataset; answer e >= 0.5 from layer 9-11 on. RND floor at NR layers 0-2 is up to 0.44 (CF) and 0.32 (D1).
  Not checkable from the pulled files: res_mcleish_VB, res_llama_nr_VB and res_huginn_D4 were pulled with items = []
  (the pull preceded their rows), so the VB and Huginn D4 timings and the statement "source e 0.00 on every dataset"
  rest on the remote log only until those files are pulled.
- Timing, seconds per item (contended shared machine, batch 1). McLeish: D1 290 (test item), CF 125, D2 165, D4 300 (cross
  item, full search over S+), D5 150, VB 100; the full search adds about 140 s. NR: D1 57, CF 22, D2 20, D4 25, D5 24,
  VB 19. Huginn K = 32: D1 390-520, D2 455-475 (test items), D4 590 (cross, full search).
- Clean-filter yield (debug scans, small n): McLeish + NR joint D5 2 of 129 (NR A_fail 100 of 129), CF 2 of 26
  (McLeish A_fail 15 of 26), VB 2 of 5. At this yield D5 gives about 11 joint items from 686 candidates, below the
  40-item floor of T8, so McLeish/NR D5 will be reported and not interpreted (DV10).
- FD check debug (fp32, 2 D1 + 2 D2 items): McLeish Lmid@32 median relative error 0.0084 at a = 1e-2, sign agreement
  1.0 (4 of 4 cells above floor); Llast@1 and Llast@16 0 of 4 cells above floor (G and FD both below F = 5.7e-6).
  NR layers 4, 7, 14 above floor on 4 of 4 cells, median relative error 0.004-0.008, sign agreement 1.0. Spearman
  (G_bf16, G_fp32) over S: McLeish median 0.91 (min 0.79), NR 0.998. Verdicts "insufficient" (fewer than 10 cells by
  design of the debug). Llast@K-1 class added after this (DV9); CPU selftest and the shared machine tiny-model selftest rerun ok
  (code md5 04d12f28c2e997d30668ae236e99f30f, laptop = the shared machine). FD rerun launched 00:18 (`v3_fd_mcl_nr_b.log`).
- 2026-10-08 00:25 the VPN dropped (the shared machine and the fast host unreachable until at least 00:47). Pending on the shared machine, not yet
  read: the FD rerun result (`debug/v3/fd_mcl_nr.json`, overwrites the first debug FD file; the first is kept at
  `artifacts_spark/k17_public/debug_v3/fd_mcl_nr.json`), the Huginn K = 16 / K = 8 debug rows (job 3438408), the Huginn
  FD debug (not launched) and `--summarize` on `debug/v3` (not run). Check these before any rerun.

- 2026-10-08 01:29 (v4) [ran] remote state after the VPN restart: the FD rerun (`debug/v3/fd_mcl_nr.json`), the
  Huginn K = 16 / K = 8 debug rows and the VB and Huginn D4 rows had finished; all v3 debug files pulled to
  `artifacts_spark/k17_public/debug_v3/` (the FD rerun as `fd_mcl_nr_b.json`). `--summarize` with the v4 code on that
  directory runs with the huginn8 files present (F3 check on real files).
- 2026-10-08 01:30-02:08 (v4 code md5 99004f32) [ran] laptop: `--selftest` ok, `--selftest_model` ok (tiny Llama;
  driver test sequential = sharded). The shared machine CPU: `--selftest` ok. Debug v4 into `debug/v4/`, pulled to
  `artifacts_spark/k17_public/debug_v4/`: mcl_nr `--debug 2 --datasets D2,VB` 8.2 min; FD check `--fd_items 2` (D1 2 +
  D2 2); huginn `--debug 2 --datasets D2` 11.6 min (K = 32, 16, 8). Pre-run hashes written to `inputs_<tag>.json`
  before the models loaded. Hook checks all_ok on every runner. Results (2 items each, descriptive):
  - PIG: `pig_a1_equals_B` true and ids differ only at the source on every item. Cost per E8 item: McLeish t_pig 8-9 s
    (t_grad 1.0 s), Huginn 8.7-9.0 s (t_grad 1.6-2.4 s), NR 0.8 s.
  - Huginn D2 `D2|op1|36 + 11|66 + 11` (the review 3 F1 item): G's top site over S is c24.3 (e 0.007 at a = 1); PIG's
    top site is c32.2 (e 0.423 at a = 1, the largest verified e on the item; e 0.018 at a = 0.25). Per-dose terms at
    c32.0: +0.81, +4.30, +1.87, -1.17 for a = 0.25, 0.5, 0.75, 1 (G -2.47). On the second Huginn item and on McLeish
    and NR D2, G and PIG pick the same or an equally good site.
  - FD (fp32 vs bf16): McLeish above-floor Spearman median 0.997 (min 0.993), top-1 and top-3 agreement 4 of 4; NR
    0.998. a = 0.25 bf16 noise at the McLeish class sites: mean |e_bf16 - e_fp32| 0.0013-0.0028 (max 0.0085), below
    the 0.03125 cut at every site.
  - VB: IN-persist(31) runs (McLeish e_LO 0.96, 0.99).
  - Timing (less contended than v3): McLeish D2 114-148 s per item, Huginn D2 218-270 s, Huginn K = 16 68 s, K = 8 4 s.
    The 13.2 estimate keeps the contended v3 timings.
- 2026-10-08 02:15 (v4 final code md5 8e330aa4; differs from 99004f32 only by an untruncated last summary line in
  `--selftest_model`) [ran] laptop `--selftest` ok and `--selftest_model` ok; the shared machine CPU rerun into
  `debug/v4/selftest_b.log` and `selftest_model_b.log`: both exit 0; selftest_model_ok, summarize_ok, parts and
  driver tests ok on tiny McLeish, tiny Huginn and tiny Llama (02:52). The first shared machine `--selftest_model`
  (99004f32) printed mcleish and huginn parts ok, PIG checks true and the raven driver test ok (K = 16 runner: 4 items,
  sequential = scan + 2 shards + merge, gate decided "stop" on 0/1 eligible pilot); its overall flag was cut off by the
  6,000-character print limit, which the final line now avoids.

- 2026-10-08 S0 hashes [ran, from inputs_*.json; not yet in LEDGER]: step 1 mcl_nr D1 plan v5 sha256
  4cfde8784d1a6ba32dfbbb355a75547b5d4dc6459df36365e90cc6cb967ebb9f, code md5 8e330aa4, hashed 08:00:06 UTC; step 3
  huginn D1 plan v6 sha256 1906d7ab2b67b01d2d287cbfa0eb97f0c179126946e34ee42264c6ac33cfad8f, code md5 8e330aa4,
  config sha256 edd73617, hashed 10:24:40 UTC.
- 2026-10-08 11:03 UTC (v7, RA-1 public core, 13.4) step 9 launched on the shared machine: mcl_nr VB, D2, D4. Plan snapshot
  `out/k17_public/plan_k17_public_v7.md` sha256 f56df948258d6ec9a2b4b1efc6aca0ceb0682c2b13558666e7e396f58188349d,
  code md5 c4c52fe156787eb3d07c0a67d1d0914c, config sha256 e7cde942ac85b92b7e21b4db1291f2eb3283fed5ba1593daa9d455aad172a92f,
  hashed 2026-10-08T11:03:03Z. The shared machine CPU `--selftest` ok on the v7 code before launch.
- 2026-10-08 13:15 UTC [ran] S0 step 3 (huginn D1) done: 31 kept of 267 scanned (pool exhausted), 170.8 process-min
  (in parallel with step 9 from 11:03, about 2x slower per item). Seeds: kept 32 / 31 vs 31, sym_diff 1 / 0.
  `s0_3_huginn_oom1.log` is an earlier attempt that stopped with a CUDA error before the scan.
- 2026-10-08 12:50 UTC (coordinator, from the PI): launches allowed until 2026-10-11 12:00, landing by 2026-10-11
  23:59; Huginn K = 32 runs after McLeish and NR. Step 10 therefore starts after step 9 ends (not in parallel).
- 2026-10-08 14:42 UTC [ran] step 9 (mcl_nr VB, D2, D4) done in 219.0 process-min: joint kept VB 60 (of 67 scanned),
  D2 65 (of 65), D4 60 (of 64). Step 10 (huginn D2) is launched with the plan snapshot `plan_k17_public_v8.md` (this file
  with this line); rules and predictions unchanged from v7.
- 2026-10-08 14:50 UTC step 10 launched on the shared machine (GPU otherwise idle, 117 GB available): huginn D2, `--targets
  D2=40 --full_search_cap huginn=0 --no_reduced --budget_min 900`. Plan snapshot sha256
  d94461ac4b20e49d5e66c8227be0273f0e235675860902de9df3bd426a69719d, code md5 c4c52fe1, config sha256
  29c85d78b74ec92c01252a30f9dd6b5bb6da9733713164b982e3296a33e3d77b, hashed 2026-10-08T14:50:47Z.
- 2026-10-08 16:01 UTC [ran] step 10 done: huginn D2 40 kept of 63 scanned, 70.2 process-min (GPU alone, 98 s per
  item). RA-1 public core landed. Process minutes: mcl_nr 118.5 + 219.0, FD mcl_nr 16.5, huginn 170.8 + 70.2; total
  595.1 (wall time on a shared GPU). All files pulled to `artifacts_spark/k17_public/full/`; summarize ran on the
  laptop into that folder.
- 2026-10-08 16:15 UTC summarize fix (code md5 a8096fcc, CPU only, no scored row changes): the PC key "RND valid at
  every cell" read the `src_vs_RND` / `ans_vs_RND` comparison entries, which have no flip, so it was always null.
  It now reads only `src_RND` and `ans_RND`. Result: 339 of 346 RND cells have flip <= 0.02; 7 Huginn D2 cells
  (Lmid@31 src, Llast@31 ans, post_inj@32 src, Lmid@32 ans, Llast@32 ans, L0@32 src and ans) have flip 0.025 (1 of
  40 items), so the PC RND key is false. `--selftest` ok.

### 13.2 Slurm scale configuration (30 x 96 GB; v4, review 3 F2)

- Config S1 (the one to scale): groups `mcl_nr` (McLeish K = 32 + NR, joint items), `huginn` (K = 32, plus K = 16 and
  K = 8 on D2/D4), `ouro_think` (E1 only). Every dataset at its full candidate pool after the clean filter; the
  Selector caps of section 8 apply once, in one global scan per (group, dataset). Row set per item as in this plan
  (v4: PIG and its verify rows on E8 items, VB IN-persist(K-1)). Full search on every search item
  (`--full_search_cap '*=9999'`) subject to the pilot gate of 7 E8; PC-site on the first 50 kept items per dataset
  (`--pc_site_n 50`). Batch 1, bf16 weights, float32 readout, the frozen thresholds and seeds of this plan. FD check
  (G, FD, bf16 agreement, a = 0.25 noise) on 50 D1 + 50 D2 items per model in fp32 (`--fd_items 50`).
- Kept items expected in S1 = min(Selector upper bound, debug yield x candidates) [ran: `--count_caps` on the
  candidate files; debug yields from the v3 scan logs, n = 2-129 scanned, so the yields are rough]:

  | dataset | bound Llama / Huginn / Ouro | debug yield (joint mcl_nr; huginn) | S1 mcl_nr | S1 huginn | S1 ouro |
  |---|---|---|---|---|---|
  | D1 | 108 / 73 / 78 | 2/2; 2/3 after position-0 drops | 108 | 49-73 | 78 |
  | CF | 3,261 / 896 / - | 2/26; 2/5 after drops | about 1,160 | <= 896 | - |
  | D2 | 65 / 65 / 65 | 2/2; 2/2 | 65 | 65 | 65 |
  | D4 | 1,036 / 1,036 / 1,036 | 2/2; 2/2 | 1,036 | 1,036 | 1,036 |
  | D5 | 86 / 86 / - | 2/129; not run | about 11 | <= 86 | - |
  | VB | 513 / 512 / - | 2/5; not run | about 205 | about 205 | - |

  Search items (dev + cross) at the bound: D1 86 (Llama) / 55 (Huginn), D2 49, D4 752. Huginn total at most 2,361
  items, McLeish/NR about 2,585 (v3 said about 4,700 and 3,300: those counts ignored the caps).
- GPU-hours [ran: per-item seconds from the v3 debug on contended shared machine (McLeish D1 290, CF 125, D2 165, D4 160 plus
  140 for the full search, D5 150, VB 100; NR 19-57; Huginn D1 455, D2 465, D4 290 plus 300 for the full search);
  belief: v4 additions (PIG + 6 verify rows: McLeish +15 s, Huginn +32 s, NR +2 s per E8 item), Huginn CF / D5 / VB
  (2x McLeish), Huginn K = 16 / 8 rows (130 s per item), Ouro (10 s per item), scan seconds per candidate (McLeish +
  NR 1.2 s, Huginn 4.4 s)]. In the shared machine GPU-hours:

  | group | D1 | CF | D2 | D4 (gate stops / passes) | D5 | VB | K16+K8 | scans | total |
  |---|---|---|---|---|---|---|---|---|---|
  | mcl_nr (McLeish + NR) | 12.5 + 1.7 | 40.3 + 7.1 | 5.2 + 0.4 | 50.8 / 79.6 + 7.5 | 0.6 | 5.8 + 1.1 | - | 6 | 139 / 168 |
  | huginn | 14.5 | 62.2 | 13.1 | 93.5 / 155.3 | 7.2 | 11.4 | 39.8 | 8 | 250 / 312 |
  | ouro_think | 0.2 | - | 0.2 | 2.9 | - | - | - | 0.5 | 4 |

  FD checks about 3 h, seeds about 0.5 h. Total about 395 shared machine GPU-hours if the McLeish and Huginn D4 gates stop the
  search (expected from the debug eligibility 0/2 on McLeish D4; Huginn D4 eligibility is unmeasured) and about 490 if
  both pass. At the 3x per-item speed of a 96 GB GPU over contended shared machine (recon_mathcode F16; a belief for this code,
  to be measured on the first shard) that is about 130-165 cluster GPU-hours. CF is about 110 shared machine-hours of it and
  carries no E8; S1' = S1 with `--targets CF=300` on both raven groups saves about 75 shared machine-hours (25 cluster-hours).
- Wall time on 30 GPUs [belief]: stage scan, one job per (group, dataset), all in parallel; the longest are the
  mcl_nr CF scan (15,066 candidates, about 5 h) and Huginn D4 (scan 2.1 h plus the 10-item pilot, about 1.7 h). Stage
  rows: about 150 cluster-hours over 30 GPUs, about 5 h. Merge, FD and summarize about 1 h. About 11 h in total.
- Commands. All groups use ONE output directory (OUT): ouro_think reads McLeish's kept list from OUT to align its
  items, summarize builds `cross_model_D1_capitals` and `pn3_paired` across groups only from one directory, and the
  shards of a (group, dataset) write `res_<model>_<ds>.shard<k>of<n>.json` next to the res file. Every stage of one
  (group, dataset) must use the same `--targets`, `--full_search_cap` and `--pc_site_n` (the rows and merge stages
  refuse a search plan that differs from the scan stage). Example for McLeish/NR D4, n = 8 shards:

  ```
  # 0. CPU, before anything runs: hashes for the LEDGER (G8)
  python k17_public.py --hash_inputs --model mcl_nr --data DATA --plan PLAN --datasets D4 --targets D4=9999 \
      --full_search_cap '*=9999' --pc_site_n 50 --stage scan
  # 1. one job: joint scan (global caps) + E8 pilot rows (first 10 search items) + gate decision
  python k17_public.py --run --model mcl_nr --data DATA --out OUT --plan PLAN --datasets D4 --targets D4=9999 \
      --full_search_cap '*=9999' --pc_site_n 50 --stage scan --budget_min 1440
  # 2. n jobs, k = 0..7: rows for kept index % 8 == k
  python k17_public.py --run --model mcl_nr ... (same flags) --stage rows --shard k/8
  # 3. one job: merge shards (kept order), dev-tuned pass (needs the GPU), D1 only: seed sensitivity
  python k17_public.py --run --model mcl_nr ... (same flags) --stage merge
  # 4. FD check per raven group, then ouro_think (after mcl_nr's scans exist in OUT), then summarize (CPU)
  python k17_public.py --fdcheck --model mcl_nr --data DATA --out OUT --fd_items 50
  python k17_public.py --run --model ouro_think --data DATA --out OUT --plan PLAN --datasets D1,D2,D4 \
      --targets D1=999,D2=999,D4=9999 --stage all
  python k17_public.py --summarize --out OUT --plan PLAN
  ```
  The huginn group's scan stage also scans its K = 16 and K = 8 runners on the K = 32 kept list (D2, D4), and its
  rows and merge stages process them. Without `--no_seeds`, merge on D1 runs PB7 (raven models).
- Cluster prerequisites: the four checkpoints at the revisions of 1.1, a venv with torch 2.11 and transformers 4.56.2
  (the shared machine `.venv` pins; the laptop's transformers 5.7 cannot load the raven remote code), the candidate files of
  `out/k17_public/data/` (sha256 in build_meta.json), no host names in outputs.
- Plan targets on the shared machine (for the DV11 scope): McLeish/NR about 26 h, Huginn about 38 h (with K = 16 / 8), Ouro 2 h,
  FD 3 h: about 70 shared machine GPU-hours (v3: 75), with D2 capped at 65 and the v4 rows included.

### 13.3 shared machine run scope S0 (v5; fixed before the first scored row)

- Cap: 8 GPU-hours on the shared machine, counted as the sum of the wall time of every k17_public process (`process_minutes` in
  the meta files, FD wall time from the FD file). The shared machine is shared (GPU use 88-90% from other K17 jobs at 07:55 UTC),
  so wall time includes waiting; this count is conservative. Processes run one at a time: in the v3 debug two
  parallel processes were 1.25-1.9x slower per item than the v4 single runs.
- The plan targets cost about 70 shared machine GPU-hours (13.2). S0 therefore runs one dataset, D1, and moves the rest to the
  cluster (S1, 13.2, unchanged). D1 is the paper's main placement dataset and carries E1, E5 and E8.
- Steps, in order (each step starts after the previous one ends):
  1. `--run --model mcl_nr --datasets D1 --targets D1=40`, default full-search cap (every search item, under the
     frozen pilot gate of 7 E8), seeds on, `--budget_min 285`. Expected about 290 min at the v3 contended speed.
  2. `--fdcheck --model mcl_nr --fd_items 10` (10 kept D1 items + 10 D2 candidates).
  3. `--run --model huginn --datasets D1 --full_search_cap huginn=0` (verify rows and PIG on every item, no full
     search: Huginn E8 is descriptive), seeds on. Target, fixed by this rule: B = 480 - U - 35 minutes (U = measured
     minutes of steps 1-2; 35 = reserve for step 4), t = mean `t_item` of the McLeish D1 test items of step 1 in
     minutes, T_h = min(40, floor((B - 17) / (1.7 t))) (1.7 = Huginn / McLeish seconds per item in the debug: 1.58 v3
     D1, 1.86 v4 D2; 17 min = load, hook checks, scan and seeds). `--budget_min B`; `--descriptive` if T_h < 40.
  4. `--fdcheck --model huginn --fd_items 10` if at least 30 min of the cap remain, else `--fd_items 5` (verdict then
     "insufficient"; the a = 0.25 noise is still measured).
  5. `--summarize` on the run directory (CPU).
- v6 extension (written 2026-10-08 08:15 UTC, after step 1 showed McLeish D1 at 101 s per test item and 174 s per
  search item, about 2.6x faster than the 13.2 estimate; written before any step-1 result was read, and fixed before
  steps 2-8 start). If cap remains after step 4, these run in order, one at a time, each with `--budget_min` = the
  cap left minus 10 min (stopped-by-budget cells are flagged and descriptive): 6. mcl_nr VB, target 40; 7. mcl_nr
  D2, target 40 (D2 E8 under the frozen gate); 8. mcl_nr D4, target 40 (D4 E8 under the frozen gate). This order
  follows the never-cut list of section 11 (McLeish/NR pairs on VB, D2, D4 and their E8). A step starts only if at
  least 30 min of the cap remain. Summarize (step 5) runs last.
- Not run on the shared machine (S1 only): CF, D2, D4, D5 and VB on every group, Huginn K = 16 and K = 8 (they run on D2/D4 only),
  the Ouro anchor, the Huginn full search.
- Kept from the never-cut list of section 11 on every S0 item: hook checks, PC-final, PC-site, SZ, SC, E1 O1 and RND
  at every site, E5 single, persistent and absolute-matched rows including j = K-1, the FD check. Cut: the McLeish/NR
  pairs on D2, D4 and VB and their E8 on D2 and D4 (DV16).
- Predictions (section 10) are unchanged. The report scores the cells S0 runs; every other prediction is "not run on
  the shared machine (S1)". Run directory: remote `out/k17_public/full/`, pulled to `E:/Research/loop-flow/artifacts_spark/k17_public/full/`.

### 13.4 RA-1 public core (v7, 2026-10-08 11:10 UTC; fixed before the rows of this section run)

- Binding scope (notes/ra_briefs.md RA-1 "Public core", adopted in LEDGER 2026-10-08 03:40): McLeish r32 and NR on D1,
  D2, D4 and VB, E1 (edit site) and E5 (input refresh) rows with the controls of section 6; then Huginn K = 32 on D1
  and D2 at n = 40, the same rows. Cut: Huginn K = 16 and K = 8, the Huginn full search, D5 and CF on public models,
  the site-selection search on public models, the Ouro anchor. No GPU launch after 2026-10-09; landing by 2026-10-10
  18:00, else Huginn D2 is dropped.
- State at this point [ran]: S0 step 1 (mcl_nr D1, 40 joint items, full E8 search; 118.5 min) and step 2 (FD check
  mcl_nr, 20 items, McLeish and NR PASS) are done. Step 3 (huginn D1, `--full_search_cap huginn=0`) is running:
  the scan kept 31 items (the clean-filter pool is exhausted at 267 scanned), so Huginn D1 is below 40 and is
  reported, not interpreted (T8). It runs to completion. E8 rows already in the D1 files stay in the files; under
  RA-1 they are not scored in this report.
- Code change (v7, md5 c4c52fe1): `--no_reduced` skips the Huginn K = 16 / K = 8 runners and the K = 32 PB8 rows
  (RA-1 cut). Default behaviour is unchanged; `--selftest` ok. E8 verify and PIG rows still run on D2 and D4 by the
  code path (no flag removes them; about 15-30 s per item); with `--full_search_cap '*=0'` no full search runs.
- Step 9 (now, parallel to step 3; the shared machine has about 89 GB available, two k17_public processes at most):
  `--run --model mcl_nr --datasets VB,D2,D4 --targets VB=60,D2=65,D4=60 --full_search_cap '*=0' --budget_min 1500`
  (targets of section 8; D2 capped at 65 by the Selector bound). Same `--out full/`, so summarize sees D1 too.
- Step 10 (after step 3 ends): `--run --model huginn --datasets D2 --targets D2=40 --full_search_cap huginn=0
  --no_reduced --budget_min 900`.
- Projection [belief, from 13.1 timings x 1.5 for two parallel processes]: step 9 about 12 h (VB 3 h, D2 4 h, D4 5 h),
  landing about 2026-10-09 00:00 UTC; step 10 about 4-5 h, landing about 2026-10-08 18:00 UTC. Both before the
  2026-10-10 18:00 deadline, so Huginn is kept.
- Then FD check for Huginn is not run (E8 is out of RA-1 scope); summarize on `full/` (CPU).
- Predictions (section 10) unchanged. The report scores PB1-PB6, PN1-PN3, PD, PC and PY on the cells run;
  PB7 from S0; PB8, PE8, PG4, PO1 are cut or not run.

### 13.5 RA-19 fills (v9, 2026-10-08 evening; fixed before the rows of this section run)

- Binding scope (notes/ra_briefs.md RA-19 items 1 and 4, Status GO): Huginn K = 32 on VB (GSM8K copy) at n = 40,
  then, if a card is free, Huginn K = 32 on D4 (code) at n = 40. Same rows as the other public models: E1
  (edit site) and E5 (input refresh) with the controls of section 6, as step 10 of 13.4. Cut as in 13.4: Huginn
  K = 16 and K = 8 (`--no_reduced`), the Huginn full search (`--full_search_cap huginn=0`), FD check. E8 verify and
  PIG rows run by the code path on D4 and are not scored here.
- Code k17_public.py md5 a8096fcc (the 16:15 summarize fix of 13.1; no row code change since c4c52fe1). Candidate
  files and build_meta.json unchanged (sha256 as in the pre-run records of 13.4).
- Host: the fast host, one card, a host not used by 13.3 and 13.4. The model files are a lab-internal copy of the
  same Huginn snapshot bb6621b6; the loader asserts the modeling-file sha16 (hook check f). Cross-host cells are
  compared only as cells, never row by row.
- Fill A: `--run --model huginn --datasets VB --targets VB=40 --full_search_cap huginn=0 --no_reduced
  --budget_min 900`, out `fill/`. Fill B (after fill A ends): the same with `--datasets D4 --targets D4=40`, same
  out. Each fill has its own pre-run record (`inputs_huginn.json`) and its own hash line in LEDGER.md.
- Predictions (section 10) unchanged: PB2 (source Llast@1, Llast@2 <= 0.15; Llast@16 <= 0.30; Llast@31 in
  [0.2, 0.8] on D4), PB3, PB4, PB6, PC; PY for Huginn VB and D4 (>= 40 kept) is new for these cells and stated
  here before the scan: [belief] VB >= 40 (512 candidates), D4 >= 40 (1,036 candidates).
- Summarize runs on `fill/` (CPU) after both fills; the D1 and D2 Huginn cells stay in `full/`.

## 14. Deviations from the brief and open points

- DV1. NR added (orchestrator), labelled "same-data non-recurrent reference" (A5); executed-depth map DC6.
- DV2. Huginn targets below 80 per dataset (cost).
- DV3 (v3, replaces v1 DV3). Huginn E8 full search covers all of S+ on up to 25 search items per dataset instead of
  36 sites on every dev item (critic 2 F1). If cut (8) applies, Huginn E8 is descriptive.
- DV4. Ouro-Thinking runs E1 only; its E8 is in B-K17-VARBENCH.
- DV5. S excludes Llast@K on raven as Parcae v2 does; A7 literally excludes only the post-coda state. S+ is reported.
- DV6 (v3, critic 2 F8). D1 per model is limited by the capital pool (<= 49 pairs) plus city_mid (<= 33); on Llama
  the >= 2-token subjects are all city-template items. Targets kept (McLeish/NR 80, Huginn 60); the shortfall, if any,
  is reported. PY now predicts Huginn D1 >= 40.
- DV7. The laptop build directory `E:/Research/loop-flow/out/k17_public/data/` is not named in the brief; it holds
  only the candidate files (copied unchanged to the remote).
- DV8 (v3). Huginn K = 8 runner added for PB8 (critic 2 F10).
- DV9 (v3 debug). FD check site classes gain Llast@K-1 (NR layer 11). Reason: Llast@1 and Llast@K/2 have G and FD
  below the floor on McLeish, so they are "insufficient" and the verdict would rest on Lmid@K alone, which tests no
  backward step through the recurrence. Diagnostic `n_G_below_FD_above` per class added. Frozen pass rule unchanged.
- DV10 (v3 debug). Joint McLeish/NR D5 yield about 1.6 percent (NR fails A on most D5 items): D5 is expected below
  40 joint items and is reported, not interpreted. No code change.
- DV11 (v3 debug; v4 numbers). The measured cost of the plan targets is about 70 shared machine GPU-hours (13.2), far above
  the 7.0 h cut threshold of section 11; cut rules (1)-(8) remove at most about 25 percent. The full run is not
  launched until the scope is decided. v4 inputs for that decision: McLeish D4 E8 is expected descriptive (pilot
  gate, 0/2 eligible in the debug); McLeish D2 E8 needs >= 41% eligibility of at most 49 search items; every target
  can now be set from the command line (`--descriptive`, `--full_search_cap`, `--pc_site_n`).
- DV12 (v4, review 3 F1). PIG (PLAN G4) uses the input path, not a per-site lerp (7 E8); four extra backward passes
  per E8 item.
- DV13 (v4, review 3 F4). E8 pilot gate on raven models, fixed before any scored row from the debug eligibility.
- DV14 (v4, review 3 F5). D4 selection caps both content keys at runtime; the candidate files are unchanged.
- DV15 (v4, review 3 F7). The tiny-edit floor no longer gates the a = 0.25 classes; the FD check measures the a = 0.25
  bf16 noise instead.
- DV16 (v5). The shared machine run is scope S0 (13.3): D1 only, McLeish/NR at 40 items with E8, Huginn at the item count
  the 8 GPU-hour cap leaves (rule in 13.3), Huginn E8 without the full search. The brief's datasets (CF, D2, D4, D5,
  VB), the Ouro anchor and Huginn K = 16 run only in S1 on the cluster. Reason: the plan targets cost about 70
  shared machine GPU-hours. v6: steps 6-8 of 13.3 use any cap left for McLeish/NR on VB, D2 and D4.
- O-1: critic K4 and EXPERIMENT_MATRIX M8 say the NR training recipe is unknown; HF metadata says
  Nemotron-CC-Math-v1 and base Llama-3.2-1B (1.2). Those files are not mine to edit.
- O-2: K10 contraction numbers come from GSM8K chain positions, not from these items; PB1-PB6 rest on them as belief.
- O-3 (closed in v4): item-index sharding inside one output directory is implemented (`--stage`, `--shard`, 13.2).
- O-4 (v4, review 3 F1; replaces the v3 builder's "Huginn E8 selector" issue, which planned to settle it with the
  Huginn FD check). The FD check cannot settle it: its classes use a <= 1e-2, where G is right (CGCHECK and the
  McLeish debug FD agree to 1%), while the failure is at a = 1. The debug data already show it on Huginn D2 (G
  negative at c32.0, c32.1, c31.3; a = 0.25 agrees in sign within 2x; a = 1 reverses the sign; c24.x, where G is right,
  wins the argmax only because the last-loop G is negative). This is dose nonlinearity, not a gradient bug. It is
  tested by PIG (PG4) next to G on the full set; the FD verdict stays a harness gate only.
- O-5 (v4). The v4 code has run the CPU selftest and the tiny-Llama selftest on the laptop; the tiny raven selftests
  (McLeish and Huginn modeling code) and the v4 debug job need the shared machine (the laptop's transformers cannot load the raven
  code). Status in 13.1.

## 15. Critic log

### Critic 1 (applied in the code before the stop; recorded in v3 from the code)

The first critic's findings were applied to `k17_public.py` (comments "critic F1".."F10") but not written here before
the stop. Changes found in the code and now in the text: E8-only dev/test/cross split by unit hash (F1); answer-class
thresholds scaled by a and the SC relabel "state forgotten" / "not read" (F3, DV6 of v2); full-set hit and chance on
fully searched dev items (F4); |W_e delta e| per call and the K-j+1 call count in the E5 notes (F6); the per-pair
initial-state seed shared by scan, rows and models, and E5 ratios to IN-persist(1) (F7); the SRCIN decomposition and
PN3 (b) as descriptive (F8; v3 makes (b) scored again under F6 below); D5 removed from E8 (F9); the full-answer
filter for multi-token answers (F10). The city_mid D1 template, dyadic clusters on D2/D4/D5, Huginn K = 16 on D2/D4,
INP_J with j + 1 values and SC at every Llast@j also date from v2.

### Critic 2 (F1-F13; all applied, two with a partial rejection of the argument)

- F1 (major, PE8 passes by construction) applied: T7 scores selector hit - latest-site hit over S and within loops
  1..K-1 (Llast@K-1), paired cluster CI, on search items over the full set; band chance is a diagnostic; PE8
  rewritten. Huginn full search covers all of S+ on up to 25 items (not a 36-site subset). Partial disagreement: the
  latest site of S is Lmid@K only on Huginn; on McLeish it is c32.4 (core layer 4 of the last loop), so the baseline
  is "the site nearest the readout in execution order", with the uniform Lmid@K kept as a test-item baseline. Code:
  `hit_v4`, `latest_site`, `summarize_e8`. Test: selftest `hit_v4`, `latest`, `e8_full`.
- F2 (major, no FD check of the raven gradient path) applied: E8a with the CGCHECK frozen rule on fp32 weights, 10 D1 +
  10 D2 items, site classes {Llast@1, Llast@K/2, Lmid@K} (NR layers 4, 7, 14), a in {1e-3, 1e-2} plus the 1e-6 floor
  probe; bf16 vs fp32 Spearman of G; E8 not interpreted without PASS. D1 and D2 are pooled per class for the verdict
  (10 items per dataset give at most 10 cells, the rule's minimum), per dataset reported. Code: `fdcheck`,
  `fd_summary`. Test: selftest `fd_rule`; selftest_model runs the FD path on tiny fp32 models (median relative error at
  a = 1e-2 below 0.05 on the tiny Llama [ran]).
- F3 (major, schedule comparisons not matched in absolute strength) applied: IN-single(j)xn with n = K-j+1 at j in
  {K/2, K-1} and j = 1 (off-manifold); summed |W_e delta e| next to every E5 cell and both deltas next to PB5; PB4
  scored under relative and absolute matching. Code: `Item.e_ed(n=)`, `run_e5`, summarize `e5_cells`. Test:
  selftest_model `in1s_n1_exact`, `v3_rows`.
- F4 (major, e_own gate not normalized or guarded) applied: e_own(j) = dm_own(j) / D_j with D_j >= 2, "not testable at
  site" below the guard or with < 10 gated items, n_gated and mean D_j / D per cell, PB3 restated on gated items.
  Code: `Dj_of`, `eown_of`, summarize `answer_classes`. Test: selftest `eown_guard`.
- F5 (major, RND rule contradicts the matrix) applied: RND validity = flip <= 0.02, e reported as the floor; "works"
  = EXPERIMENT_MATRIX 5.4 (mean e - e_RND >= 0.5, flip >= 0.5, CI lower bound >= 0.3); RND now at every E1 site and
  both positions, plus RND-input for E5; PC rewritten. Code: `rnd_names`, `label_vs_rnd`, `Item.e_rnd`. Test:
  selftest `works_rule`, `rnd_counts`.
- F6 (major, NR comparison layer not tied to the read window) applied: L* measured on dev items; PN3 (b), (c) at
  min(N(j), lower bound of L*_dev), scored on non-dev items, with the curve over l <= L*_dev. Code: `pn3`.
- F7 (major, few informative positive controls; E5 nulls without a rule) applied: SC at every E1 site with j < K and
  the attribution rule at every such site; frozen E5 rule (IN-single null = "erased by refresh" only where
  IN-persist(j) >= 0.5). Code: `sc_names`, `run_e5`, summarize `src_attribution`, `e5_rules`. Test: selftest
  `sc_counts`, selftest_model `v3_rows`.
- F8 (major, D1 on no-BOS models) applied in part. The argument that DC3 drops every city item on Huginn and Ouro does
  not hold for the v2 code: the city_mid template keeps 69 Huginn and 130 Ouro city candidates (1.4). Applied: order
  balanced on relation x pos_len, the kept cross-table reported, cross-model D1 comparisons on capitals only, the
  capital ceiling recorded (DV6), PY revised. Code: `run_group` D1 order, summarize `cross_model_D1_capitals`.
- F9 (minor, E8 cells uninterpretable at the planned counts; split fraction unstated) applied: split fractions stated
  (8); A2-1 read on search items = dev + cross (the selector and the latest rule are not fitted), test items only for
  the dev-tuned baseline; D4 protected from cut (6) for McLeish/NR E8.
- F10 (minor, PB8 cannot separate the readings) applied: Llast@(K-r) for r in {1, 2, 4, 8} at K = 32 and 16, a K = 8
  runner, pairs scored only when both cells are off the floor (e >= 0.2). Code: `run_pb8`, `k16_vs_k32`. Test:
  selftest_model `k8_rows`, `v3_rows`.
- F11 (minor, one-sided clusters on D2 and CF) applied in v2 already: D1, CF, D2, D4, D5 clusters are dyadic and the
  split requires both units in the same half (8). No code change in v3.
- F12 (minor, O2 mislabelled) applied: "cross-template same-pair transfer", no transfer claim; `o2_label` and
  `O2_label` in checks.json and summary.md.
- F13 (minor, no noise floor at a = 0.25) applied: tiny O1 at a = 1e-3 at the class sites on 20 items per model and
  dataset, |dm| / D floor and the changed-element fraction reported; a = 0.25 classes scored only where the floor is
  below half the cut. Code: `run_e1(tiny=)`, summarize `tiny_floor`.

### Audit fixes P1-P14

P1 |delta| and summed deltas, absolute and relative E5 matching (F3). P2 not applicable (no generation). P3 strata,
intersections, capitals-only cross-model D1. P4 aligned O2 only. P5 position-0 rule (DC3). P6 not applicable. P7 O2
diagnostics, answer classes at both doses with the D_j guard. P8 not applicable (no knockout). P9 n_items, n_clusters,
dyadic CIs. P10(2) and P10(5) in D4/D5 selection. P11 not applicable. P12 D5 copy-confound drop, leading-digit label
on Huginn, full-answer filter. P13 single and persistent at j = K-1 and K, "argument only" label. P14 per-item margin
gap logged; plan sha256 into checks.json; the coordinator writes it into the LEDGER before the first scored row.

## 16. Review log

### Review 3 (2026-10-08, findings F1-F12; code md5 04d12f28 -> see the run meta `k17_public_md5`)

All twelve findings were checked against the code and the pulled debug files before the fix; none is rejected.
Partial rejections of a proposed mechanism are stated.

- F1 (major) fixed. The G4 row (path-integrated gain) and G8 hashing were missing. Checked [ran]: on Huginn D2
  item `D2|op1|36 + 11|66 + 11`, G at c32.0 / c32.1 / c31.3 is -2.47 / -2.37 / -2.79, a = 0.25 gives e -0.020 /
  -0.023 / -0.025 and a = 1 gives +0.142 / +0.134 / +0.133; c32.2 (latest site) has G -0.20 and e(1) +0.42. Code:
  `grad_gain`, `run_e8` (PIG, `Ga`, `pig_a1_equals_B`, PIG top-3 verify rows), `summarize_pig`, PG4 in predictions;
  `pre_run_hashes`, `run_config`, `--hash_inputs`, `inputs_<tag>.json`, `meta.pre_run`, `--plan` required for a scored
  run. Deviation from the finding's mechanism: PIG uses the input path (source-span token embeddings at lerp(A, B, a),
  every site follows), not "leaves at lerp(H_A, H_B, a)" per site, for the two reasons in 7 E8 (cost 4 x |S| passes;
  a joint per-site 'set' blocks the gradient). The "Huginn E8 selector" open issue is rewritten (O-4): the debug data
  already show the a = 1 sign reversal at the last-loop sites, and the FD check (a <= 1e-2) cannot detect it.
  Not done here: writing the hashes into LEDGER.md, which this brief does not own (hard rule); `--hash_inputs` prints
  the line and the coordinator writes it before launch (P14). Tests: selftest `pig_summary`; selftest_model `pig`
  (a = 0 equals G bit for bit, a = 1 equals clean B, top-3 verified).
- F2 (major) fixed. Counts and GPU-hours recomputed from the Selector over the candidate files times the debug yields
  (13.2; `--count_caps`); D2 is capped at 65 (section 8). `--full_search_cap` and `--pc_site_n` added. Item-index
  sharding inside one output directory: `--stage scan` (one global selection pass per (group, dataset) plus the E8
  pilot), `--stage rows --shard k/n`, `--stage merge`; the full-search set is decided from the global kept list
  (`search_plan`), not from a running counter, so shards and the sequential run choose the same items; rows and merge
  refuse an incomplete scan or a changed search plan; load_results ignores shard files. One `--out` for every group
  (ouro_think reads McLeish's kept list there and warns if it is missing). Test: selftest `overrides`, `e8_gate`;
  selftest_model `driver` (sequential vs scan + 2 shards + merge: same kept list, full-search set, gate and rows bit for
  bit).
- F3 (major) fixed. `arch_of('huginn8')` = (raven, 8, 4); summarize skips E1/E5 cells for the PB8-only runner (it
  also raised KeyError on IN1_J[8] after the arch fix). Test: selftest `arch_ok`; selftest_model writes
  res_huginn8_D2.json into the summarize directory.
- F4 (major) fixed by a rule fixed now, not by a scope choice: the raven pilot gate (7 E8). Measured [ran, verified
  rows of the debug items, a = 1, eligibility over S]: McLeish D1 2/2, D2 1/2, D4 0/2; Huginn D1 2/2, D2 2/2; NR 6/6.
  At 0/2 the McLeish D4 search is expected to stop after its 10-item pilot (the E8 cell is then descriptive) and save
  about 30 shared machine-hours in S1; whether it stops is decided by the pilot, not by 2 debug items. For DV11: McLeish D4 E8
  is expected descriptive; McLeish D2 E8 is at risk (49 search items at most, so it needs >= 41% eligibility).
- F5 (minor) fixed. The Selector caps both content keys of D4 at runtime (`Selector.units`); bound 1,036 instead of
  1,351. The candidate files are unchanged (no rebuild, the sha256 values hold). Test: selftest `sel_d4`.
- F6 (minor) fixed. Rank agreement over above-floor sites plus top-1 / top-3 agreement (`fd_rank_agreement`); FD files
  keep G to 7 significant digits; PF rewritten. Recomputed on the first debug FD file [ran]: McLeish above-floor
  Spearman 0.993-0.9995 (38-60 sites of 191), top-1 and top-3 equal on 4 of 4 items; NR 0.996-1.0. The v3 build note
  "McLeish bf16 G: min 0.79, rank noise not negligible" is withdrawn; E8 stays in bf16. Test: selftest `rank_agree`.
- F7 (minor) fixed. The a = 0.25 noise is measured directly: fp32 and bf16 rows at a = 0.25 at the class sites on the
  FD items (`a025_rows`, `a025_noise`); the class gate and PT use it; the tiny row is a diagnostic. Huginn K = 16 class
  sites are not measured and their a = 0.25 classes are "not scored". Test: selftest `a025_noise`.
- F8 (minor) fixed. VB runs IN-persist(K-1) (`vb_inp`), one row per VB item; the E5 rule reads "not testable
  (IN-persist(j) not run)" if a persistent row is missing. Test: selftest `vb_inp_ok`.
- F9 (minor) fixed. `seed_sensitivity` re-runs the offset-0 scanned pair_ids with no Selector and compares statuses item
  by item (`status_changed`, `changes`, `sym_diff`); PB7 adds the item-level bound. Budget: about 0.5 shared machine-hour in S1.
- F10 (minor) fixed. `scan` reports a deadline stop; res meta `stopped_budget` and `scan_complete` are set per run;
  summarize flags "short of target: stopped by budget" and lists `datasets_short_by_budget`. Test: selftest
  `scan_stop`; selftest_model `driver.deadline_flag`.
- F11 (minor) fixed. Run log 13.1 corrected for Huginn IN-persist(j) (0.67-0.97 at j = 31, 0.86-1.02 at j = 30)
  [ran on res_huginn_D1/D2.json]; the unpulled VB and Huginn D4 files are named. Pulling them waits for the network
  (VPN down at the time of this review round).
- F12 (minor) fixed. `--descriptive` allows targets below 40 (recorded in the pre-run config and reported as
  descriptive); `--full_search_cap` covers cut rule (8). The DV11 scope choice stays with the coordinator (14).
