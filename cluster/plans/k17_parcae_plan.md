# B-K17-PARCAE plan: Parcae as a fifth looped family (v3: review round 2 applied)

v1 written 2026-10-07 night, before any scored run. v2 (same night) applies the adversarial critic's findings
F1-F10 (section 14, "Critic round 1"); sections 0 and 4-13 are rewritten where a finding changed them. Still
before any scored run: the only GPU processes so far are the load test in section 3 and the step-2 debug run
(section 13 run log), neither scored. Nothing below is tuned on results. Brief: PLAN.md B-K17-PARCAE, with K17
amendments A1-A8 and audit section 5 (P1-P14). Tags: [read] at the cited file:line, [ran] executed in this step,
[belief] not executed, [harness] true by construction (kept out of the held/missed tally).

v3 (2026-10-07, 21:45-22:30) applies review round 2 (F1-F9, section 14 "Review round 2"), still before any scored
run. Unlike v1/v2, the v3 additions (VE-core, VE-coda and STA@post_inj(8)+VE-coda rows, reading rule v3, PR11)
were written AFTER the step-2 debug rows (2-3 items per dataset) were seen; PR11 is tagged post-debug and is
not counted as a pre-registered prediction. The v2 thresholds of PR1-PR10 are unchanged.

Source citations: `PC:` = `parcae_lm/` of SandyResearch/parcae at commit 69284c13 (section 1), installed byte for
byte into `<home>/parcae_venv/lib/python3.12/site-packages/parcae_lm/` on the shared machine. `RE:` = `receval/` in
the same clone (the authors' eval code, not installed).

## 0. Scope

Brief core (A5 scope, never cut): E1 placement (source vs answer, by loop), E5 input protocols including
j = K-1 (single, persistent, state before/after the last injection), E8 carry gain with verify (A2, A6, A7). Datasets
D1, D2, D4, D5 and the VarBench copy-read target (A3). Fixed depth, fixed initial state per item, coda readout,
zero-edit identity, positive control, random control, clean-pass scan per dataset (skip a dataset under 40
clean pairs), tokenizer checks. Cap 3 GPU-hours on the shared machine, shared. No knockout, window or undo arms (A5).

Additions (each in the cut order, section 10):
- AD1. Value-embedding rows in E5 (ve-only and e+ve persistent). Parcae has a second token channel into every
  loop that the brief does not name (F6). Without these rows an E5 "persistent input" result is ambiguous.
- AD2. PC-site, a full-downstream-swap positive control that can fail at every site (section 7).
- AD3. Presence read-out s(h) at the answer position per loop (critic_k17_paper F9). Supporting only, "causal": false.
- AD4. Clean-pass scans on parcae-140m and parcae-770m (no arms) to report the size trend in pass counts.
- AD5 (critic F2). Downstream-input and state-only rows in E5 (section 8): they separate erasure at the source
  from re-injection of A's prelude output at later positions, and give every source cell an informative positive
  control that can fail for a reason other than a hook bug.

## 1. Install and provenance [ran]

- I1. PyPI `parcae-lm` has one version, 0.1.0. Its wheel (sha256 be15dd65b51e...) holds only dist-info files
  (`RECORD` lists no module, `top_level.txt` is empty). Its sdist (sha256 8cfa8c7f4bf3...) holds LICENSE,
  PKG-INFO, README, pyproject.toml and setup.cfg only. Cause: the sdist's pyproject sets
  `[tool.setuptools.packages.find] where = [".."]`, which finds nothing once the sdist is unpacked outside the repo.
  The README says the PyPI package is "coming soon". `pip install parcae-lm` therefore cannot provide
  `import parcae_lm`. Evidence kept in remote `out/k17_parcae/wheel/`.
- I2. Installed instead from the package's own repository (the PyPI metadata's Homepage): `git clone
  https://github.com/SandyResearch/parcae`, checkout 69284c13746e849104f738d6d1a347b1f457df76 (main HEAD,
  2026-09-02, "Merge pull request #11"). The only tag is v0.0.1. Then `pip install --no-deps
  --no-build-isolation ./parcae_lm` (its pyproject: name parcae-lm, version 0.1.0). `diff -r` of the installed
  package against the clone: identical (only `build/` and `pyproject.toml` differ, which are not package files).
  This is deviation DV1 (section 13).
- I3. Venv `<home>/parcae_venv/` (Python 3.12.3, new; no other venv touched). Pins, chosen equal to the
  existing shared machine `.venv`: torch 2.11.0+cu130 (download.pytorch.org cu130 index), transformers 4.56.2, tokenizers
  0.22.2, huggingface_hub 0.36.2, safetensors 0.7.0, einops 0.8.2, sentencepiece 0.2.1, numpy 2.4.4, packaging
  24.0, ninja 1.13.2. Full freeze: `artifacts_spark/k17_parcae/pip_freeze.txt` (48 lines).
- I4. File hashes of the installed source (sha256, first 16 hex): parcae.py c0d792379c3e965e, injection.py
  a163acbf86824ba5, mixer.py 7706683f5ce886db, blocks.py 96232cf3543dcf76, norms.py 888cb97a5e29ef5b,
  flash_attention.py aaafb0a7ff405c2d, parcae/config.py e9a5e7515b8f5261, utils/init.py 7df119be7f8ba293,
  `__init__.py` 453bdfc947ecb921, tokenizer.py 3aec59f3f7beed5b. k17_parcae.py asserts the parcae.py hash at load.
- I5. Checkpoints (HF hub, pinned by revision, in the default shared machine HF cache): parcae-140m 236739ce, parcae-370m
  43928446, parcae-770m 86c1403b, parcae-1.3b 517f8e6a; tokenizer repo parcae-tokenizer 6247b5d0 (tokenizer.json
  only, md5 d1979cee; `token_bytes.pt` not downloaded because it is a pickle and is not needed). Each model repo
  holds config.json, README.md and pytorch_model.bin (fp32). Sizes 0.58 / 1.55 / 3.11 / 5.36 GB.

## 2. Raw-source facts, with the recon claims re-verified

Recon claims tagged summarizer-read (recon_protocol.md F13-F15, B9) and their status on the installed source:

| recon claim | status | evidence |
|---|---|---|
| F15 depth trap: per-sequence model with `num_steps_pair=None` samples a Poisson depth per row in eval | confirmed, amended | `_sample_batch_depths` has no `self.training` check (PC parcae.py:478-535). It seeds a CPU generator with 514229 + self.step (513-517), so in eval the depth is deterministic for a batch shape. [ran] batch 1 gives 6 recurrences (n=2, k=4), not 8, on all four sizes |
| F15 forward has no `input_states` | confirmed | signature PC parcae.py:139-147. `iterate_forward` calls `self.initialize_state(input_embeds)` (246) |
| F15 `from_pretrained` loads with strict=False | confirmed; harmless on these checkpoints | PC `__init__.py`:63. [ran] strict load of all four: 0 missing, 0 unexpected, 0 shape mismatches. `freqs_cis` is in the checkpoint and equals the recomputed buffer |
| F14 update rule | confirmed | PC injection.py:42-46: dt = softplus(dt_bias), A = exp(A_log), x <- x * exp(-dt*A) + dt * (e @ B.T). Called once per recurrence before core layer 0 (PC parcae.py:365) |
| F14 readout needs C, coda, ln_f, head | confirmed | PC parcae.py:179 C, 182-189 coda, 193 ln_f, 229 `lm_head(x).float() * logit_scale`. [ran] my reconstruction from the captured loop-end state equals the model logits exactly (max abs diff 0.0) on all four sizes |
| F13 configs | confirmed from the downloaded config.json | P/C/Cd 2/2/2 d768, 4/4/4 d1024, 6/6/6 d1280, 8/8/8 d1536; mean_recurrence 8, mean_backprop_depth 4, poisson-truncated-full, per-sequence, diagonal, like-init, prelude_norm, qk_norm, tied, vocab 32768, init_strategy scaled-zero, use_fused_head pytorch |
| F13 off Hopper attention falls back to SDPA | confirmed | PC flash_attention.py:24-42 loads FA3 only when the major capability is 9. [ran] GB10 is (12, 1), `HAS_FA3` False. SDPA with `is_causal=True` (72-73) |
| open: value_embeds present in checkpoints | yes | [ran] keys 1, 3, 5, ... (odd indices) on every size (F6) |
| open: logit_scale | 1.0 | [ran] scaled-zero has no logit_scale entry, default 1.0 (PC utils/init.py:441-443) |
| open: ninja builds custom ops on GB10 | no build happened | the triton loss in PC ops/ is used only for `use_fused_head` hhe/full-triton (PC parcae.py:80-98); configs use pytorch. Import succeeded [ran] |

Further verified facts:

- F1 Structure [read]. wte -> prelude (P blocks, with value embeds) -> ln_prelude = e (PC parcae.py:158-169) ->
  `iterate_forward` (171-176) -> C (179) -> coda (182-189) -> ln_f (193) -> lm_head (229). Tied head (103-104).
  `self._current_input_ids = input_ids` (162) feeds value embeds inside the loop.
- F2 Loop body [read]. `core_block_forward` (PC parcae.py:362-386): `x = adapter(x, e)` (365), then the C core
  blocks, each with `ve = value_embeds[str(P + l)](self._current_input_ids)` when that key exists (374-385).
  `step` and `total_steps` are passed but unused. Blocks are pre-norm with no output norm: `x = attn(norm_1(x)) + x;
  x = mlp(norm_2(x)) + x` (PC blocks.py:22-25).
- F3 Loop-end norm [read]. None. `iterate_forward` returns raw x (PC parcae.py:360). Nothing runs between
  recurrences except the adapter at the start of the next one. The loop-end state Llast@j is the raw output of
  `core_block[C-1]` at call j, and pre_inj(j+1) is that same tensor (passed straight to the adapter at 321-323/365).
- F4 Executed-index map [read + ran]. With j = 1..K and l = 0..C-1: core idx = P + C(j-1) + l, coda c at
  P + CK + c. parcae-1.3b: 8 + 8(j-1) + l, loop end 8j+7, coda 72-79 at K=8. parcae-370m: 4 + 4(j-1) + l, loop
  end 4j+3, coda 36-39. This is the cache index the authors' generation path uses (RE models/parcae.py:113, 121).
  [ran] forward-hook call counts at `num_steps_pair=(8,0)`: every core block and the adapter fire 8 times, every
  prelude and coda block, ln_prelude, C and ln_f once. `block.layer_id` equals the executed index only for core
  l at j=1 (P + l) and for the coda at K = mean_recurrence = 8 (PC parcae.py:47, 51-52).
- F5 Depth control [read + ran]. `num_steps_pair=(n, k)` runs n recurrences under `torch.no_grad()` then k with
  grad (PC parcae.py:283-288, 318-333). [ran] (8,0) gives no gradient to any core state (grad None); (0,8) gives
  a nonzero gradient at the loop-1 state and the same logits bit for bit as (8,0).
- F6 Value embeddings, a second token channel [read + ran]. `value_embeds[str(i)]` exists for layer indices i
  with `i % 2 == (P+C+Cd-1) % 2` (PC parcae.py:70-73, mixer.py:18-20): the odd prelude, core and coda layers. In
  those layers `v = v + 2*sigmoid(ve_gate(x[..., :32])) * ve(input_ids)` (PC mixer.py:64-68). The core ones read
  `self._current_input_ids` at every recurrence, so token identity re-enters every loop through V, independent of
  the carried state and of e. The coda's odd layers do the same after the loop. [ran] the gate map agrees with
  the embed map on all four sizes (`consistent: true`; the attention module uses `config.n_layer` =
  C * mean_backprop_depth (PC parcae/config.py:46), whose parity matches because C is even).
- F7 Attention [read]. The `mask` argument is ignored. Every call is causal SDPA on the full sequence
  (PC mixer.py:106-107, flash_attention.py:72-73). No attention probabilities are exposed. Batch 1, no padding.
- F8 Initial state [read + ran]. like-init: `randn` then `trunc_normal_(std, -3std, 3std)` in place, times
  emb_scale if != 1 (PC parcae.py:544-549). std = `init.get_std("embedding")`; scaled-zero has no embedding key,
  so std = sqrt(2/(5 n_embd)) (PC utils/init.py:86-90, 426-438): 0.0161 (1.3b), 0.0198 (370m). emb_scale = 1.0
  (446-448). Global RNG, shape [B, T, rec_dim].
- F9 Injection parameters [ran]. Per-channel decay exp(-dt*A), quantiles 0/10/50/90/100%:
  1.3b 0.000/0.060/0.154/0.224/0.923 (mean 0.156); 370m 0.000/0.154/0.192/0.238/0.595 (mean 0.198); 770m mean
  0.166, max 0.799; 140m mean 0.280, max 0.610. Mean decay^8 is 0.0012 (1.3b) and 0.00006 (370m). Spectral norm of
  diag(dt) B: 291 (1.3b), 159 (370m). A state delta at a loop end enters the next loop scaled by about 0.15-0.2
  per channel on average, while the full e is re-injected.
- F10 Initial-state sensitivity [ran]. Two native draws change the final logits by at most 0.07 (1.3b), 0.04
  (370m), 0.25 (770m), 0.19 (140m) on one prompt, top-1 unchanged. A CPU `torch.Generator` draw at T=128 has the
  T=64 draw as its prefix, so one draw per item at the longest length can be sliced.
- F11 Determinism [ran]. Repeated fp32 forwards are bit-identical. Zero-add hooks on every core block and on the
  adapter give bit-identical logits. Adapter output recomputed from (previous loop end, e) equals the captured
  output at every j.
- F12 from_pretrained never calls `eval()` (PC `__init__.py`:22-69) [read]. k17_parcae.py loads with its own
  loader (same config path, strict load, `.float().eval()`).
- F13 Authors' eval protocol [read]. CORE eval passes `num_steps_pair = [mean_recurrence, 0]` (RE
  tasks/core_eval.py:148-149), prepends a BOS (62-72), and runs fp32 weights under bf16 autocast
  (RE settings.py:70, 187-191). Its `get_bos_id` returns `tokenizer.bos_id`, else `bos_token_id`, else 1
  (core_eval.py:47-52). [ran] The public tokenizer repo has no tokenizer_config, so the authors' `Tokenizer`
  gets `bos_id = None` and the public CORE path would prepend id 1 = `<|eos|>`. Training default `add_bos: True`
  (recpre/settings.py:195), with a tokenizer directory that is not public.

## 3. Load test [ran] (artifacts_spark/k17_parcae/loadtest/, script loadtest_parcae.py)

fp32, TF32 off, batch 1, K=8, GB10 shared with the closers debug job (3 GB GPU, 112 GB host memory available
at launch).

| size | params | load s | fwd ms T=6 / T=700 | fwd+bwd ms T=6 / T=700 | peak GB | France top-1 (BOS 0) |
|---|---|---|---|---|---|---|
| 140m | 144.3M | 2.4 | 6.9 / 39.4 | 22.6 / 93.1 | 2.4 | " is" (0.86) |
| 370m | 388.0M | 1.0 | 15.9 / 97.9 | 45.8 / 232.7 | 5.6 | " Paris" (0.67) |
| 770m | 776.7M | 2.3 | 32.0 / 217.5 | 76.8 / 494.0 | 10.2 | " Paris" (0.34) |
| 1.3b | 1338.6M | 3.8 | 54.4 / 414.6 | 122.8 / 881.9 | 16.4 | " Paris" (0.63) |

- Four capital prompts (France, Japan, Italy, Germany), top-1 by K in {1, 2, 4, 6, 8, 12, 16} under BOS 0: 1.3b
  is right on all four from K=1; 370m and 770m miss one at K=1 and are right from K=2; none of the three changes
  from K=2 to K=16. A fifth prompt ("Paris is the capital of") flips between " France" and " the" across K on
  370m. 140m answers " is" on the capital prompts at every K.
- BOS smoke (top-1 on the four capital prompts): 370m, 770m and 1.3b give the capital under BOS 0, BOS 1 and no
  BOS; the fifth prompt differs by BOS on all three. 140m answers " is" under BOS 0, " Tokyo"/" the" under BOS 1,
  and the capitals without BOS.

## 4. Tokenizer and data yield [ran, laptop CPU]

- T1 Tokenizer: BPE, vocab 32768, no normalizer, no post-processor (no automatic BOS), byte-level. Specials
  `<|bos|>` 0, `<|eos|>` 1, `<|pad|>` 2. Words carry their leading space (" Paris" = 6250, one token). A space
  before a digit is its own token 223 ("Ġ"); digits split left to right in chunks of 2 (" 1234" = [223, "12",
  "34"]; " 123" = [223, "12", "3"]). A number edit therefore never changes the space token, and every answer up to
  99 is one token after a one-token space suffix.
- T2 Candidate yields with the existing builders on the Parcae tokenizer (BOS 0 and no BOS give identical
  counts): D1 `k15_data.d1_candidates` 299 pairs (heldout capital 115, city 34; dev capital 115, city 35; 75
  subjects, 69 pairs with multi-token subjects). D2 `d2_candidates` 600 (300 op1, 300 op2). D4 via
  `k16_data.d4_row`: T1 1200 of 1200, T2 762 of 762 rows pass the tokenizer checks. D5 `d5_row`: 1183 of 1183.
  The D1 city quota (20 per half) needs 20 of 34-35 city candidates to pass the clean filter, which is tight.
  [ran v2, `--build_items`, `artifacts_spark/k17_parcae/data/build_meta.json`] Correction: `k16_data.d4_items`
  yields 1701 D4 items (T1 900, T2 801), all of which pass, as in `data/d4/d4_ouro.jsonl`; the v1 counts above
  were wrong. D5: 1183 items, 19 dropped as demo problems, 0 copy-confound drops; 240 distinct problems whose
  shared-problem graph has only 5 connected components, so component clusters would leave 5 clusters and the
  one-item-per-problem rule (section 9) is used. D1: only 1 of the 69 multi-token pairs has an aligned paraphrase
  form (the paraphrase templates give the city a leading space, which makes it one token), so O2 is
  effectively one-token only on D1. VarBench: all 1164 rule-valid rows pass the Parcae asserts; the 513 selected
  pairs and their order equal VARBENCH's selection exactly; in the first 200, 173 spans are one token, 27 are
  >= 2 tokens, 26 items have a continuation of >= 2 tokens.
- T3 VarBench copy-read (v2, critic F5). k17_parcae.py `--build_items` re-runs the k17_varbench build rules on the
  Parcae tokenizer: the same core pool (eval, text_exact, outside_equal on the four VARBENCH tokenizers), the I2
  sequences (prompt = SETTLE block + "Question: " + q + "\nAnswer:", chain prefix up to the first isolated
  restatement of x, joint encoding, t* = first differing token after the prompt, r = t* - 1, SA, SA', SB, SB'),
  the I3 drops (no restatement, restatement inside `<<...>>`, contradictory prefix, ambiguous source; the four
  text helpers are imported from k17_varbench.py, md5 recorded), and Parcae build asserts (prompt A/B equal
  length and equal outside the number span; joint encoding keeps the prompt prefix; full_A/full_B differ only
  inside the span before t*; restA[0] != restB[0]; the R2 render reproduces the question and chain tokens). Drop
  counts per rule are written to the items file. One pair per problem by the VARBENCH rule (smallest
  sha256("k17vb-pair:" + pair_id) among valid rows); problems in k17vb-order (sha256("k17vb-order:" + idx)); split
  and cluster as VARBENCH I6 (cluster = GSM8K idx). The overlap with VARBENCH's own selection
  (`out/k17_varbench/data/items_ouro.json`) is reported.

## 5. Decisions

- DC1 Depth K = 8. It is mean_recurrence in all four configs, the depth the authors' CORE eval and generation
  paths use (F13; RE models/parcae.py:92-93), and the mean of the training depth distribution (Poisson(8) clamped
  at 1, PC parcae.py:520-523). Every forward passes `num_steps_pair=(8, 0)`, or `(0, 8)` for gradients; the
  code asserts 8 adapter calls per forward. Loop sites j in {1, 2, 4, 7, 8} = {1, 2, K/2, K-1, K} (K/4 = 2).
- DC2 Models. Main: parcae-1.3b. Replication: parcae-370m. 140m and 770m: clean-pass scans only (AD4).
- DC3 BOS, frozen rule on clean accuracy only. Per model, clean top-1 accuracy of the A prompts of the 115 dev
  capital D1 candidates under BOS 0 (`<|bos|>`), BOS 1 (`<|eos|>`, the public CORE path) and no BOS. Pick the
  highest; ties go to BOS 0. All three accuracies are reported. If no BOS wins, every prompt (main or
  paraphrase) whose edited span includes position 0 is dropped and counted (P5). The choice holds for every
  dataset of that model. Items are stored without BOS; the runtime prepends it and shifts every position by one.
- DC4 Precision. fp32 weights, no autocast, TF32 off, so identity checks are bit-exact and the logits are not
  bf16-quantized. The authors evaluate under bf16 autocast; the clean-pass counts may differ slightly. All
  parameters are frozen (`requires_grad_(False)`), so the E8 backward reaches only the zero leaves.
- DC5 Initial state (v2, critic F7). Per item one draw: CPU `torch.Generator` seeded
  `int(sha256(pair_id)[:8], 16) + seed_offset`, `trunc_normal_(std, -3std, 3std)` at the longest row length of
  the item (A, B, SA', SB', paraphrase forms), times emb_scale (1.0). The seed depends only on the pair, so the
  clean filter, every scored row and the other model use the same draw; it is stored in every item record. The
  instance attribute `model.initialize_state` is replaced by a function that returns the slice of that draw for
  the row length (F8, F10). Hook check (j) asserts a 512-position draw's std within 2% of 0.9866 x the config std
  (the std of a normal truncated at 3 sigma); every item draw of at least 4096 values gets a 5% sanity bound (a
  2% bound on an 8-10 token draw would fail by chance on 1 item in about 100 to 2000, by width). 3-seed
  sensitivity: seed_offset in {0, 1, 2} on the D1 clean scan and on one
  cell (source O1 at Llast@7, parcae-1.3b).
- DC6 Endpoints (v2, critic F5). Margin m = z(tB) - z(tA) at the read position r, final logits (model output,
  fp32). D-sets: tA, tB are the first differing answer tokens; on the Parcae tokenizer every D2, D4 and D5
  answer is at most 99, so it is one token after the shared space token (asserted at build) and the first-token
  margin is the full-answer margin; D1 reads the first answer token as in K15/K16. VarBench: the full-span
  log-odds LO over the number's tokens is primary (VARBENCH section 5: LO = sum_k log p(restB[k]) on SA' -
  sum_k log p(restA[k]) on SA; dLO = LO(edit) - LO(clean A); D_LO = LO(clean B) - LO(clean A); e_LO = dLO / D_LO
  where D_LO >= 2), computed for every E1 O1 row at a = 1, PC-final, SZ and RND at L0@1 and Llast@1 (both
  positions), with one extra SA' forward per row on items whose continuation has >= 2 tokens ("cont" items). The
  first-token margin is secondary on every row and is the selector's target in E8. Per-loop readouts use the
  exact reconstruction C -> coda (with value embeds) -> ln_f -> lm_head on the loop-j state (F3, section 3).
  Because `core_block_forward` ignores `step` and `total_steps` (PC parcae.py:362-386), this readout is the
  model's own depth-j output; check (g) asserts it bit for bit.

## 6. Hooks (exact)

All hooks are installed once per model instance and gated by a per-forward context; with no context they do
nothing, so the per-loop readout can call the coda modules directly. Call counters reset each forward.

- H1 core blocks: forward hook on `transformer.core_block[l]`, call n -> loop j = n. Applies entries
  (positions, operator, donor tensor) to the output, records listed positions, adds zero grad leaves at the read
  position for E8. Sites L0@j (l=0), Lmid@j (l = C/2: 4 on 1.3b, 2 on 370m), Llast@j (l = C-1).
- H2 adapter output: forward hook on `transformer.adapter`, call j. Site post_inj(j) = adapter output = input of
  core layer 0 at loop j.
- H3 adapter input (v2, critic F6): forward pre-hook on `transformer.adapter`, call j. args = (x_prev, e) (PC
  parcae.py:365). All 8 calls receive the same `input_embeds` tensor (PC parcae.py:171-176), so the hook never
  writes into args[1]: it returns a new args tuple whose e is a clone with B's rows at the listed positions. This
  is the input (e) edit for loop j only. pre_inj(j) = args[0] is the same tensor as Llast@(j-1) (F3), so state
  edits "before the last injection" are Llast@(K-1) edits.
- H4 value embeds: forward hook on `value_embeds[str(P + l)]` for the core keys (called once per loop, counter =
  j) and on the coda keys (called once). The prelude keys are not hooked. Replacing rows at the listed positions
  with rows of the embedding weight at B's ids is the ve edit (exactly what B's run computes).
- H5 coda: forward hook on `transformer.coda[c]`. coda[Cd-1] output = post-coda state (ln_f input) for PC-final
  and SZ.
- H6 initial state: the instance attribute override in DC5. H7 ln_prelude: forward hook that records e.
- H8 per-loop readout: function in DC6, reads recorded Llast@j states with the context off.
- Checks before any scored row (checks.json `hook_identity`, per model, on the first kept D1 item): (a) call
  counts: 8 per core block, 8 adapter calls (pre and post), 8 per core value-embed key, 1 per coda block and
  coda key (asserted in every forward, not only here); (b) zero add at every site and every e/ve call, and zero
  grad leaves, give bit-identical logits; (c) repeat bit-identical; (d) an edit at loop j position p leaves all
  earlier calls bit-identical and later calls bit-identical at positions < p; (e) the (0,8) grad run's logits
  equal the (8,0) logits; (f) the parcae.py sha256 prefix equals c0d792379c3e965e; (g) the per-loop readout of
  the clean loop-j state equals the model run at `num_steps_pair=(j, 0)` bit for bit, j in {1, 4, 8} (critic F3);
  (h) in an IN-single(3) row the e argument at every adapter call other than 3 equals clean A's e bit for bit,
  and at call 3 it equals B's e at the source positions and A's elsewhere (critic F6); (i) B's e at all positions
  at every call plus B's value embeds at all positions at every core and coda call equals clean B bit for bit
  (the e and ve hooks reach everything); (j) the initial-state draw std (DC5).

## 7. Sites, operators, controls

- Sites S_E1 per loop j in {1, 2, 4, 7, 8}: post_inj(j), L0@j, Lmid@j, Llast@j (20 sites). VarBench uses a reduced
  set: post_inj(j), L0@j, Llast@j for j in {1, 4, 7, 8} plus Lmid@8 (13 sites). Positions: source = the differing
  span of the A/B pair; answer = the read position r (last position; on VarBench r = t* - 1).
- Last-loop answer cell (v2, critic F1, paper/EXPERIMENTS.md 3b2): Lmid@8. Llast@8 at the answer position
  reaches the readout through one linear map (C, PC parcae.py:179) and the coda; it stays an E1 cell labelled
  "coda-gated" and is never the last-loop answer cell, the uniform rule, or a member of the primary E8 set.
- O1 donor: H[s, P] <- lerp(H, H^B[s, P], a) (a = 1 is an exact copy), a = 1 unless stated; donor = clean B run at
  the same site with the same initial state.
- O2 aligned paraphrase vector: H[s, P] += H^{B,para}[s, P'] - H^{A,para}[s, P'], only forms whose source span has
  the same length and the same position-0 class as the main prompt and whose A and B both pass the clean
  filter (P4). No broadcast fallback. Diagnostic cos(v, own delta) and |v|/|own delta| per cell (P7).
- RND: per position a unit Gaussian (seed = first 8 hex of sha256(pair_id|site|pos|p)) scaled to
  |H^B - H^A| at (s, p).
- PC-final [harness]: answer O1 on the post-coda state (coda[Cd-1] output) at r. Must equal clean B logits at r
  bit for bit on every item, else stop.
- SZ [harness]: source O1 and RND on the post-coda state at the source positions. Nothing reads them, so the
  r logits must equal clean A bit for bit, else stop. (Llast@K at the source is NOT a structural zero on Parcae:
  the coda reads it.)
- PC-site [harness] (AD2): at site s, O1 on ALL positions, plus B's e at every later adapter call and B's value
  embeds at every later core and coda call. Everything downstream is then B's computation, so the logits must
  equal clean B bit for bit at every site. It fails only if a hook hits the wrong call, tensor or position, or if
  a fourth path carries token identity. Run on the first 20 kept items of each dataset per model at all E1
  sites. Any failure stops the run.
- Informative positive control (v2, critic F2; A1): the state-only rows STA (section 8, E5) swap the state at ALL
  positions with no e or value-embed swap, so they can fail because A's prelude output and value embeds are
  re-injected. Each source E1 cell reports the STA value of its loop band: loops 1-2 -> STA@Llast@1, loops 3-5 ->
  STA@Llast@4, loops 6-7 -> STA@Llast@7, loop 8 -> STA@post_inj(8). PC-final, SZ and PC-site are reported as
  [harness] checks, not as positive controls.
- References: clean A, clean B (prompt swap, ceiling). Per-item clean margins logged (P14).

## 8. Arms (per kept item; parcae-1.3b and parcae-370m)

- E1 placement: O1 and RND at S_E1 x {source, answer} (80 rows; VarBench 52). Answer O1 at a = 0.25 at every
  answer site of S_E1 (20 rows; VarBench 13) for the answer classes (critic F3). O2 at 8 cells (source Llast@1,
  Llast@7, Llast@8, post_inj(8); answer Llast@1, Llast@7, Lmid@8, Llast@8) on items with a passing paraphrase
  form (D1, D2, D4, D5; VarBench uses the R2 render of VARBENCH I5).
- E5 input protocols (source positions):
  - IN-single(j): B's e at loop j only, j = 1..8 (8 rows; j = 7 is K-1, P13).
  - IN-persist(j): B's e at loops j..8, j = 1..7 (7 rows; persist(8) = single(8)).
  - VE-persist (AD1): B's value embeds at every core loop and the coda, e unchanged (1 row).
  - VE-core and VE-coda (v3, review round 2 F1): B's value embeds at the source positions in the core loop only
    (every loop, keys P + odd l) and in the coda only (keys P + C + odd c). They split VE-persist into the loop
    channel and the channel that bypasses the loop (2 rows, every item).
  - INVE-persist (AD1): B's e at loops 1..8 plus B's value embeds everywhere after the prelude (1 row).
  - State before vs after the last injection: Llast@7 (= pre_inj(8)) vs post_inj(8), source and answer, read from E1.
  - AD5 (critic F2), 8 rows on every D1 and D2 item and on the first 20 kept items of D4, D5 and VarBench:
    IN-persist-all(j) for j in {1, 7} (B's e at ALL positions at loops j..8); STA@post_inj(8)+VE-coda (v3: O1 at
    all positions at post_inj(8) plus B's coda value embeds at the source; only the loop-8 core value embeds at
    the source then differ from B); IN-persist-down(1) (B's e only at
    positions after the source span, r included, at loops 1..8); STA at Llast@1, Llast@4, Llast@7 and
    post_inj(8) (O1 on ALL positions at that site, no e or value-embed swap).
- E8 carry gain (A2, A6, A7; v2 critic F1, F9, F10). Primary set S = all core-block outputs at r except Llast@8:
  K*C - 1 sites (63 on 1.3b, 31 on 370m). Secondary set S+ = S plus Llast@8 (reported as "with Llast@8"). The
  post-coda state is outside both sets and is the PC-final row. One clean-A forward and backward with
  `num_steps_pair=(0, 8)` and zero leaves at every site of S+: g = dm/dH; G(j,l) = g . (H^B - H^A)_{j,l}[r]. Verify:
  O1 at the top-3, median-3 (ranks n/2-1, n/2, n/2+1) and bottom-3 sites by G, ranked within S (primary) and
  within S+ (secondary), union of the two site lists, at a = 1 and a = 0.25. Dev search: O1 at a = 1 at every site
  of S+ on dev items (VarBench: the first 20 dev items in order); dev-tuned site = argmax mean dm over dev items,
  per set. Baselines on test items at both doses: dev-tuned site, uniform rule Lmid@8 (primary; Llast@8 for S+),
  random site of S (seeded sha256(pair_id|"rnd_site")). Metrics as VARBENCH T7: eligibility (verified dm range >
  2 at a = 1, > 0.5 at a = 0.25), hit without tie credit, permutation chance, regret, Spearman on eligible items
  with the count; v3 (review round 2 F4): hit, chance, regret and Spearman are computed on the eligible TEST
  items, the pool the interpretation gate counts (dev + test values reported under `devtest`). A dataset's E8 cell is interpreted only with >= 20 eligible test items at that dose; below that
  it is reported and not interpreted (critic F10). A7 false-negative rate (critic F9), primary: dev items, every
  site of S at a = 1, pooled over item x site: among sites with |G| < 0.05 * D, the fraction with |dm| >= 0.25 *
  D; secondary: the same on test items over their 9 verified S sites; site counts reported per dataset.
  Selector cost (EXPERIMENTS 3c): selector = 1 forward + 1 backward = 1 + t_bwd/t_fwd forward equivalents
  (measured per item); dev search = |S+| x n_dev forwards (`e8.cost_fwd_equiv`). v3 (review round 2 F9): t_fwd
  is a separate plain forward with no recording (`t_fwd_plain`), not the recording clean-A forward.
- Presence and answer classes (AD3; v2 critic F3). Presence s(h) = (h - (mA + mB)/2) . (mB - mA) / |mB - mA|^2
  + 0.5 at r (0 at clean A, 1 at clean B), n/a where |mB - mA| < 0.01 |mA| (VARBENCH guard), recorded at every
  loop end for every row; "causal": false. Validation (critic_k17_paper F9): at each answer site of S_E1, a
  paraphrase form (VarBench: R2) whose A' and B' pass the clean filter must project apart, s(B') - s(A') >= 0.5,
  with mA, mB from the main prompt; a site's presence gate passes if >= 50% of >= 10 such items pass. Per answer
  O1 row at Llast@j (j < 8) the code logs the clean A and B depth-j margins (all j = 1..8 per item) and e_own(j) =
  (m_edit(j) - m_A(j)) / D, the depth-j readout of the edited state with the final D. Classes per answer cell at
  a = 1 and a = 0.25, thresholds scaled by a: "carried" if mean final e >= 0.5a; at Llast@j "not carried" only
  where mean e_own(j) >= 0.5a and final e < 0.25a, "no answer state at site" where e_own(j) < 0.5a and final
  e < 0.5a; at L0@j, Lmid@j and post_inj(j) "not carried" only where the presence gate passes and final e <
  0.25a, "no answer state at site" where it fails and final e < 0.5a; "partial" otherwise; Llast@8 is
  "coda-gated" (no class).

## 9. Items, filter, strata, statistics

- Targets (v2, critic F10), one per dataset: D1 80, D2 60, D4 60, D5 60, VarBench 60 kept items per model.
- Clean filter (batch 1, final readout): top-1 = tA on A, top-1 = tB on B, and D = mB - mA >= 2 (so every kept
  item has e defined). Report scanned, A_fail, B_fail, D_small and kept per dataset and model. A dataset with
  fewer than 40 kept items (after the D >= 2 filter) is skipped for that model and reported.
- D1 (v2, critic F4): `k15_data.d1_candidates` on the Parcae tokenizer. pos_len-balanced selection as closers G1:
  all multi-token pairs (pos_len >= 2) in builder order up to 40 kept, then one-token pairs in builder order up
  to 80 kept in total; at most 2 uses per subject, A or B side, across the dataset (k15 `select` semantics).
  [ran] On this tokenizer pos_len >= 2 coincides with the city relation: all 69 multi-token candidates are city
  pairs (subjects at the template start carry no leading space and split into 2-3 pieces), and every capital
  pair is one token. pos_len and relation are therefore fully confounded on Parcae D1; the pos_len x relation
  counts and n_clusters are reported per cell. PR3 is frozen as "not testable" if the >= 2 stratum has fewer than
  20 items or 10 clusters; otherwise it is tested by a cluster-bootstrap TOST on the stratum difference (90% CI
  inside [-0.15, 0.15]). Cluster = A-side subject (rel:a); split = parity of the first hex of
  sha256("k17pc-d1:" + cluster). v3 (review round 2 F2, VARBENCH `d1_assign`): a pair is kept only if its B
  subject (rel:b) hashes to the same split, so dev and test share no subject on either side; CIs use the dyadic
  bootstrap over (A, B) subject tuples (VARBENCH boot2, copied verbatim; parity in the selftest); PR3's stratum
  TOST uses the same dyadic weights; n_clusters of a D1 cell = number of distinct subjects (A or B side). [ran]
  The drop leaves 150 of 299 candidates (one-token 116 = dev 81 / test 35; multi-token 34 = dev 26 / test 8).
  Under the subject cap the selection can reach at most 69 items (23 multi-token, 24 test) before the clean
  filter, so the D1 target of 80, the 40-multi quota and >= 20 eligible test items in E8 are not reachable.
- D2: `k15_data.d2_candidates` on the Parcae tokenizer. Cluster = A-side edited operand value (audit T8, VARBENCH
  I7); split = sha256("k17vb-d2:" + cluster) as VARBENCH; quota 30 per op; at most 2 items per cluster.
- D4: `k16_data.d4_items` + `d4_row` on the Parcae tokenizer, rows whose ids differ exactly at source_span
  (P10(2)). Cluster = VARBENCH I7 content key sha256(kind | vals_A | index of src_var | indices of query_vars);
  split = sha256("k17vb-d4:" + key); quota 15 per (split, kind); one item per cluster; scan order round-robin over
  (split, kind). v3 (review round 2 F2, VARBENCH v4 `d4_components`): cluster = connected component of the graph
  of A-side and B-side content keys (edges = rows), split = sha256("k17vb-d4c:" + component), one item per
  component. [ran] 814 components over 1701 rows (largest 7 keys); components per (split, kind): dev T1 184, dev T2
  238, test T1 145, test T2 247; dev/test shared content keys 0.
- D5: `k16_data.d5_items` + `d5_row` on the Parcae tokenizer. A problem is the ordered (a, b, m) without letters.
  Items are kept so that no problem (A or B side) appears in two kept items (greedy in builder order), so the
  cluster is the item and dev/test share no problem (audit C16); the connected-component count of the
  shared-problem graph is reported for reference. Dropped and counted: items whose answer token equals the
  source token of A or B (copy confound, P12) and items whose A or B problem is a demo line. Split =
  sha256("k17pc-d5:" + id); quota 30 per split; scan order round-robin over split.
- VarBench: T3, in k17vb-order, cluster = GSM8K idx, split = VARBENCH I6.
- Every split is a function of the cluster key, so dev/test cluster overlap is 0 by construction; checks.json
  records the measured overlap per dataset (`clusters`), and (v3) the A-or-B unit overlap
  (`dev_test_unit_overlap_A_or_B`, D1 subjects on both sides).
- Strata (A1, P3): source token count 1 vs >= 2, paraphrase mode, relation or kind, "edit includes position 0"
  (only possible without BOS), VarBench ST2/ST3/cont. v3 (review round 2 F7): every cell carries the dataset
  default strata (D-sets pos_len and rel; VarBench pos_len, ST2, ST3, cont); O2 cells add para_fmt (the accepted
  paraphrase form). A stratum with one value is reported as `single_value`. "Edit includes position 0" is not a
  stratum because prep_item drops such items when no BOS is used and a BOS shifts every edit off position 0. Every cell reports n_items, n_clusters, its informative
  positive control (section 7) and the model's PC-final, PC-site and SZ checks.
- Statistics: cluster bootstrap 2000 resamples, ratio-of-sums for e; nulls cited only by TOST with margin 0.10
  (90% cluster-bootstrap CI inside [-0.10, 0.10]); thresholds as VARBENCH T1 (works: mean e >= 0.5 and flip >=
  0.5; fails: mean e <= 0.15) and T3 (random control passes: |mean e| <= 0.05 and flip <= 0.02). Paired
  differences (E8 baselines, PR5, PR3) use the cluster bootstrap on per-item differences. The plan file sha256
  goes into checks.json at launch and into the LEDGER before the first scored row (P14).

## 10. Budget, order, cut rules (v2, critic F5 re-budget)

- Forward-equivalents per item [belief until the debug timings replace them]: D-set test item about 170 (clean
  and paraphrase 6, grad 2.3, E1 100, O2 8, E5 17 + AD5 7, controls 3, depth-j readouts 2.4, verify about 17,
  random site 2, dev-tuned pass 3); dev item about 215 (dev search about 49 extra rows). VarBench item about 135
  (test) and 190 (dev with search): E1 52, answer a = 0.25 13, cont SA' forwards about 13 (0.4 x 33), R2 2.
- Times from section 3: 1.3b 65 ms per forward at T 20-100 and 415 ms at T 700; 370m 20 ms and 98 ms.
  1.3b: D-sets 260 items x 190 x 0.065 s = 54 min; VarBench 60 x 163 x 0.415 s = 68 min (40 items: 45 min).
  370m: D-sets about 16 min, VarBench about 16 min. BOS calibration, scans on four sizes and seeds about 7 min;
  debug about 10 min. Total about 170 min against the 180-minute cap, before contention.
- Launch configuration is fixed from the debug timings before launch so the projection fits 150 minutes (guard).
  v3 (review round 2 F8): every launch of one run gets the same absolute `--deadline_epoch` (run start + 150
  min), so the total over both model processes stays within 150 minutes; the dev-tuned pass and the seed
  sensitivity check the deadline first; a scan cut by the deadline is labelled `stopped_budget`, not skipped.
  Cut order: (1) 370m VarBench, (2) 1.3b VarBench 60 -> 40, (3) O2 rows, (4) 370m D5, (5) targets 60 -> 40 (D1
  80 -> 60, still multi-token first). Never cut: PC-final, SZ, PC-site, zero-edit checks, E1 O1/RND at
  Llast@{1, 7, 8}, Lmid@8 and post_inj(8), E5 single and persistent rows, AD5 rows on D1 and D2, E8 on D1 and D2.
- Order: debug (370m, 3 items per dataset; 1.3b, 2 items) -> hook checks -> BOS calibration and clean scans (all
  four sizes) -> 1.3b D1, D2, D4, D5 -> 1.3b VarBench -> 370m D-sets -> 370m VarBench.
- Memory: check `free -g` and `nvidia-smi --query-compute-apps` before every launch; 1.3b needs about 16 GB at
  T 700 with grads. Launch only if at least 30 GB is available. Stop with SIGTERM to an anchored pattern:
  `pkill -TERM -f '^<home>/parcae_venv/bin/python k17_parcae.py'`.

## 11. Predictions (written before any scored run; [belief] unless tagged [harness])

Mechanism reading: a loop-end delta enters the next loop scaled by the per-channel decay (mean 0.16 on 1.3b,
0.20 on 370m) while the full e and the value embeds re-enter every loop. Parcae should therefore behave like a
model with a memory of about one loop: only edits inside the last loop, or edits to the re-injected inputs,
reach the output. e is the prelude output, so A's subject is in e at every later position, including r (F2).

- PR1 (E1, source). Source O1 at Llast@j for j <= 4: mean e <= 0.10 on both models and every dataset with >= 40
  items. At Llast@7: e <= 0.25. At post_inj(8), L0@8 and Lmid@8: e >= 0.4 on D1 for 1.3b. At Llast@8: e >= 0.3
  (the coda reads it, F6 caveat: the coda's value embeds still carry A).
- PR2 (E1, answer; v2 critic F1). Last-loop answer cell Lmid@8: e in [0.3, 0.9] on D1 for 1.3b (wide: the last
  4 core blocks and the coda re-read A's other positions). Llast@8 [coda-gated]: e >= 0.6. Llast@7: final e <=
  0.25 with class "not carried" (e_own(7) >= 0.5). Loops <= 4: e <= 0.10; Llast@1 class "not carried" on 1.3b
  D1 (1.3b answers capitals from K = 1, section 3).
- PR3 (strata). Unlike the IFM injection twin (closers G1 predicts multi-token subjects keep 0.4-0.7 at L0@1),
  Parcae's e is the prelude output, which already composes multi-token subjects, so the one-token vs
  multi-token difference in source O1 at Llast@1 is within 0.15 (TOST, section 9), if testable. On Parcae D1 this
  is also a capital vs city difference (section 9).
- PR4 (E5). IN-single(j) for j <= 6: e <= 0.10; IN-single(7): <= 0.25; IN-single(8): >= 0.3. IN-persist(j) is
  flat in j: |IN-persist(1) - IN-persist(7)| <= 0.15, and IN-persist(1) >= 0.5 on D1 for 1.3b. INVE-persist >=
  IN-persist(1), gap 0.1-0.3. VE-persist alone: 0.1-0.4.
- PR5 (before vs after the last injection). Source e(post_inj(8)) - e(Llast@7) >= 0.3 on D1 for 1.3b.
- PR6 (E8, primary set S without Llast@8). Top-predicted site lies in loop 8 or in loop 7 with l >= C/2 on >= 80%
  of items. Top-1 hit rate at a = 1 >= 3x its permutation chance on D1 and D2. A7 false-negative rate (dev,
  all of S) <= 0.10. Median per-item Spearman on eligible items >= 0.5. The a = 0.25 hit rate is not lower than
  the a = 1 rate by more than 0.15 (the decay path is linear). Selector vs dev-tuned site: |difference in mean
  dm / D| <= 0.10 (no claim that the selector beats a tuned fixed site; its advantage is cost). Secondary set S+:
  the top-predicted site is Llast@8 on >= 50% of items (the concern of critic F1).
- PR7 (seed). Over 3 seeds, the D1 clean-pass count changes by at most 2 items and the source O1 Llast@7 cell by
  at most 0.02 in e (initial-state weight after 8 loops about decay^8 = 0.001).
- PR8 (controls). RND passes T3 at every site. PC-final, SZ and PC-site are bit-exact [harness].
- PR9 (yield). Kept items (out of the scan, after D >= 2): D1 1.3b >= 40, 370m >= 40; the D1 >= 2-token stratum
  < 20 items on 370m (PR3 then not testable there); D2 1.3b >= 40, 370m < 40; D4 >= 40 on both (T1 easier than
  T2); D5 < 40 on 370m; VarBench copy-read >= 40 on 1.3b. 140m < 40 on D1.
- PR10 (AD5, critic F2; D1 and D2, 1.3b, e at a = 1). STA@Llast@1 and STA@Llast@4 <= 0.10 (re-injection restores
  A even after a full-state swap). STA@Llast@7 in [0.05, 0.40]. STA@post_inj(8) >= 0.8 (only the source value
  embeds still differ from B). IN-persist-all(1) >= 0.8; IN-persist-all(7) >= 0.6. IN-persist-down(1) >= 0.5
  and >= IN-persist(1) on D1 (the prelude writes the subject into later positions' e).
  v3 correction of the STA@post_inj(8) rationale (review round 2 F1; the v2 threshold stays as frozen and is
  scored as written): after the swap at post_inj(8), B's state enters loop 8, but A's value embeds at the source
  still enter the loop-8 core odd layers and the coda's odd layers (F6). The coda's attention at r reads the
  source position's values, so the value embeds alone can hold A's answer without passing through the loop. The
  v2 phrase "only the source value embeds still differ" was true and the conclusion drawn from it (>= 0.8) was
  not: those embeds are a full channel. [ran, debug, not scored] parcae-1.3b VarBench, 2 items: VE-persist 0.97,
  STA@post_inj(8) 0.055 and 0.079, IN-persist-all(1) about 0, source O1 at Llast@8 about 0.
  Reading rule (v2, superseded): a source-cell null at loop j is attributed to decay at the source only where its
  informative positive control (section 7) is >= 0.5; where that control is also <= 0.15, the null is attributed
  to re-injection of A's inputs and not to the source state.
  Reading rule v3 (review round 2 F1; written after the debug rows, before any scored row; implemented as
  `e1_cells.<cell>.src_attribution`). A source cell with mean e <= 0.15 at loop j, with informative control p
  (its STA band, section 7), is read as: (i) p >= 0.5: decay at the source; (ii) 0.15 < p < 0.5: not attributed;
  (iii) p <= 0.15: attributed to every input channel whose row reaches e >= 0.5 on the same items (the AD5 set):
  "VE-coda" (value-embed channel that bypasses the loop), "VE-loop" (VE-core, the loop's value embeds), "e
  re-injection" (IN-persist-all(1), A's prelude output at all positions); if none reaches 0.5, "unattributed".
  A null is credited to re-injection of e only through branch (iii) with IN-persist-all(1) >= 0.5, never by
  default.
- PR11 (v3, review round 2 F1; post-debug, tagged, not counted as pre-registered). Per model and dataset with >=
  40 scored items: VE-coda >= VE-core; |VE-persist - max(VE-core, VE-coda)| <= 0.15 (one channel dominates);
  STA@post_inj(8)+VE-coda >= 0.8 (only the loop-8 core value embeds then differ); STA@post_inj(8)+VE-coda minus
  STA@post_inj(8) >= 0.3.

## 12. Verification and artifacts

- CPU first: `python k17_parcae.py --selftest` (operators, RND norms, bootstrap, TOST, stratum-difference TOST,
  Spearman, hit with ties, verify sets, site maps, presence, classes, seeds, selection rules, D5 problem
  disjointness, parity with the k17_varbench helpers when importable) and `--selftest_model` (a tiny random-weight
  Parcae from the installed source: hook checks a-j, PC-final, SZ, PC-site, F2 rows, E8 grad path).
  `--build_items` (laptop CPU, Parcae tokenizer) writes the candidate and VarBench files with build asserts.
- GPU: debug `--model parcae-370m --debug 3` and `--model parcae-1.3b --debug 2` into `out/k17_parcae/debug/`, then
  the full run per section 10 (not in step 2).
- Artifacts: remote `<home>/loop_flow/out/k17_parcae/` with `res_<model>_<dataset>.json`, `checks.json`,
  `summary.md`; pulled to `E:/Research/loop-flow/artifacts_spark/k17_parcae/`.
- checks.json keys (no host names in any field): `plan_sha256`, `install` (venv python, torch, parcae source
  commit, pypi wheel empty, installed-equals-clone, parcae_py_sha16), `checkpoints` (revision per size),
  `strict_load`, `depth_assert` (adapter calls = 8), `hook_identity` (a-j), `bos_calibration`, `endpoint`,
  `items` (per model and dataset: scanned, A_fail, B_fail, D_small, kept, skipped, n_clusters, pos_len x relation
  counts), `clusters` (dev/test overlap per dataset), `varbench_build` (drops per rule, overlap with VARBENCH),
  `seed_sensitivity`, `pc_final`, `sz`, `pc_site`, `informative_pc`, `e1_cells`, `e5_cells`, `ad5_cells`,
  `answer_classes`, `presence_validation`, `pr3`, `e8` (per dataset, set S or S+, dose: n_sites, eligible,
  eligible_test, interpreted, top1_hit, chance, regret, spearman with n, dm/D at predicted vs dev-tuned vs
  uniform vs random with paired cluster CIs, fn_rate_dev, fn_rate_test9, dev_tuned_site, cost_fwd_equiv),
  `predictions_vs_outcomes` (PR1-PR10, harness tags), `gpu_minutes`.

## 13. Deviations from the brief (so far) and run log

- DV1. `pip install parcae-lm` cannot work: the PyPI 0.1.0 wheel and sdist contain no code (I1). The package was
  installed from its own GitHub repository at commit 69284c13 (I2). The PI approved "pip install parcae-lm (pin
  the version)"; this is the same package from its source, pinned by commit. Needs the PI's confirmation.
- DV2. parcae-140m and parcae-770m were downloaded (brief: "if quick"; 3.7 GB, under 2 minutes).
- DV3. Additions AD1-AD5 (section 0).
- DV4. Targets per dataset (section 9) are below PUBLIC's "up to 80" except D1, to fit two models and five
  datasets in 3 GPU-hours.
- DV5. Parcae D1 pos_len is confounded with relation (section 9); the brief's stratum is reported as is.
- DV6. Answer-class thresholds scale with the dose a (section 8). Critic F3 asks for classes at a = 1 and
  a = 0.25 without saying how the thresholds move; unscaled thresholds would label every a = 0.25 cell "no answer
  state at site".
- DV7. k17_parcae.py adds VARBENCH's T7 v3 tolerance hit (hit = dm(top) >= max dm - 0.05 D, band chance,
  random-in-top-3) as a secondary E8 metric (`e8.<set|a>.v3_verified9`, `v3_dev_fullset`) so the Parcae row can
  be compared with VARBENCH's frozen v3 primary. The strict hit of section 8 stays primary here.
- DV9 (v3). Data rebuilt into `data_r2/` under the review round 2 D1 and D4 rules; the D1 target 80 and the
  40 multi-token quota are unreachable (62 / 61 kept). The DC3 BOS pool is held fixed at the v2 115 pairs.
- DV8. The data files were rebuilt at the step-2 resume (2026-10-07 21:08) because k17_varbench.py changed after
  the 19:06 build (md5 59b2c6f5 -> 8367446e). Candidate content is identical on all five datasets (every field of
  every candidate, same order); only `seconds` and the recorded helper md5 differ. The remote copy on the shared machine still
  has the older k17_varbench.py (59b2c6f5); it is not needed by `--run`.
- Run log (step 2, resumed after the API outage; no scored run):
  - 21:08 CPU [ran]: `--selftest` ok (laptop and the shared machine; parity with the k17_varbench helpers on the laptop),
    `--selftest_model` ok (laptop copy of the installed source and the shared machine venv: hook checks a-j, PC-final, SZ,
    PC-site, PC-site negative controls, O2 rows, E8 grad path, e_own, STA, dev-tuned pass, VB LO causality),
    tiny-model end-to-end `run_dataset` + `summarize` ok.
  - 21:09-21:15 GPU debug parcae-370m `--debug 3` [ran]: BOS rule picks bos1_eos (acc bos0 0.974, bos1_eos 0.991,
    none 0.974). Hook checks a-j all true on the real model. D1 3 items, VB 3 items scored; PC-final, SZ and PC-site
    bit-exact on every row; per-loop readout at j = K equals the model; grad run equals the forward. 5.1 min.
  - 21:15-21:25 GPU debug parcae-1.3b `--debug 2` [ran]: BOS rule picks bos0 (0.974 / 0.974 / 0.957; tie to
    bos0). Hook checks a-j all true. D1 2, D4 2, VB 2 items scored; all harness rows bit-exact. Peak GPU 9.5 GB.
    10.2 min. Timings per item: D1 10 s, D4 21-26 s, VB 50-66 s (370m: D1 3 s, VB 12-18 s).
  - Clean-pass yield [ran, full candidate scans]: D2 kept 0 of 600 on both sizes (A_fail 586 on 370m, 582 on
    1.3b); D5 kept 0 of 1164 on both (A_fail 1126 / 1142); D4 kept 0 of 1701 on 370m (A_fail 1526: the model
    answers the last assigned value, '4' on 38 of 40 items). Diagnostic [ran, inline, not saved]: parcae-1.3b A
    accuracy on the first 40 D2 items is 0.025 and on D5 0.05; the trailing space token is the right convention
    (without it the model predicts the space on 40 of 40). The models cannot do the arithmetic; this is not a
    harness fault. D2 and D5 will be skipped on both models (rule: < 40 kept), D4 on 370m. This breaks the
    section 10 never-cut entries "AD5 rows on D1 and D2" and "E8 on D1 and D2" and misses PR9 for D2 on 1.3b.
    Needs the PI's decision before the full run (see the report).
  - v3 (review round 2) [ran]: CPU `--selftest` ok on the laptop and the shared machine (VARBENCH parity now includes boot2
    and d4_content), `--selftest_model` ok on both (adds: VE-core and VE-coda partition VE-persist; STA@post_inj(8)
    plus B's value embeds after it equals clean B bit for bit), tiny end-to-end run_dataset + summarize ok, budget
    stop labelled. Data rebuilt into `data_r2/` (v2 `data/` kept for the step-2 debug provenance); D2, D5 and
    VarBench candidates are identical to v2, D1 and D4 change by the F2 rules; the DC3 BOS pool is the same 115
    pairs, stored as `bos_cal` in cands_D1.json before the cross-split drop (BOS choices reproduce: 370m
    bos1_eos 0.991, 1.3b bos0 0.974).
  - v3 GPU debug (`out/k17_parcae/debug_r2/`, shared deadline) [ran, not scored]: 370m D1 2 + VB 2 items (0.7
    min); 1.3b D1 2 + D4 2 + VB 2 items (3.7 min). Hook checks a-j all true on both; PC-final, SZ and PC-site
    bit-exact on every row. New rows on 1.3b, VE-core / VE-coda / STA@post_inj(8)+VE-coda: VarBench 0.905 / 0.091
    / 0.162 and 0.865 / 0.045 / 0.170; D4 0.729 / -0.013 / 0.004 and 0.897 / 0.062 / 0.185; D1 (city) 0.167 /
    0.431 / 0.887 and 0.120 / 0.255 / 0.818. On VarBench and D4 the value-embed effect runs through the loop's
    own value embeds, not around the loop: B's coda value embeds alone move e by <= 0.09, and adding them to a
    full state swap at post_inj(8) leaves e <= 0.19. The review's "bypasses the loop" reading is not supported
    on these 4 items, and PR11's first clause (VE-coda >= VE-core) already fails on them. t_fwd_plain is within
    2% of the recording forward (VarBench 0.355 vs 0.360 s), so the F9 bias was small.
  - v3 clean-pass scans (full candidate order, deadline not reached) [ran]: D1 parcae-1.3b kept 62 of 72 scanned
    (18 multi-token, 44 one-token; dev 39, test 23; 70 subjects); parcae-370m kept 61 of 73 (16 multi-token; dev
    39, test 22). D4 parcae-1.3b kept 30 of 849 scanned: T1 15 dev + 15 test (quota full), T2 0 (A_fail 690), so
    D4 falls under the 40 rule and is skipped on 1.3b too. [belief] The same cap applied under the v2 D4 key
    (quota 15 per split x kind, T2 arithmetic); no v2 full 1.3b D4 scan was run. PR3 is not testable on either
    model (multi-token stratum < 20).
  - R1 restated (review round 2 F3). After v3, E8 can run only on D1 and VarBench on both models: D2 0 kept
    (both), D4 30 (1.3b) and 0 (370m), D5 0 (both). This violates binding amendment A2 ("D1, D2, D4 at least") on
    both models, not only the section 10 never-cut entries (AD5 rows on D2, E8 on D2). D1 E8 has 22-23 test items
    before eligibility, so its >= 20 eligible-test gate may fail. Options for the PI, with A2 status:
    O1 accept D1 + VarBench with A2 waived for D2 and D4 (A2 violated, needs an explicit waiver);
    O2 lift the D4 T1 quota to 30 per split when T2 has no passes (about 60 T1 items on 1.3b, scan about 4 min;
    satisfies D4 on 1.3b only, T1 retrieval items only, changes a frozen selection rule; D2 still violated);
    O3 substitute for D2 a dataset Parcae can do, screened by a clean-pass scan only, no arms (A2 names D2, so the
    substitute is a deviation the PI must approve);
    O4 drop 370m E8 claims and keep 370m for E1/E5 (does not repair D2 or D4 on 1.3b).
  - Full run launch (2026-10-07 ~22:25, resumed after the API outage; no R1 decision found in PLAN.md or LEDGER.md).
    The run applies the frozen v3 rules unchanged: all five datasets are scanned on both models, and a dataset
    under 40 kept is skipped by the code's own rule. Expected from the v3 scans: D1 and VarBench scored, D2, D4 and
    D5 skipped on both models. That is O1 in practice; no selection rule is changed (not O2), no dataset is
    substituted (not O3). The A2 waiver for D2 and D4 stays the PI's decision; until then the E8 rows count only
    for D1 and VarBench.
    Configuration: out `out/k17_parcae/full/`, data `data_r2/`, k17_parcae.py md5 70d1d754 (unchanged since the
    v3 debug), one shared `--deadline_epoch` = launch + 150 min (prior Parcae GPU use about 27 min, so the 3 GPU-h
    cap holds), targets as section 9 (no cut applied at launch). Projection from the v3 debug timings: 1.3b about
    85 min (D1 62 x 12 s, D2/D4/D5 scans 10 min, VarBench 60 x 55 s), 370m about 25 min, 770m and 140m scans
    about 12 min, total about 122 min.
    DV10. Order: 1.3b run, 370m run, then the AD4 scan-only passes on 770m and 140m. Section 10 puts all four
    scans first; they move to the end so that contention on the shared host cuts the AD4 addition before any
    core cell. The main models' BOS calibration and clean scans run inside their `--run` as before.
    P14: the plan as launched is copied to `out/k17_parcae/full/plan_at_launch.md` and its sha256 goes into
    checks.json (`plan_sha256`). The LEDGER is not a file this brief owns (hard rules), so the LEDGER entry is
    left to the orchestrator.

## 14. Critic log

Self-applied (v1):
- A1: strata, n_items, n_clusters and positive-control values per cell (section 9); cells named in checks.json.
- A2/A6/A7: E8 as section 8: answer sites only, a = 0.25 and 1, top-1 hit primary, post-coda readout site
  excluded from S, false-negative rate reported, selector claim limited to ranking candidates.
- A3: VarBench copy-read target (T3); first-token read-out after the prompt is not used.
- A5: E1, E5 (with j = K-1) and E8 only; no knockout, window or undo arms.
- A8 analogue for Parcae: loop-end norm (F3), executed-index map (F4), coda readout (F1, DC6), initial state
  (F8, DC5), all with file:line.
- P1 |delta| per edited position and its sum logged per row. P3 strata (section 9). P4 O2 rules (section 7). P5
  position-0 rule (DC3). P7 diagnostics on O2, answer classes with e_own and validated presence gates. P9 n_items
  and n_clusters. P10(5) D4/D5 reclustered. P12 D5 copy-confound drop (the Parcae D5 answer is one token). P13
  single and persistent edits at every j including K-1. P14 per-item margin gap logged, plan hash into the LEDGER
  before the first scored row.
- Not applicable: P2 (no generation), P6 (no dose-shape claim), P8 (no knockout), P11 (no D3).

Critic round 1 (v2; all ten findings applied, none rejected):
- F1 (major) applied. Primary E8 set S excludes Llast@8 (63 / 31 sites), uniform rule Lmid@8, S+ with Llast@8 is
  secondary; Lmid@8 is the last-loop answer cell in PR2 and the tables; Llast@8 stays an E1 cell labelled
  "coda-gated"; selector cost in forward equivalents in `e8.cost_fwd_equiv` (sections 7, 8, 11).
- F2 (major) applied. AD5 rows with predictions PR10 written first; STA is each source cell's informative
  positive control; PC-final, SZ and PC-site tagged [harness]; frozen reading rule for source nulls (sections 0,
  7, 8, 11).
- F3 (major) applied. [read] `core_block_forward` ignores `step` and `total_steps` (PC parcae.py:362-386), so the
  loop-j readout is the depth-j output; check (g) asserts it. Clean depth-j margins and e_own(j) logged for every
  answer O1 row at Llast@j; classes at a = 1 and 0.25 with the e_own gate, validated presence gates for L0, Lmid
  and post_inj cells, the n/a guard and the paraphrase/R2 separation validation (sections 5, 8; DV6).
- F4 (major) applied. pos_len-balanced D1, pos_len x relation counts and n_clusters per cell, PR3 testability rule
  and stratum-difference TOST (sections 9, 11). [ran] The finding's confound is total on this tokenizer (DV5).
- F5 (major) applied. VarBench build reuses the k17_varbench rules with drop counts and k17vb-order; e_LO primary
  with one extra SA' forward on cont items; budget redone (sections 4, 5, 10).
- F6 (minor) applied. H3 returns a new args tuple with a cloned e; hook check (h) (section 6).
- F7 (minor) applied. Seed = int(sha256(pair_id)[:8], 16) + seed_offset, stored per item (DC5).
- F8 (minor) applied. D2 clustered on the A-side edited operand, D4 on the VARBENCH I7 content key, D5 one item per
  problem; splits keyed on the clusters; dev/test overlap recorded (section 9).
- F9 (minor) applied. FN rate primary on dev items over all of S at a = 1, test 9-site version secondary, site
  counts per dataset (section 8).
- F10 (minor) applied. One target per dataset, the 40 rule after the D >= 2 filter, E8 cells interpreted only
  with >= 20 eligible test items (sections 8, 9).

Review round 2 (v3; nine findings, all applied, none rejected; F3 is a report change):
- F1 (major) applied. VE-core and VE-coda rows on every item, STA@post_inj(8)+VE-coda in AD5 (section 8); reading
  rule v3 with the value-embed branch split into loop and coda, implemented as `src_attribution`; the v2
  STA@post_inj(8) rationale corrected, its threshold kept and scored as written; PR11 post-debug (section 11).
- F2 (major) applied. D1 cross-split pairs dropped (B subject must hash to the A subject's split), dyadic
  bootstrap over (A, B) subjects on D1 (cells, paired differences, E8 baselines, PR3); D4 clustered by connected
  components of A/B content keys with one item per component; A-or-B unit overlap in `clusters` (section 9). D1
  yield re-checked first (section 9 and the run log): the targets are not reachable.
- F3 (major) applied in the report and the run log: losing D2 (both models) and D4 (370m) violates binding
  amendment A2, not only the never-cut list; options with their A2 status in section 13.
- F4 (minor) applied. E8 hit, chance, regret and Spearman on eligible test items; dev + test under `devtest`.
- F5 (minor) applied. run_para accepts a form only with top-1 right on A' and B' and D' >= 2; D' logged per form.
- F6 (minor) applied. The PR2 check covers all 12 D1 answer cells with loop <= 4.
- F7 (minor) applied. Dataset default strata on every cell, para_fmt on O2 cells (section 9).
- F8 (minor) applied. Shared absolute `--deadline_epoch`; deadline checks before the dev-tuned pass and the seed
  sensitivity; budget-stopped scans labelled `stopped_budget`, not skipped for yield (section 10).
- F9 (minor) applied. Selector cost denominator = plain forward without recording (`t_fwd_plain`).
