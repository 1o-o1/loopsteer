# RA-29 review: raven cells (section 13), Thinking cells and grid (section 14), D8 to D11

Plan sha256 6655a019 (matches plan_ra29.sha256). Scope: ra29.py, make_data.py, lib/k17_freeze_raven.py,
tests/test_ra29.py, data/items_*__{mcleish,huginn,think14,think26}.json. Read only; nothing else was changed.

Verdict: GO after fixes. The Ouro, Thinking and McLeish/Huginn mechanics are correct. B1 and B2 apply to the raven
cells only, and nothing has run on raven yet. Fix them before the raven pilots (needs a plan amendment and a new hash).
The Thinking and Ouro cells can run now.

## What was verified

- V1 Item maps (my script re-tokenizes every built item from the local tokenizer.json at the pinned revisions):
  for huginn, mcleish, think14 and think26 on every dataset, the ids sha256 match, T is equal for A and B,
  r = T - 1, the token at r is the cue " (", A and B are equal outside [b0, b1) and differ inside it, and the gold
  label is inside the block. Zero failures. The block starts at "Ġ(" on every item. After the block comes "Ċ"
  (Huginn), "<|im_end|>" (Thinking), or "A" when D10 applied (McLeish: AQuA 24, CSQA 1, ARC 510, where the last
  block token is ".Ċ", "%Ċ" or "?Ċ"). The Thinking tail is "<think>\n</think>\nThe answer is (". There is no leak
  of B's answer: B differs from A only inside the block, and the exemplars, cue and tail are the same ids.
- V2 The raven CPU tests pass (TestRaven, TestRavenE2E: 6 tests, both variants).
- V3 On the CPU fake (both variants), a patch at input.prelude_out at the block positions with B's prelude output
  makes the block's loop-end states bit-equal to B's at all 32 loops through the prod hooks. On the fake, a blk_L1
  perturbation decays to a block-state distance ratio of 0.01 (Huginn fake) and 0.000 (McLeish fake) by loop 32.
  The fake has random weights, so this shows the mechanism only, not the size on the real models.

## Answers

1. D8 is correct. Huginn reads out as lm_head(ln_f(coda(ln_f(h_32)))) and McLeish as lm_head(ln_f(coda(h_32)))
   (tree/prod/steer/adapters/raven.py:406-453). The coda attends to every position, so loop32.loop_end at r is not
   the readout input. coda.L{Cd-1}.resid_post is followed only by per-position ln_f and lm_head. So pc = 1 and
   sz_fin_blk = 0 hold exactly. lt_L32 is a valid result arm. But a "lands" bar of 0.5 mixes a commit at r with the
   coda reading the block and cue directly (N3).
2. Yes, blk_persist patches loop1..loop32.loop_end at the block with B's per-loop states (ra29.py:133, payload_for
   :331, patch_written over all 32 loops :361-366). Yes, single-loop block edits vanish largely by construction:
   every loop recomputes the block state as adapter(cat[state, e_A]), and the raven models are trained to forget
   their initial state (path independence). So blk_L1, blk_L2 and blk_L3 being about 0, and lt_L1 being about 0,
   do not test the read schedule. A null on blk_persist (< 0.5) is uninterpretable. A's option content still
   reaches the cue positions and r through e (the prelude attends to A's tokens), and through the in-loop K/V of
   the block (computed from B's state plus e_A). So a null cannot say "the block is not read". See B1 and B2.
3. Yes. The x0 table is keyed by the item key, and row() uses that key for clean A, clean B, zero, every arm and
   the gate (ra29.py:269). nohook takes the same x0_table(key)[:T] (lib/k17_freeze_raven.py:62-72). t_max is the
   longest built prompt + 1, the same in pilot, full and resume (ra29.py:763-764). Within an item, arms differ only
   by the patch.
4. D9: the prompt does not depend on the depth, so the length anchor is valid. The k = 8 accuracy is a projection
   input only (N6). D10: correct. The boundary at c1 + 1 is applied to A and B alike, and B is re-checked
   (make_data.py:171-196). D11: correct. "</think>" is re-encoded by check_tail_ids, and the grid's prompt
   n_prompt_tokens match on every built item. A/B length equality, block bounds and the readout position hold on
   every item (V1).
5. The gates are bit-exact with no dtype trap. Donor payloads are float32 recordings of bf16 values, so the cast
   back is lossless. Random payloads are rounded to bf16 before the patch_written compare (ra29.py:341). pc nld ==
   1.0 is exact in IEEE (m equals mB).
6. Huginn CSQA is 6.32 single-card hours, 3.79 h at 0.6x, so one job fits under 9 h. With B1 and B2 (21 forwards
   per scored item) it is about 6.8 single-card hours, 4.1 h at 0.6x. All other cells are under 1.5 h at 0.6x. No
   shard is needed, and SIGTERM resume (exit 5) covers an overrun.

## Blocking

B1. blk_persist can land with no persistence at all. On raven the coda reads the block's loop-32 states directly,
    so the loop-32 write in blk_persist alone may carry the effect (on Ouro blk_L4 is a structural zero, but here it
    is not). PR1 "lands only as a persistent edit" is then not tested.
    Fix: ra29.py:127-143 (ARMS_RAVEN), add ("blk_L32", "donor", (32,), "block", "end") after blk_L31. Report
    blk_persist - blk_L32 in results.json (write_outputs :501-509) as a descriptive field. Add to plan section 13
    (amendment, new hash): "PR1 lands is read with blk_L32 beside it; if blk_L32 >= 0.5 the persistent half of PR1
    is reported as carried by the coda read, not by persistence."

B2. There is no complete persistent block edit, so a PR1 null is uninterpretable, and the about-0 half is not
    separated from contraction.
    Fix (a): add arm blk_e, a patch of input.prelude_out at the block positions with B's prelude output (one
    write). The block then runs exactly as in B at every core layer and loop. Only the prelude route at the
    post-block positions stays A.
    - recs_all :287 and M_of :290 record I.expand("input.prelude_out", {"kind": "once"}, X.g)[0] at the block
      positions in the clean runs.
    - site_of :296 gets a label "pe".
    - payload_for :331 returns MB["pe"][sel["block"]].
    - patch_written :361-366 reads Mz["pe"].
    - The others-check at :373-376 must skip label "pe" (r changes from loop 1).
    - Add the gated identity blk_e_block_equals_B: H[j][block] == HB[j][block] for j = 1..32. It is bit-exact on the
      fake (V3). Add it to GATED_RAVEN :144.
    Fix (b): record, with no new forward, ratio_blk32 = ||H[32][block] - HA[32][block]|| / ||HB[32][block] -
    HA[32][block]|| and the same ratio at r for every raven arm in rec["arms"][name]. An about-0 NLD with ratio
    about 0 is decay, not a read-schedule result. Write both into plan section 13 before the hash.
    Tests: extend TestRaven.test_arms_and_identities_both_variants for blk_L32, blk_e and the new gate.

## Non-blocking

N1. write_outputs (ra29.py:501) prints PR1/PR2 verdicts on pilot-only and under-80 cells. Gate them on kept_ge_80
    or label them "not scored".
N2. "About 0" is scored on the point mean. Also report whether the 90 percent CI lies inside +-0.1.
N3. lt_L32 < 0.5 does not show "no commit at r", because the coda also reads A's block and cue. blk_L32 (B1) gives
    the coda's direct share. Optionally add a loop-32 end patch over b0..r as an exact-1 identity (D8 from the
    other side).
N4. debug_gate norm_ratio_min_max lists rnd_lt_L4 (ra29.py:831). On raven it is None. Add rnd_lt_L32.
N5. run.sh cell() starts the full run without checking the pilot's timing.full_card_hours_at_this_rate against the
    cap. Raven rates are estimates. Add a check (stop at > 8.5 h x 0.6). Also submit with --signal=TERM@120: one
    Huginn item takes about 27 to 45 s, which can exceed the default 30 s KillWait. Resume still recovers.
N6. McLeish projections use k = 8 accuracy. K = 32 accuracy differs. The pilot projection decides, so this is
    only a planning note.
N7. Under D10 the McLeish block holds one more token (".Ċ") on 535 items. Note it in any cross-model block
    comparison.
N8. lt_L31 being about 0 is also partly predicted: e_A(r) is re-injected at loop 32. Read PR2's about-0 half with
    the B2(b) ratio at r.
