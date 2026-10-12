# RA-30 senior review (plan sha256 21b0a62f..., 2026-10-12)

Verdict: GO after fixes (B1 to B4). The CPU suite passes (20 of 20, run with -B, nothing written).

## Verified for this review

- V1 On the prod CPU fake (scratch script, nothing written to the repo), cvP and cv1 give bit-identical K/V caches and logits after the prefix pass.
  cv3 and kv4 give bit-identical K/V caches at every loop, layer and position. They differ only in the next-token logits at the last prefix position (max 0.28). Their 20-token decodes are identical.
- V2 text and cv1 have identical K/V at loops 2 to 4 and identical last-position logits. They differ only in the loop-1 K/V at P.
- V3 torch.lerp at weight 1 returns `end` exactly (end - diff * 0). The donor is bf16 from the model, so the bf16 to float32 to bf16 round trip is exact. G5 for cv_j, kv4 and the full neuron set compares bf16 with bf16 and has no RA-28 dtype trap.
- V4 OuroAdapter.decode (prod/models/ouro.py 397-436) keeps the batch at full width. Finished rows feed pad, and the loop ends only when every row has finished. Rows in one batch therefore share kernels and batch size for the whole decode, and batch time is set by the longest row.

## Blocking

- B1 plan_ra30.md:16-17 (P4), 96-98 (arm table), 105-108, 248 (R7). The loop ladder has two arms that are structural duplicates (V1).
  cvP equals cv1 because the context before P is identical in A + prefix_A and in A + prefix_B. After post@1, loops 2 to 4 at P already recompute the donor's states.
  kv4 equals cv3 for every later token because post@3 sets the loop-4 input at P, and that input fixes the loop-4 K/V at P.
  As a result, P4's "kv4 gives the least of the four" and the cv3 - kv4 difference are pre-registered comparisons that tie by construction.
  Fix:
  - Reword P4 to "CA cv1 >= cv2 >= cv3".
  - State cvP = cv1 and kv4 = cv3 as predicted identities. Print the token-identical share, and allow divergence only at a near-tie under the G2 rule.
  - Add one CPU test with the V1 assertions (K/V equality of cv1 against cvP, and of cv3 against kv4).
  - Recommended RA call: add a kv1 row ({"op": "kv", "loops": [1], "pos": P}). By V2, text = cv1 + loop-1 K/V at P, so kv1 is the only read the current arms never edit, and the P3 ratio cannot be interpreted without it. The batch grows to 16 rows. Alternatively, kv1 replaces kv4.
- B2 Stale-donor hole. SteerHooks.ensure_donors (steer_gen.py, "if key in self.donors ... continue") never re-records a key it already holds. G5 compares the patched state against that stored donor.
  If Phase 2 reuses a row key across items, every arm silently patches the previous item's donor. G5 still passes, and so does the a_ids prefix check in decode.
  Section 13 also says "donors shared per sequence". record_donor records kv only when the row has a kv edit, so a donor copied from a cv row has no kv and kv4 fails.
  Fix (plan section 13 and the Phase 2 code):
  - Row keys are "<ds>:<idx>|<arm>".
  - steer.rows, steer.donors, steer.stats and hook.specs are cleared after each item.
  - The shared prefix donor is recorded once from the kv4 row's info, so that it includes kv.
  - G5 on the gate items compares against a fresh record_donor of A + prefix_B, not the cached entry.
- B3 Prompt B length is never checked. lib/arms.py:66 gives scratch rows donor_range = (span0, len(prompt_A)) and applies it to b_ids = prompt B. make_data.py:99 checks only prompt A against the pool plen.
  If prompt B is shorter, the run raises. If it is longer, the base arm silently takes B's position plen - 1, which is not B's last token, and the span and ns arms read shifted rows. R5 (tokenizer drift) makes this possible.
  "plen_mismatch" is also missing from DROP_REASONS (lib/chain.py:19-39).
  Fix: in the screen, require len(prompt_ids(B)) == len(prompt_ids(A)) == pool plen and equal span bounds. Otherwise drop with "plen_mismatch", which must be added to DROP_REASONS. Also assert this in row_info for scratch rows.
- B4 lib/chain.py:253-258. The search for the first written derived_A starts at the end of the previous annotation. Two cases give a wrong P:
  - derived_A is already written earlier in the chain (F13: 2 GSM8K items). The edit then leaves an unedited earlier copy that later steps can read.
  - derived_A equals r_(i0-1). Its ">>X" copy starts exactly at `start`, so P lands on the previous result.
  Fix: after `occ` is found, drop with a new reason "value_earlier" if any NUM_RE match in body(chain_text)[:occ.start()] has the value of derived_A, or if occ.start() >= eqs[i0]["span"][1]. Add the reason to DROP_REASONS and add one test.

## Non-blocking

- N1 Section 11 arithmetic does not follow from its own rates:
  - 160 items x 54 s = 2.4 h, not 2.0 h. Full GSM8K is therefore about 3.5 h against the 4 h stop.
  - The GSM8K pilot is 90 x 10 s + 35 x 54 s = 47 min, not 35 min. The total pilot is about 1.9 card-hours.
  - By V4 a single broken row (rand, r5, neuron) runs the whole 15-row batch to the horizon, so the "worst case" will be common.
- N2 Horizon. Keep 512 in the pilot, because the clean screen and Table 2 a1 use it. Pre-register this cap rule now: the pilot logs the token index of the answer marker per row. If fewer than 1 percent of rows with a parsed answer reach it after min(512 - prefix, 2 x clean continuation + 64), the full run uses that cap.
- N3 Batch against single is tested only on the float32 fake, with 7 prefix arms, 15 tokens, and no neuron or scratch rows. Add nc_* and ns_* rows to test_batch_of_arms, because the per-row loop of the neuron hook in a batch is untested.
- N4 Comparing rows inside a batch is valid (V4): an edit that does nothing gives a token-identical row. Eligibility, however, comes from the batch-1 clean chain, and the in-batch floor can diverge at near-ties. Print CA both on all items and on the items where floor (prefix batch) and clean (scratch batch) end in answer A.
- N5 T5. The printed norm ratio is computed in float32 before the bf16 cast (steer_gen _stat on new - cur). Label it "pre-rounding", or measure it on the applied bf16 delta with a 1e-2 tolerance. Never use torch.equal or 1e-5 on it on the card.
- N6 All fake tests run in float32. Run T4, the neuron hook and the idle-hook tests once more with the fake cast to bfloat16, to prove V3 on CPU instead of asserting it.
- N7 The neuron arms replace only the loop-1 MLP write of layer 23 at P. This is a component of the state difference, not a bound on the effect: the decode is nonlinear, so nc/cv1 is not capped at 1. Change the table note to "carries only the last layer's MLP part of the loop-1 state difference".
- N8 There is no answer-B leak through the prompt: every prefix arm and its donor use prompt A, and only the scratch arms use prompt B (by design). Two coincidence risks remain:
  - gold_B occurs as a number in question A or prefix_A. Count these items and give CA without them.
  - derived_B is a prompt number (derived_B_in_prompt, 7 GSM8K items in F13), which makes carry = B ambiguous. Report carry without these items.
- N9 P includes unchanged leading digits (for example "1" in 15 -> 10). The donor difference there is only numerical noise, so cv and the random arm are near no-ops at those positions. This is harmless, but note it next to the per-position norm ratio.

## Decisions on the open risks

- R1 Do not add the "expr = number" fallback, and keep the chain_arithmetic_error drop. Reasons:
  - GSM8K is the main line at 156, about twice the gate. ASDiv is overflow under the schedule correction.
  - The fallback adds an untested regex path to item selection after the prescreen.
  - Prose equations have no written result to cross-check.
  - The consistency rule trusts the written results.
  If the full ASDiv screen keeps fewer than 80 items, report it as a gate failure.
- R4 Accept the single-card lists (395 and 1,036). The chains are regenerated, so the list only sets which items are used, and the base arm is measured in this run. The table note names the single-card a1 scored list and does not quote 391 or 1,034.
- R6 Keep the brief's 50-item split. ASDiv ds5 then reads about 40 items. Print n and the interval, and do not add cross-fitting now.
- R7 Accept rnd3. A random patch at post@4 cannot reach later tokens (T1 post@4 test), and rnd3 is the matched control of cv3 = kv4. Keep cvP because the brief lists it, but report it as the identity with cv1 (B1), not as its own result.
