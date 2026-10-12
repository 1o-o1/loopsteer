# RA-30 McLeish r32 review (plan 6c, sha 292a1238)

Reviewed: plan_ra30.md 6c and 8, ra30_mcl.py (sha 1e6694fb, 691 lines, the version with RA D1/D2), lib/steer_gen_raven.py
(sha a862710f, matches the plan), tree/prod/steer_gen.py (edit kernel), the checkpoint's raven_modeling_minimal.py
(a5f6f126, static cache and forward), lib/chain.py, tests/test_ra30_mcl.py. Ran: `pytest tests/test_ra30_mcl.py`,
9 passed (CPU fake).

## Verdict: GO after fixes

## Blocking

B1. The code and the plan of record do not agree. ra30_mcl.py:46-49 (MIN_ELIGIBLE_FULL = 20, SHARDS_FULL = 3),
pilot_of_record at ra30_mcl.py:381-398, run.sh:84-86 (3 shards), and the comment "RA D1 (plan 6c)" all refer to
decisions that plan 292a1238 does not contain. plan_ra30.md:224 and 232-233 still say the full run stops under 80, and
the plan has no shards. The pilot records plan 292a in hash_line.json, and that plan describes a different full run.
Fix: before the pilot, add D1 (80-item gate waived, n >= 20 projected eligible) and D2 (3 item shards, order[s::3],
3 gate items per shard, summarize merges the shard folders) to 6c. Re-hash plan_ra30.sha256. Run the pilot under the
new sha. pilot_of_record accepts only ACCEPTED_PILOT_PLANS or the current sha, so a plan edit after the pilot makes
the full run exit with EXIT_PLAN.

B2. When M2 fails, the result cannot be interpreted. Add an e-patch arm before the pilot freezes the arm set.
Facts from the model code: the core K/V that later tokens read at P are computed inside each loop from
adapter(state, e). Under cvP, the state is the donor's, but e is A's at every loop. So core K/V at P for loops 1..32
carry e_A. The prelude K/V at P (4 layers) are A's. Only the coda K/V at P and the logits at P are the donor's.
cvP is therefore testable: it is not zero by construction, because the coda entries and the readout at P are B's.
But its ceiling is below text by construction. A low CA(cvP) cannot separate two causes. Cause 1: the loop
re-derives from the re-injected input e_A, which is the claim. Cause 2: later tokens read A's digits from the
prelude K/V at P, a token channel with no loop.
Fix (no lib change, because the hook already supports site pre_out at loop 0 and the donor record already holds
post[0]):
- ra30_mcl.py:60, after kvP: `("eP", "prefix", "A+prefA", "A+prefB", [("patch_e", [0], "P")], "input-injection patch: prelude output e at P = donor's")`.
- ra30_mcl.py:194-197 row_info: `elif op == "patch_e": edits.append({"op": "patch", "site": "pre_out", "loop": [0], "pos": p, "a": 1.0})`.
- summarize, next to ra30_mcl.py:553-554: print `CA(eP)/CA(text)` and `CA(cvP)/CA(eP)`.
- 6c: in the arm table and T1, add "eP: the core and coda entries at P equal the donor's, and only the prelude entries
  stay A's". State the reading: eP close to text removes cause 2. Then cvP below eP is the input-refresh effect.
- tests: add an eP case to test_T1_cache_semantics. The core and coda slots at P must be allclose to text. Slot 0 must
  equal floor.
Cost: the prefix batch goes from 7 to 8 rows. Batch time is set by the longest row, so the added time is small.

## Non-blocking

N1. Plan 6c:205 says "prelude and coda K/V not edited". The coda K/V at P are not A's. They are computed from a
residual that attended to the donor's core K/V. Write "prelude K/V at P are A's, coda K/V at P are mixed".
Optional: add loop 0 to kvP (`[0] + LOOPS`). The hook and the donor record support it. kvP then differs from text
only in the coda entries at P.

N2. G5 (ra30_mcl.py:298-353) checks loop 1 only. M2 depends on cvP and kvP. On 1 gate item, also check that post@j
and kv@j land at P for every j = 1..32.

N3. Q3 (identity and dtype) found no trap. lerp(cur, t, 1.0) returns `end - diff * 0`, which is exactly t in float32,
and the cast back to bf16 returns the donor's bf16 value. G5 compares bf16 to bf16. G1 compares the float32 casts of
the bf16 logits. The x0 tables are bf16 on both the donor path and the decode path.

N4. Q4 (x0 sharing) is correct. MclAdapter.table maps every key to its problem id (key // 100, arm numbers < 100). The
screen, all arms, both donor records, and every decode step (x1 = table[pos]) use the same bf16 table. Table 2 used
the same seed (row key = task idx). Table 2 decoded in left-padded batches of 16, so fresh chains are not bit-identical
to the stored chains. The 0.325 eligible rate is a belief until the pilot measures it.

N5. Q5 (value_is_operand) matches 6c (operands of e1 only). It can drop more items than needed, and that is safe.
Llama token counts are guarded three times: the character-length check in analyze (equal digits give equal 3-digit
chunks), token_count_ids on standalone B ids, and the decode-equality check in token_boundary. clean_up_tokenization_spaces=False is used everywhere.

N6. Cost. On the stated per-item numbers, each shard is 7.9 / 3 = 2.6 h. At 0.6x speed that is about 4.4 h, plus
about 10 min for the model load and gates, so it is under 9 h. The margin is about 1.9x. The weak inputs: (a) 60 s
per clean decode uses the mean stored chain length. Wrong chains that run to the 512-token horizon cost about 435 s
each. (b) 180 s per armed item assumes no row in a batch runs past the clean length. One rambling rnd1 or kv row sets
the time for the whole batch. Rule at pilot review: keep 3 shards only if checks.projected_full_card_hours <=
16.2 (3 x 9 h x 0.6). Otherwise raise the shard count. run.sh:84 hard-codes 3, so that change is a code edit.

N7. The checks.json key gate_80 (ra30_mcl.py:584) still prints a boolean against 80. config.json says it is waived.
Rename the key or print the D1 rule (n >= 20) so the two files agree.

N8. The full run screens the pilot's ~62 items again (separate folder), about 1 card-hour. This is acceptable, but
6c should state it.
