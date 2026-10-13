# RA-29 plan review (plan sha256 eda049ad..., 2026-10-12)

Verdict: GO after fixes (B1 to B3). No card run is needed for any fix.

## Blocking

B1. ra29.py:69 (plan section 5 row 9, N2, R2, section 1 P1 text; ra29.py:291). persist_lt_L1to3 equals lt_L3 bit-exact
at every position, so it is not informative. r = T - 1 is the last position. No position before r attends to r, so
the loop-1 and loop-2 patches at r reach only r's own stream, and the loop-3 end patch overwrites r with HB[3][r].
Loop 4 then reads the same states as in lt_L3. The same argument makes the briefed persist_lt_L1to4 equal lt_L4 at
every position, not only at r. Fix: delete the ARMS entry at line 69. In line 291 replace "persist_lt_L1to3" with
"lt_L2". Change "18 forwards" to 17 in plan sections 5 and 8. Rewrite N2 and R2: both persistent arms at r reduce to
the last single-loop arm by causality, so the brief's persistent arm is reported as identical to lt_L4. Re-run the
CPU tests (forwards = 3 + 1 + 13).

B2. ra29.py:500-521. A kept item that fails an identity is written to records.jsonl before exit 4. The resume
command (the documented recovery after exit 5) puts that key in `done`, skips it and continues. The failed item then
enters the means, and the full run finishes with only a count in checks.json. Fix: after `recs = load_jsonl(rec_path)`
add `bad = [r["key"] for r in recs if identity_failures(r)]` and `if bad: die(EXIT_IDENTITY, "records hold identity
failures: %s" % bad[:5])`. After `write_outputs` in a full run, die with EXIT_IDENTITY if `chk["identity_failures"]`
is not empty. Add a CPU test: exit 4, rerun the same command, expect exit 4 again.

B3. ra29.py:504-512 and 107. The drift check is per item and lazy. A drift on item n stops the full run after
n items, and a resume stops at the same item again with no recovery except a data rebuild. Also, suffix_ids, cue_ids
and letter_ids come from meta (built under transformers 5.7) and are never re-encoded on the card (4.56.2). The ids
sha256 includes them, so drift in them is not detected, and the prompt can differ from the grid's prompt without any
check. Fix: after load_ctx and before the item loop, materialize every built item (CPU, seconds). Also check
`enc(meta["suffix_text"]) == meta["suffix_ids"]`, `enc(" (") == meta["cue_ids"]` and `enc(l) == [meta["letter_ids"][l]]`
for a to e. Exit 2 on any mismatch. Keep the materialized list for the loop.

## Checked and correct

- Arms: lt_L1 is ARMS[0] and is the first row of every table. lt_L4 and pc are the same spec (same site, position
  and payload), and a = 1 lerp writes the donor exactly. The final logits at r depend only on loop4.loop_end at r
  (post-norm state, readout = lm_head of that state), so pc, lt_L4 and persist_lt_L1to4 equal clean B at r.
  blk_L4 is a structural zero: r is outside the block and lm_head rows are independent. The brief's arm list is
  complete.
- Prompt B is a valid minimal pair. T, r, the block boundaries and every id outside the block are equal (built and
  checked in make_data.py:129-135, re-asserted on every item in the tests). The texts are the same multiset. The
  readout was checked on the local tokenizer: " (a)." encodes as [365, 81, 595], the letters a to e are 81 to 85,
  and the stored suffix ids reproduce. The few-shot head and the question stem are the same in A and B, so they do
  not leak B's letter.
- Gates are dtype-safe. Donor payloads are float32 copies of bf16 states. The random payload is rounded to bf16
  before it is written (line 201). Each bit-exact comparison compares two outputs of the same bf16 path.
- Cost: 18 forwards per kept item (asserted in the tests) and 3 per screened item. At 0.15 to 0.25 s per forward,
  1.4B AQuA plus CSQA is 0.7 to 1.3 card-hours (CSQA keeps 600 to 900). This agrees with "about 1".

## Non-blocking

N1. T5: the norm ratio is stored, not gated. "Within 1e-2" is not proven on bf16. Read the min and max in the pilot.
N2. T1: before_patch_unchanged and causal_prefix_unchanged cover only the recorded set (block, r, pre). Say this in
    the plan. The brief's decoding identity reduces to the hook identity because the readout is the prefix top-1.
N3. The 80-item rule counts items after the gap drop (scored n), but the brief counts screen survivors. Name the
    count that the RA decision uses (the scored n is recommended).
N4. The new gold letter depends on the walk order, not only on the item key (the brief says "seeded from the item
    key"). The RA's uniform-letter decision allows this. Record it as a deviation.
N5. Letter balance after the screen is only reported (by-letter means, no CI, four arms). This is enough for R3.
N6. clean_loop_margin uses a one-row lm_head GEMM, so loop 4 can differ from mA in the last bits. Do not compare
    the two.
N7. A partial last line in records.jsonl after a SIGKILL breaks load_jsonl on resume. Skip a trailing line that
    does not parse.
N8. The local tokenizer is the transformers 5.7 fast GPT2Tokenizer. On 4.56.2 the same tokenizer.json is used, so
    drift is unlikely. B3 makes any drift an early exit 2.
