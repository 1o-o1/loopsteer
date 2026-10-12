# RA-31 senior review (2026-10-12)

Reviewed: plan_ra31.md (sha256 7868dba8e0f5...), ra31.py, lib/ra31_hooks.py, tests/test_ra31.py, lib/ra28_hooks.py,
lib/run1c.py (random rows), lib/k17_freeze.py (freeze), tree/prod/steer/adapters/ouro.py (forward), modeling_ouro.py
(OuroRMSNorm, layer forward), data/run1c_ref.json. Ran the suite: 16 tests, OK, 14 s (CPU).

Verdict: GO after fixes. No defect in the Part A engine. Three plan items must change before the hash, because
the hash freezes them.

## Blocking (fix before plan_ra31.sha256)

B1. plan_ra31.md:101-103, G4 can fail for a dtype reason. dall minus all at alpha is
(alpha - 1)[(c_all - w) - (c_all_clean - w_clean)], with w the bf16 norm output and c_all an exact float32 sum.
This is unbiased bf16 representation noise of about 1 ulp at each of 72 sites, and it is the same size at
alpha 0 and alpha 2. A code bug (wrong W, g, eps, rms, column map or clean record) gives a systematic gap. The
test test_reduction_detects_a_wrong_clean_value shows this. A fixed 0.01 bound on mean |gap| tests the noise, not
the bug.
Fix: per dataset and branch at alpha 0, gate on |mean signed gap| <= 0.01 (bias) and mean |gap| <= 0.03 (noise
ceiling). Print max |gap| and the alpha 2 values. State that the alpha 2 dall-all gap is the noise floor for
reading top and rand at alpha > 1.

B2. ra31.py:206-209 and plan_ra31.md:93-94, first_site_scale is listed in G1 (exit 3) and depends on dtype. The
norm ratio ||w' - w_c|| / ||Δ|| has a bias of about (k ||w'||/||Δ||)^2 / (2 alpha), with k ~ 1.6e-3 (bf16
rounding of w'). At alpha 0.25 this exceeds 0.02 when ||Δ|| < ~1.6 percent of ||w'||. That is unlikely but not
excluded for attention at loop 2, layer 0.
Fix: use the projection ratio ((w' - w_c) . Δ) / ||Δ||^2 with Δ = w_E0 - w_c. Its rounding error is smaller by
about sqrt(d). Keep the 0.02 tolerance.

B3. plan_ra31.md:130-139, the pre-registered readings are not numeric. Points 4a and 4e below give the readings.
Write them into section 8 before the hash.

## Answers to the five checks

1. Arms.
   - alpha 0, "all": lerp(x, t, 1) returns t exactly (PyTorch uses end - diff * (1 - w) for |w| >= 0.5).
     test_all_alpha0_equals_prod_hold proves it bit-equal to k17_freeze ans_frz_* on the bf16 fake. G3 checks it
     on the card.
   - alpha 1: a = 0 and edit_row adds 0 to w.float(), so the model is untouched. alpha1_equals_patched proves it
     with torch.equal on the float32 logits for every set and branch.
   - Random payload: built exactly as in run1c.py process_ouro.
   - Subset op: w + (alpha - 1)(c_S - c_S_clean) is an exact decomposition at fixed rms, not a linearization. It
     equals the brief's definition: kept units interpolate and the rest stay at alpha 1. g * u / rms(u) matches
     OuroRMSNorm, with no bias on down_proj or o_proj.
   - "dall reduces to all" is a valid check of the implementation (W, g, eps, rms, (t, l) keys, clean records).
     It does not check the subset index map, because the dall path builds its own range. See N1.
2. T1 to T5.
   - T3, T4 and T5 are proven by bit-exact G1 gates: pc_exact, sz_exact, donor_loop1_state, rnd_loop1_state and
     set_counts. All pass on the fake.
   - T1 for Part A is asserted. adapters/ouro.py:225-235 runs use_cache=False with cache_position = arange(T), and
     _check_T is tested, but only while Units31 is installed. This is acceptable because Part A has no decode.
   - T1 for Part B is planned only.
   - T2: equal length is asserted, and mask_report is written and tested. The token print at r and at the source
     span is Phase 2 and is not proven yet.
3. Identity gates. Every bit-exact gate compares tensors of the same dtype and the same path: bf16 payloads, and
   float32 logits from one adapter. None repeats the RA-28 failure. The only gates that can fail for a dtype
   reason are B1 and B2.
4. Risks.
   - (a) Confirmed: the brief's clause is inverted. Concentrated means share(top, 0) >= 0.5 and the 90 percent
     interval of top - rand excludes 0. Spread means share(top, 0) < 0.5. Record this in section 8 as a correction
     of the brief.
   - (b) See B1.
   - (c) Reject ||c_i|| on the clean prefill. Top by contribution norm removes much more write mass than a random
     5 percent, so "top changes accuracy less than rand" fails by construction. Use a fixed set from Part A
     instead: rank neurons (heads) by how many Part A donor items, facts and addition, put them in their top set,
     with ties broken by summed score. Freeze it to JSON before Part B. Print ||c_S|| of top and rand per alpha.
   - (d) Resolved. The ra31 tree copy (md5 3b6e3153) is the cluster bundle's copy
     (loopsteer_wt cluster/pilot/tree/prod/steer_gen.py, commit 662fd94). Compared with the e79de96 copy (md5
     0a9cd5e1), it only adds ouro_2_6b_think registry and settings entries, with no change on the 1.4B base path.
     loopedcot/prod (4078a6b3) is a different development copy and must not be used. Phase 3 must confirm against
     the file_md5 in the r00 meta. The token-by-token identity arm is the proof.
   - (e) Addition endpoints are fixed by identity: alpha 0 = 1.41 (run 1c hold) and alpha 1 = 0.81 (E0). On facts
     they are 1.28 and 0.00. "Falls to 0 by alpha 1" is therefore not a test. The falsifiable parts are:
     - shape: chord deviation at 0.25, 0.5 and 0.75;
     - "stays above": for alpha in {1.25, 1.5, 2}, NLD(all) - NLD(E0) has a 90 percent interval inside
       [-0.1, 0.1];
     - attention flat: every alpha within [-0.1, 0.1] of E0.
     Use these on both datasets.
5. Cost: consistent.
   - Forwards per item: 10 base + 66 arms = 76, asserted by the test.
   - Full 1.4B run: 530 x 76 x 0.09 to 0.12 s = 1.0 to 1.34 card-hours.
   - dall MLP adds about 72 full float32 copies of W per forward, a few ms. Write norms add about 150 host syncs
     per arm, a few ms.
   - The figures are an estimate until G5 measures them.

## Non-blocking

N1. Add a test: "top" with every unit listed explicitly gives the same result as dall, bit-exact. This covers the
    MLP flat-index map (u // I, u % I), which no current test exercises.
N2. Plan section 6 lists first_site_scale and sets_act in G1, but the code returns them in `info`. Phase 2 must
    state which keys exit 3.
N3. ra31.py:219-220 raises on unequal length. Brief T2 says drop and count. Phase 2 must catch it as a counted drop.
    This matters for the 2.6B items only.
N4. G3 gates on the mean. On the same card and stack the gap should be exactly 0. Also print the number of items
    with a nonzero gap. A nonzero count shows a stack difference, not a 0.005 tolerance question.
N5. Section 8 says "threshold reading if the interval excludes 0 at two of three points". That shows nonlinearity
    only. Call it "not linear", and call it a threshold only when the deviation sign is the same at all three
    points.
N6. The schedule rule "drop any arm flat within its interval" would drop the attention arms, and flat attention is
    the prediction. Keep attention "all". Drop attention top and rand if "all" is flat. This saves about 25 percent
    of card time.
N7. ra31.py:291 stores four 6,758-index lists per item, about 85 MB of JSON for 530 items. Store the top sets only.
    The rand sets come back from the seed.
N8. share(top, alpha) uses the lerp "all" as its denominator. At alpha 0 and 2, also print it with dall as the
    denominator, so that both sides of the ratio use the same op.
