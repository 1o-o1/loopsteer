"""Correctness checks of the cluster smoke (cluster/env/smoke_cluster.sh). The smoke proves the experiment code runs
correctly on the cluster; it does not ask for our numbers bit for bit (that comparison is a report only).

Pass rule: every task line of the two smoke lists left its done marker (it exited 0), every pipeline wrote its
records and results.csv, and the run's own harness checks hold:
  k17_varbench (Ouro-1.4B, 3 items)  hook identity (zero edit bit identical at every site, wrapper and repeat bit
                                     identical, hook call counts as expected), positive control exact on every item,
                                     structural zero exact, edit and knockout placement, causality rows all ok, the
                                     merge union equals the planned items on both datasets with no duplicate
  steer_gen a1 GSM8K (3 items)       the four gates of checks mode (zero edit: inert hooks bit identical, prefill bit
                                     identical; placement; cache equivalence; identities), row r00 exit 0 with cells
                                     written, items.json with scored items, analysis.json
  families (smoke_fam.tsv, 3 items)  raven k17_public mcl_nr D1 and Parcae k17_parcae parcae-370m D1: every
                                     hook_identity entry all_ok and zero edit identical, every pc_final, sz, pc_site and
                                     sc_harness entry all_exact; IFM k17_freeze_ifm D1: zero edit, freeze without edit,
                                     frozen source attention, PC, SZ and readout identities all exact; McLeish
                                     generation (3 problems): every checks_mcl case passes, row r00 exit 0 with cells
                                     written. SMOKE_VALUES.json
                                     holds no reference values for these: their results.csv values are reported only.

Cross-host gate (with --values): the e and margin values of both pipelines must lie within 0.001 (absolute) of the
recorded smoke's values (cluster/SMOKE_VALUES.json, bound to SMOKE_HASHES.txt by the source value hashes; the fields
are named in cluster/smoke_hash.py: e = results.csv estimate, random_control_estimate, positive_control_value;
margins = logit_B - logit_A of the vb_ouro.json items and rows). Every other field is reported, not gated.

    python cluster/env/smoke_check.py --root <smoke output tree> --lists <jobs dir> [--report <json>]
        [--values cluster/SMOKE_VALUES.json --hashes cluster/SMOKE_HASHES.txt]
        [--out-root OUT_ROOT [--override-reason TEXT]]
Exit 0 when every check passes, 1 when a correctness (identity) check fails, 3 when only the cross-host tolerance
fails. With --out-root the decision is written there: _smoke.ok on a pass; _smoke_fail.txt on a failure; with a
non-empty --override-reason (SMOKE_OVERRIDE_REASON in smoke_cluster.sh) a tolerance failure still writes _smoke.ok and
$OUT_ROOT/_smoke_override.txt (reason plus the full diff, also printed), exit 0. Identity failures are never overridden.
"""
import argparse
import csv
import glob
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smoke_hash  # noqa: E402  (cluster/smoke_hash.py)

VB = "B/k17_varbench/ouro"
GEN = "A/steer_gen/a1_gsm8k"
FAM_RAVEN = "B/k17_public/mcl_nr_D1"
FAM_PARCAE = "B/k17_parcae/parcae-370m"
FAM_IFM = "C/k17_freeze_ifm/ifm_p1_ouro_D1"
FAM_MCL = "A/steer_gen_raven/mcleish_smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lists", required=True, help="folder with smoke_g1_copy.tsv and smoke_g2_gen.tsv")
    ap.add_argument("--report", default=None)
    ap.add_argument("--values", default=None, help="reference e and margin values (cluster/SMOKE_VALUES.json)")
    ap.add_argument("--hashes", default=None, help="cluster/SMOKE_HASHES.txt (binds the reference values)")
    ap.add_argument("--out-root", default=None, help="write _smoke.ok / _smoke_fail.txt / _smoke_override.txt here")
    ap.add_argument("--override-reason", default="", help="operator reason: pass despite a tolerance miss")
    a = ap.parse_args()
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    def jload(rel):
        p = os.path.join(a.root, rel)
        try:
            return json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError) as ex:
            ck("readable " + rel, False, repr(ex)[:200])
            return None

    def get(d, *ks):
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    # every stage exited 0: the array template touches $OUT_ROOT/_done/<id> only after the task line succeeded
    for g in [g for g in ("g1_copy", "g2_gen", "fam") if os.path.exists(os.path.join(a.lists, "smoke_%s.tsv" % g))]:
        for ln in open(os.path.join(a.lists, "smoke_%s.tsv" % g), encoding="utf-8"):
            jid = ln.split("\t", 1)[0]
            if jid:
                ck("stage %s exited 0" % jid, os.path.exists(os.path.join(a.root, "_done", jid)))

    def results(rel):
        p = os.path.join(a.root, rel, "results.csv")
        n = len(list(csv.DictReader(open(p, encoding="utf-8")))) if os.path.exists(p) else 0
        ck("results.csv rows " + rel, n > 0, "%d rows" % n)

    def report_values(rel):
        """results.csv e values of a family without reference values (reported, not gated)."""
        p = os.path.join(a.root, rel, "results.csv")
        if not os.path.exists(p):
            return []
        return [{k_: r.get(k_) for k_ in ("dataset", "arm", "site", "loop", "metric", "n_items", "estimate")}
                for r in csv.DictReader(open(p, encoding="utf-8"))]

    # ---- k17_varbench
    c = jload(VB + "/checks.json")
    k = "ouro"
    if c is not None:
        hi = get(c, "hook_identity", k) or {}
        for f in ("identity_ok", "call_counts_ok", "zero_add_every_site_bit_identical", "wrapper_bit_identical",
                  "repeat_bit_identical"):
            ck("varbench hook_identity." + f, hi.get(f) is True, str(hi.get(f)))
        ck("varbench hook_identity.all_ok not false", hi.get("all_ok") is not False, str(hi.get("all_ok")))
        for name, d in (("pc_final", get(c, "pc_final", k)), ("structural_zero", get(c, "structural_zero", k)),
                        ("placement_vbd.pc_final", get(c, "placement_vbd", k, "pc_final")),
                        ("placement_vbd.structural_zero", get(c, "placement_vbd", k, "structural_zero"))):
            if name.startswith("placement_vbd") and d is None:
                continue
            ok = isinstance(d, dict) and d.get("n", 0) > 0 and d.get("exact") == d.get("n")
            ck("varbench %s exact" % name, ok, "%s of %s" % (get(d, "exact"), get(d, "n")))
        ck("varbench edit_placement.ok", get(c, "edit_placement", k, "ok") is True)
        for side in ("direct", "relay"):
            v = get(c, "knockout_placement", k, side, "ok")
            if v is not None:
                ck("varbench knockout_placement.%s.ok" % side, v is True)
        cz = get(c, "causality", k) or {}
        ck("varbench causality rows_ok == rows", cz.get("rows") and cz.get("rows_ok") == cz.get("rows"),
           "%s of %s" % (cz.get("rows_ok"), cz.get("rows")))
        ck("varbench items kept", (get(c, "items", k, "kept") or 0) > 0, str(get(c, "items", k, "kept")))
    m = jload(VB + "/merge_ouro.json")
    if m is not None:
        for ds in ("varbench", "vbd"):
            d = m.get(ds) or {}
            ck("varbench merge %s union equals planned" % ds, d.get("union_equals_planned") is True and d.get("duplicates") == 0,
               "planned %s merged %s duplicates %s" % (d.get("planned"), d.get("merged"), d.get("duplicates")))
    ck("varbench records vb_ouro.json", os.path.exists(os.path.join(a.root, VB, "vb_ouro.json")))
    results(VB)

    # ---- steer_gen a1 GSM8K
    g = jload(GEN + "/chk/checks_a1.json")
    if g is not None:
        for gate in ("gate_zero_edit", "gate_placement", "gate_cache_equiv", "gate_identities"):
            ck("steer_gen %s.pass" % gate, get(g, gate, "pass") is True)
        for f in ("inert_hooks_bit_identical", "prefill_bit_identical"):
            ck("steer_gen gate_zero_edit." + f, get(g, "gate_zero_edit", f) is True)
    it = jload(GEN + "/items.json")
    if it is not None:
        ck("steer_gen items scored", len(it.get("scored") or []) > 0, "%d scored" % len(it.get("scored") or []))
    sj = jload(GEN + "/r00/steer_job.json")
    if sj is not None:
        ck("steer_gen r00 exit 0 (no failed row)", sj.get("exit") == 0, "exit %s" % sj.get("exit"))
    ck("steer_gen r00 cells written", bool(glob.glob(os.path.join(a.root, GEN, "r00", "cells_*.jsonl"))))
    ck("steer_gen analysis.json", os.path.exists(os.path.join(a.root, GEN, "analysis.json")))
    results(GEN)

    # ---- the further families (smoke_fam.tsv): in-run identities gated, values reported
    fam_values = {}
    if os.path.exists(os.path.join(a.lists, "smoke_fam.tsv")):
        for fam, rel in (("raven", FAM_RAVEN), ("parcae", FAM_PARCAE)):
            c = jload(rel + "/checks.json")
            if c is not None:
                hi = c.get("hook_identity") or {}
                ck("%s hook_identity present" % fam, bool(hi), str(sorted(hi)))
                for m, d in sorted(hi.items()):
                    ck("%s hook_identity.%s.all_ok" % (fam, m), isinstance(d, dict) and d.get("all_ok") is True)
                    ck("%s hook_identity.%s.b_zero_edit_identity" % (fam, m),
                       isinstance(d, dict) and d.get("b_zero_edit_identity") is True)
                for sec in ("pc_final", "sz", "pc_site", "sc_harness"):
                    ent = c.get(sec) or {}
                    if sec in ("pc_final", "sz"):
                        ck("%s %s present" % (fam, sec), bool(ent))
                    for k_, d in sorted(ent.items()):
                        ck("%s %s.%s.all_exact" % (fam, sec, k_), isinstance(d, dict) and d.get("all_exact") is True)
            results(rel)
            fam_values[fam] = report_values(rel)
        c = jload(FAM_IFM + "/checks.json")
        if c is not None:
            for f in ("zero_edit_bit_identical", "freeze_no_edit_bit_identical", "source_attn_frozen_e_zero_bit_exact",
                      "pc_exact_all", "sz_exact_all", "readout_identity_all"):
                ck("ifm freeze %s" % f, c.get(f) is True, str(c.get(f)))
        results(FAM_IFM)
        fam_values["ifm"] = report_values(FAM_IFM)
        if any(ln.startswith("smoke.fam.f_mcleish.") for ln in open(os.path.join(a.lists, "smoke_fam.tsv"), encoding="utf-8")):
            c = jload(FAM_MCL + "/chk/checks_mcl.json")
            if c is not None:
                ck("mcleish checks_mcl.pass", c.get("pass") is True)
                for k_, d in sorted((c.get("cases") or {}).items()):
                    if isinstance(d, dict) and "pass" in d:
                        ck("mcleish checks_mcl.cases.%s.pass" % k_, d["pass"] is True)
            sj = jload(FAM_MCL + "/r00/steer_job.json")
            if sj is not None:
                ck("mcleish r00 exit 0 (no failed row)", sj.get("exit") == 0, "exit %s" % sj.get("exit"))
            ck("mcleish r00 cells written", bool(glob.glob(os.path.join(a.root, FAM_MCL, "r00", "cells_*.jsonl"))))

    bad = [x for x in checks if not x["ok"]]
    rep = {"ok": not bad, "identity_ok": not bad, "n_checks": len(checks), "failed": bad, "checks": checks,
           "family_values_reported_not_gated": fam_values}
    tol = None
    if a.values:
        tol = smoke_hash.gate(a.values, a.root, a.hashes)
        rep["tolerance"] = {k: v for k, v in tol.items() if k != "diff"}
        rep["tolerance_ok"] = tol["within_tolerance"]
        rep["ok"] = rep["ok"] and tol["within_tolerance"]
    code = 1 if bad else (3 if tol is not None and not tol["within_tolerance"] else 0)
    reason = (a.override_reason or "").strip()
    rep["override_reason"] = reason if code == 3 and reason else None
    if a.report:
        json.dump(rep, open(a.report, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"ok": rep["ok"], "identity_ok": rep["identity_ok"], "tolerance_ok": rep.get("tolerance_ok"),
                      "n_checks": len(checks), "failed": [x["check"] for x in bad],
                      "tolerance": {k: rep["tolerance"][k] for k in ("n_ok", "n_fail", "max_abs_diff", "counts")}
                      if tol is not None else None}, indent=1))
    if a.out_root:
        code = decide(a.out_root, a.root, code, bad, tol, reason)
    sys.exit(code)


def decide(out_root, smoke_tree, code, bad, tol, reason):
    """Write the smoke decision into OUT_ROOT. Identity failures (code 1) are final; a tolerance failure (code 3)
    passes only with an operator reason, which is recorded with the full diff in _smoke_override.txt."""
    for f in ("_smoke.ok", "_smoke_fail.txt", "_smoke_override.txt", "_smoke_tolerance_diff.tsv"):
        if os.path.exists(os.path.join(out_root, f)):
            os.remove(os.path.join(out_root, f))
    if tol is not None:
        open(os.path.join(out_root, "_smoke_tolerance_diff.tsv"), "w", encoding="utf-8", newline="\n").write(tol["diff"])
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if code == 1:
        lines = ["identity (correctness) checks failed; SMOKE_OVERRIDE_REASON cannot override them:"]
        lines += ["  %s (%s)" % (x["check"], x["detail"]) for x in bad]
        if reason:
            lines.append("override reason given and ignored: %s" % reason)
        open(os.path.join(out_root, "_smoke_fail.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 1
    ok = {"passed_at": now, "node": socket.gethostname(), "smoke_tree": smoke_tree,
          "rule": "identity checks bit-exact (smoke_check.py) and e / margins within %g of SMOKE_VALUES.json"
          % smoke_hash.GATE_TOL, "override": None}
    if tol is not None:
        ok["tolerance"] = {k: tol[k] for k in ("n_ok", "n_fail", "max_abs_diff", "counts")}
    if code == 3:
        head = ["cross-host tolerance failed: %d of %d reference e / margin values outside |new - ref| <= %g "
                "(max abs diff %.6g)%s" % (tol["n_fail"], tol["n_reference"], tol["tol_abs"], tol["max_abs_diff"],
                                           "; binding: " + "; ".join(tol["binding_problems"]) if tol["binding_problems"] else "")]
        if not reason:
            msg = head + ["set SMOKE_OVERRIDE_REASON='<text>' to pass anyway (recorded in _smoke_override.txt)",
                          "full diff: %s" % os.path.join(out_root, "_smoke_tolerance_diff.tsv")]
            open(os.path.join(out_root, "_smoke_fail.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(msg) + "\n")
            print("\n".join(msg))
            return 3
        txt = "\n".join(["SMOKE OVERRIDE %s" % now, "reason: %s" % reason] + head + ["full diff:", tol["diff"]])
        open(os.path.join(out_root, "_smoke_override.txt"), "w", encoding="utf-8", newline="\n").write(txt)
        print(txt)
        ok["override"] = {"reason": reason, "file": "_smoke_override.txt"}
    json.dump(ok, open(os.path.join(out_root, "_smoke.ok"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    main()
