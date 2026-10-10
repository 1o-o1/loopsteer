"""Hashes of the smoke outputs that a rerun can reproduce (RA-20 Amendment 2 item 7).

A raw file hash changes on every rerun because records hold timings, process data and, where the gradient score
runs, backward-pass values that are not bit-reproducible on a GPU. This script writes, per output file, the raw
sha256 and a value sha256: for .json / .jsonl the sha256 of the canonical JSON (sorted keys, no spaces) after the
fields below are removed; for .pt the sha256 of every tensor's bytes in key order; other files the raw sha256.

    python cluster/smoke_hash.py write  SMOKE_ROOT [--out cluster/SMOKE_HASHES.txt]
    python cluster/smoke_hash.py check  SMOKE_ROOT [--hashes cluster/SMOKE_HASHES.txt]   (value hashes, rerun)
    python cluster/smoke_hash.py grad   DIR_FLAG_OFF DIR_FLAG_ON                          (--no_gradient_rank test)
    python cluster/smoke_hash.py check  SMOKE_ROOT --prefix out_new/ --present_only        (cluster smoke: report only)
    python cluster/smoke_hash.py compare REF_DIR NEW_DIR [--rel 1e-3 --abs 1e-4] [--diff diff.tsv]

`compare` is the cross-hardware comparison (another GPU type does not reproduce float values bit for bit): every
.json / .jsonl record of REF_DIR is paired with the same file of NEW_DIR after the timing, process and gradient fields
are removed; a float pair agrees when |a - b| <= abs or |a - b| <= rel * max(|a|, |b|) (defaults rel 1e-3, abs 1e-4);
integers, strings, ids, counts and generated tokens are compared exactly but their differences are reported, not
counted as out of tolerance (they may differ across hardware when a decision flips at a near-tie). The report lists
every out-of-tolerance field (file, field, expected, got, absolute difference) and the max absolute difference per file.

`grad` is the acceptance test of --no_gradient_rank: the record files of a run without the flag and a run with it
must agree on every field once the gradient-derived fields and the gradient rows (verify and search rows, named
O1|ans|<site>|<dose>, site <loop>.<layer> with an optional letter prefix) are removed; it prints what differs and the gradient keys and rows it removed.

`values` and `gate` are the cross-host gate of the cluster smoke (cluster/env/smoke_check.py). Only two kinds of
value are gated, both read from the two smoke pipelines (k17_varbench Ouro-1.4B in B/k17_varbench/ouro, steer_gen a1
GSM8K in A/steer_gen/a1_gsm8k):
  e        results.csv columns estimate, random_control_estimate and positive_control_value of every row of both
           pipelines (k17_varbench: the e / e_LO cells; steer_gen: the gold_A, gold_B and neither rates)
  margins  k17_varbench vb_ouro.json, datasets varbench and vbd, every item: the B-minus-A logit margin
           (logit_B - logit_A, the margin dm_of and Dgap of k17_varbench use) of clean_A, clean_B, each clean_A_loops
           entry, and each row's f and each of its L entries (gradient rows excluded)
A value passes when |new - reference| <= 0.001 (absolute). Every other field is reported only (`compare`, `check`).
    python cluster/smoke_hash.py values SMOKE_ROOT --hashes cluster/SMOKE_HASHES.txt --prefix out_new/ --out cluster/SMOKE_VALUES.json
    python cluster/smoke_hash.py gate   NEW_SMOKE_ROOT --values cluster/SMOKE_VALUES.json [--diff gate_diff.tsv]
`values` writes the reference values from the smoke tree whose value hashes SMOKE_HASHES.txt records and refuses when
a source file's value hash differs from its SMOKE_HASHES.txt line, so SMOKE_VALUES.json holds exactly the recorded
smoke's values (the source value sha256 sits in the file; `gate` checks them against SMOKE_HASHES.txt again).
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import sys

# timing, process and environment fields (they change on every rerun)
TIMING = {"seconds", "seconds_total", "load_seconds", "t", "t_fwd", "t_item", "t_fwd_plain", "seconds_W", "seconds_M",
          "forwards", "forwards_W", "forwards_M", "forwards_run", "n_fwd", "nvsmi", "scan_seconds", "table_seconds", "select_seconds",
          "merged_shards", "plan_launch", "phase_runs", "hashed_at_utc", "process_minutes", "deadline_used",
          "deadline_epoch", "written_at", "peak_gb", "generate_seconds", "tokens_per_s", "oom_events",
          "seconds_this_process", "gpu_minutes", "gpu_minutes_including_debug_runs", "seconds_run", "launch",
          "co_tenants", "gpu_mem_start", "gpu_mem_end", "gpu_mem_total_gb", "gpu_mem_free_gb", "env", "argv",
          "only_rows", "heartbeat", "pid", "cost_fwd_equiv", "elapsed"}
# fields derived from the backward pass (gradient score, path-integrated gain, their verify and search rows and the
# selections made from them); absent when --no_gradient_rank is on
GRADIENT = {"G", "G_PI", "Ga", "PIG", "G_bf16", "t_grad", "t_pig", "t_pi", "pig_a1_equals_B", "pig_ids_differ_only_at_src",
            "ver_S", "ver_Sp", "verify", "pig_top", "rnd_site", "uni_site", "grad_identical", "grad_run_bit_identical",
            "pi_passes", "dev_search", "dev_tuned", "e8", "carry_gain", "full_search",
            "n_rows"}                     # n_rows: the row count of an item, which includes its gradient rows
GRAD_ROW = re.compile(r"^O1\|ans\|[A-Za-z]*\d+\.\d+\|")
SKIP_FILES = ("*.log", "logs/*", "*/logs/*", "heartbeat_*", "*/heartbeat_*", "proc_*", "*/proc_*", "summary.md",
              "*/summary.md")


def strip(o, drop, removed=None):
    """Remove the keys in drop at every depth (removed collects the names met)."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in drop:
                if removed is not None:
                    removed.add(k)
                continue
            out[k] = strip(v, drop, removed)
        return out
    if isinstance(o, list):
        return [strip(v, drop, removed) for v in o]
    return o


def align_rows(x, y, removed, path=""):
    """In every "rows" dict, drop the rows that only the run without the flag has and that are gradient rows
    (GRAD_ROW: verify and search rows); every other row must be present in both and equal."""
    if isinstance(x, dict) and isinstance(y, dict):
        if path.endswith("/rows"):
            for k in [k for k in x if k not in y and GRAD_ROW.match(str(k))]:
                removed.add("row " + re.sub(r"[A-Za-z]*\d+\.\d+", "<site>", str(k)))
                del x[k]
        for k in x:
            if k in y:
                align_rows(x[k], y[k], removed, path + "/" + str(k))
    elif isinstance(x, list) and isinstance(y, list):
        for u, v in zip(x, y):
            align_rows(u, v, removed, path)


def load(p):
    if p.endswith(".jsonl"):
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return json.load(open(p, encoding="utf-8"))


def value_hash(p, drop):
    if p.endswith((".json", ".jsonl")):
        try:
            o = strip(load(p), drop)
        except ValueError:
            return hashlib.sha256(open(p, "rb").read()).hexdigest()
        return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if p.endswith(".pt"):
        import torch
        h = hashlib.sha256()

        def walk(x):
            if isinstance(x, torch.Tensor):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(x, dict):
                for k in x:
                    h.update(str(k).encode())
                    walk(x[k])
            elif isinstance(x, (list, tuple)):
                for y in x:
                    walk(y)
            else:
                h.update(repr(x).encode())
        walk(torch.load(p, map_location="cpu"))
        return h.hexdigest()
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def files(root):
    out = []
    for r, dirs, fs in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("logs", "__pycache__", "_done"))
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/")
            out.append(rel)
    return out


def write(root, out):
    drop = TIMING | GRADIENT
    lines = ["# Smoke output hashes. Columns: value sha256, raw sha256, path (relative to the smoke root).",
             "# value sha256: canonical JSON (sorted keys, no spaces) after removing the timing, process and environment",
             "#   fields %s" % ", ".join(sorted(TIMING)),
             "#   and the gradient-derived fields %s" % ", ".join(sorted(GRADIENT)),
             "#   (rows stay: the job list runs with --no_gradient_rank, so cluster records hold no gradient rows; the",
             "#   smoke runs without the flag, out_off and out_orig, also hold verify rows at gradient-ranked sites); .pt:",
             "#   tensor bytes;",
             "#   other files: raw sha256. Recompute and compare: python cluster/smoke_hash.py check SMOKE_ROOT"]
    for rel in files(root):
        p = os.path.join(root, rel)
        lines.append("%s  %s  %s" % (value_hash(p, drop), hashlib.sha256(open(p, "rb").read()).hexdigest(), rel))
    open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(json.dumps({"files": sum(1 for x in lines if not x.startswith("#")), "out": os.path.basename(out)}))


def check(root, hashes, prefix="", present_only=False):
    drop = TIMING | GRADIENT
    bad, n = [], 0
    for ln in open(hashes, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        vh, _rh, rel = ln.rstrip("\n").split("  ", 2)
        if prefix:
            if not rel.startswith(prefix):
                continue
            rel = rel[len(prefix):]
        p = os.path.join(root, rel)
        if present_only and not os.path.exists(p):
            continue
        n += 1
        if any(fnmatch.fnmatch(rel, g) for g in SKIP_FILES):
            continue
        if not os.path.exists(p):
            bad.append((rel, "missing"))
        elif value_hash(p, drop) != vh:
            bad.append((rel, "value hash differs"))
    print(json.dumps({"files": n, "differ": bad[:50], "n_differ": len(bad)}, indent=1))
    return 0 if not bad else 1


def grad(a, b):
    """a: run without --no_gradient_rank, b: with it. Record files only (summaries are derived)."""
    recs = [r for r in files(a) if fnmatch.fnmatch(os.path.basename(r), "*.json") or r.endswith(".jsonl")]
    recs = [r for r in recs if any(fnmatch.fnmatch(os.path.basename(r), g) for g in
                                   ("vb_*.json", "res_*.json", "window_*.json", "items_*.jsonl", "selection_*.json"))]
    rep = {"files": {}, "removed_from_flag_off": [], "removed_from_flag_on": []}
    drop = TIMING | GRADIENT | {"config", "config_sha256", "no_gradient_rank", "code_sha256",
                                                 "plan_sha256", "plan_sha_arg", "plan_frozen_sha256", "plan_code_sha256"}
    ra, rb = set(), set()
    for rel in recs:
        pa, pb = os.path.join(a, rel), os.path.join(b, rel)
        if not os.path.exists(pb):
            rep["files"][rel] = "missing with the flag"
            continue
        x, y = strip(load(pa), drop, ra), strip(load(pb), drop, rb)
        align_rows(x, y, ra)
        rep["files"][rel] = "equal" if x == y else first_diff(x, y)
    rep["removed_from_flag_off"], rep["removed_from_flag_on"] = sorted(ra), sorted(rb)
    rep["all_equal"] = all(v == "equal" for v in rep["files"].values()) and bool(rep["files"])
    print(json.dumps(rep, indent=1))
    return 0 if rep["all_equal"] else 1


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def tol_walk(x, y, path, rel, ab, out):
    """Collect (path, kind, expected, got, absdiff): kind float (out of tolerance), float_ok, exact (other values)."""
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y), key=str):
            if k not in y or k not in x:
                out.append((path + "/" + str(k), "exact", "present" if k in x else "absent",
                            "present" if k in y else "absent", None))
            else:
                tol_walk(x[k], y[k], path + "/" + str(k), rel, ab, out)
        return
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path, "exact", "len %d" % len(x), "len %d" % len(y), None))
        for i, (u, v) in enumerate(zip(x, y)):
            tol_walk(u, v, "%s[%d]" % (path, i), rel, ab, out)
        return
    if _num(x) and _num(y) and (isinstance(x, float) or isinstance(y, float)):
        if x != x and y != y:                                   # both NaN
            return
        d = abs(float(x) - float(y))
        bad = d > ab and d > rel * max(abs(float(x)), abs(float(y)))
        out.append((path, "float" if bad else "float_ok", x, y, d))
        return
    if x != y:
        out.append((path, "exact", x, y, None))


def compare(ref, new, rel=1e-3, ab=1e-4, out_path=None):
    """Field-level comparison of two smoke trees with a float tolerance (see the module docstring)."""
    drop = TIMING | GRADIENT
    recs = [r for r in files(ref) if r.endswith((".json", ".jsonl")) and not any(fnmatch.fnmatch(r, g) for g in SKIP_FILES)]
    rep = {"rule": "float: |a-b| <= %g or |a-b| <= %g * max(|a|,|b|); ints, strings, counts, tokens: exact, reported only"
           % (ab, rel), "files": {}, "n_files": len(recs), "n_float_compared": 0, "n_out_of_tolerance": 0,
           "n_exact_differs": 0, "missing_in_new": []}
    lines = ["file\tfield\texpected\tgot\tabs_diff\tkind"]
    for r in recs:
        pn = os.path.join(new, r)
        if not os.path.exists(pn):
            rep["missing_in_new"].append(r)
            continue
        try:
            x, y = strip(load(os.path.join(ref, r)), drop), strip(load(pn), drop)
        except ValueError:
            continue
        out = []
        tol_walk(x, y, "", rel, ab, out)
        fl = [o for o in out if o[1] in ("float", "float_ok")]
        bad = [o for o in out if o[1] == "float"]
        ex = [o for o in out if o[1] == "exact"]
        rep["n_float_compared"] += len(fl)
        rep["n_out_of_tolerance"] += len(bad)
        rep["n_exact_differs"] += len(ex)
        rep["files"][r] = {"floats": len(fl), "out_of_tolerance": len(bad), "exact_differs": len(ex),
                           "max_abs_diff": max([o[4] for o in fl] or [0.0])}
        for o in bad + ex:
            lines.append("%s\t%s\t%r\t%r\t%s\t%s" % (r, o[0], o[2], o[3], "" if o[4] is None else "%.3g" % o[4],
                                                    "out_of_tolerance" if o[1] == "float" else "exact_differs"))
    rep["within_tolerance"] = rep["n_out_of_tolerance"] == 0 and not rep["missing_in_new"]
    if out_path:
        open(out_path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=1))
    return rep


# ------------------------------------------------------------------ cross-host gate (e and margins)
GATE_TOL = 1e-3
VB_DIR, GEN_DIR = "B/k17_varbench/ouro", "A/steer_gen/a1_gsm8k"
GATE_SOURCES = (VB_DIR + "/results.csv", VB_DIR + "/vb_ouro.json", GEN_DIR + "/results.csv")
E_COLS = ("estimate", "random_control_estimate", "positive_control_value")
ID_COLS = ("grid", "card", "script", "model", "dataset", "arm", "site", "loop", "strength", "metric", "endpoint",
           "item_set", "source_key")
GATE_RULE = ("e: results.csv estimate, random_control_estimate, positive_control_value (k17_varbench ouro, steer_gen "
             "a1 gsm8k); margins: logit_B - logit_A of vb_ouro.json items clean_A, clean_B, clean_A_loops[j], "
             "rows/<key>/f, rows/<key>/L[j] (varbench, vbd; gradient rows excluded); pass when |new - ref| <= %g" % GATE_TOL)


def _cell(v):
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def gate_values(root):
    """{key: value} of the gated e and margin values of a smoke tree (keys name pipeline, row or item, and field)."""
    vals = {}

    def put(k, v):
        if k in vals:
            raise ValueError("duplicate gate key %s" % k)
        vals[k] = v

    for d, tag in ((VB_DIR, "k17_varbench"), (GEN_DIR, "steer_gen")):
        p = os.path.join(root, d, "results.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    idk = "|".join(r.get(c) or "" for c in ID_COLS)
                    for c in E_COLS:
                        if c in r:
                            put("e|%s|%s|%s" % (tag, idk, c), _cell(r[c]))
    p = os.path.join(root, VB_DIR, "vb_ouro.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))

        def mg(v):
            ok = isinstance(v, list) and len(v) >= 2 and all(_num(x) for x in v[:2])
            return round(float(v[1]) - float(v[0]), 6) if ok else None
        for ds in ("varbench", "vbd"):
            for pid, it in sorted(((d.get(ds) or {}).get("items") or {}).items()):
                pre = "margin|k17_varbench|%s|%s|" % (ds, pid)
                for f in ("clean_A", "clean_B"):
                    if f in it:
                        put(pre + f, mg(it[f]))
                for j, v in enumerate(it.get("clean_A_loops") or []):
                    put(pre + "clean_A_loops[%d]" % j, mg(v))
                for rk, rv in sorted((it.get("rows") or {}).items()):
                    if GRAD_ROW.match(rk) or not isinstance(rv, dict):
                        continue
                    if "f" in rv:
                        put(pre + "rows/%s/f" % rk, mg(rv["f"]))
                    for j, v in enumerate(rv.get("L") or []):
                        put(pre + "rows/%s/L[%d]" % (rk, j), mg(v))
    return vals


def hash_lines(hashes):
    out = {}
    for ln in open(hashes, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        vh, _rh, rel = ln.rstrip("\n").split("  ", 2)
        out[rel] = vh
    return out


def write_values(root, hashes, prefix, out):
    """SMOKE_VALUES.json from the recorded smoke tree; refuses unless every source has its SMOKE_HASHES.txt value hash."""
    rec, src = hash_lines(hashes), {}
    for s in GATE_SOURCES:
        vh = value_hash(os.path.join(root, s), TIMING | GRADIENT)
        if rec.get(prefix + s) != vh:
            raise SystemExit("%s: value sha256 %s is not the SMOKE_HASHES.txt value %s" % (s, vh, rec.get(prefix + s)))
        src[s] = {"value_sha256": vh, "hashes_path": prefix + s}
    vals = gate_values(root)
    doc = {"rule": GATE_RULE, "tol_abs": GATE_TOL, "sources": src, "n_values": len(vals), "values": vals}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=0, sort_keys=True)
        f.write("\n")
    print(json.dumps({"n_values": len(vals), "sources": sorted(src), "out": os.path.basename(out)}))


def gate(values_path, root, hashes=None, out_path=None):
    """Cross-host gate: every reference e and margin value against the new smoke tree (|new - ref| <= tol_abs).
    Verdicts: ok; out_of_tolerance, undefined_on_one_side, missing (fail); not_run (a steer_gen row the smoke list
    does not run), extra (only in the new tree) and binding problems are listed (binding problems fail)."""
    ref = json.load(open(values_path, encoding="utf-8"))
    tol = float(ref.get("tol_abs", GATE_TOL))
    binding = []
    if hashes:
        rec = hash_lines(hashes)
        for s, v in sorted(ref["sources"].items()):
            if rec.get(v["hashes_path"]) != v["value_sha256"]:
                binding.append("%s: SMOKE_VALUES.json source hash %s is not in SMOKE_HASHES.txt (%s)"
                               % (s, v["value_sha256"][:12], rec.get(v["hashes_path"], "no line")))
    new = gate_values(root)
    gen = os.path.join(root, GEN_DIR)
    ran = set(x for x in (os.listdir(gen) if os.path.isdir(gen) else []) if re.match(r"^r\d\d$", x))
    rows, n_ok, mx = [], 0, 0.0
    for k, x in sorted(ref["values"].items()):
        if k not in new:
            arm = k.split("|")[7] if k.startswith("e|steer_gen|") else ""
            rows.append((k, x, None, None, "not_run" if arm and arm.split(":")[0] not in ran else "missing"))
            continue
        y = new[k]
        if x is None and y is None:
            n_ok += 1
        elif x is None or y is None:
            rows.append((k, x, y, None, "undefined_on_one_side"))
        else:
            d = abs(float(y) - float(x))
            mx = max(mx, d)
            if d <= tol + 1e-12:
                n_ok += 1
            else:
                rows.append((k, x, y, d, "out_of_tolerance"))
    rows += [(k, None, new[k], None, "extra") for k in sorted(set(new) - set(ref["values"]))]
    fail = [r for r in rows if r[4] in ("out_of_tolerance", "undefined_on_one_side", "missing")]
    rep = {"rule": ref.get("rule", GATE_RULE), "tol_abs": tol, "n_reference": len(ref["values"]), "n_new": len(new),
           "n_ok": n_ok, "n_fail": len(fail), "max_abs_diff": mx, "binding_problems": binding,
           "counts": {v: sum(1 for r in rows if r[4] == v) for v in
                      ("out_of_tolerance", "undefined_on_one_side", "missing", "not_run", "extra")}}
    rep["within_tolerance"] = not fail and not binding and n_ok > 0
    lines = ["key\texpected\tgot\tabs_diff\tverdict"]
    lines += ["%s\t%r\t%r\t%s\t%s" % (k, x, y, "" if d is None else "%.6g" % d, v) for k, x, y, d, v in rows]
    rep["diff"] = "\n".join(lines) + "\n"
    if out_path:
        open(out_path, "w", encoding="utf-8", newline="\n").write(rep["diff"])
    return rep


def first_diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return "%s keys %s | %s" % (path, sorted(set(x) - set(y))[:6], sorted(set(y) - set(x))[:6])
        for k in x:
            d = first_diff(x[k], y[k], path + "/" + str(k))
            if d:
                return d
        return None
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return "%s len %d vs %d" % (path, len(x), len(y))
        for i, (u, v) in enumerate(zip(x, y)):
            d = first_diff(u, v, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    return None if x == y else "%s: %r vs %r" % (path, x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "check", "grad", "compare", "values", "gate"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--out", default="SMOKE_HASHES.txt")
    ap.add_argument("--hashes", default="SMOKE_HASHES.txt")
    ap.add_argument("--prefix", default="", help="check: only the hash lines under this path prefix (prefix removed)")
    ap.add_argument("--present_only", action="store_true", help="check: skip hash lines of files the smoke did not write")
    ap.add_argument("--rel", type=float, default=1e-3, help="compare: relative float tolerance")
    ap.add_argument("--abs", type=float, default=1e-4, help="compare: absolute float tolerance")
    ap.add_argument("--diff", default=None, help="compare, gate: write every differing field here (tab separated)")
    ap.add_argument("--values", default="SMOKE_VALUES.json", help="gate: reference e and margin values")
    a = ap.parse_args()
    if a.mode == "write":
        write(a.a, a.out)
    elif a.mode == "values":
        write_values(a.a, a.hashes, a.prefix, a.out)
    elif a.mode == "gate":
        rep = gate(a.values, a.a, a.hashes if os.path.exists(a.hashes) else None, a.diff)
        print(json.dumps({k: v for k, v in rep.items() if k != "diff"}, indent=1))
        sys.exit(0 if rep["within_tolerance"] else 1)
    elif a.mode == "check":
        sys.exit(check(a.a, a.hashes, a.prefix, a.present_only))
    elif a.mode == "compare":
        sys.exit(0 if compare(a.a, a.b, a.rel, a.abs, a.diff)["within_tolerance"] else 1)
    else:
        sys.exit(grad(a.a, a.b))


if __name__ == "__main__":
    main()
