"""Between phase W and phase M of k17_varbench.py (plan sections 9 and 10): append the phase-W slot selection of one
model to section 13 of a copy of the plan, so that phase M launches on a plan file that records the slots before the
first phase-M row. The copy is written into the model's output folder (plan_M_<model>.md); everything outside
section 13 stays byte-equal to the plan, so its plan_frozen_sha256 equals the phase-W launch. Phase M then passes
--plan <copy> --plan_sha $(sha256sum <copy>).

    python cluster/vb_slots.py --plan PLAN.md --out OUT_DIR --model ouro
"""
import argparse
import hashlib
import json
import os
import time


def append_slots(plan_text, entry):
    """Insert entry at the end of section 13 (before the next '## ' heading, or at the end of the file)."""
    lines = plan_text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## 13."))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[:end]) + entry + "".join(lines[end:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    win = json.load(open(os.path.join(a.out, "window_%s.json" % a.model), encoding="utf-8"))
    slots = win.get("selected") or []
    informative = win.get("selected_informative")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = ("- %s, cluster run, model %s: phase W slots selected %s (informative %s; n items %s), from "
             "window_%s.json; appended by cluster/vb_slots.py before phase M.\n"
             % (stamp, a.model, json.dumps(slots), json.dumps(informative), win.get("n_items"), a.model))
    raw = open(a.plan, encoding="utf-8", newline="").read()
    out = append_slots(raw, entry)
    dst = os.path.join(a.out, "plan_M_%s.md" % a.model)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(json.dumps({"plan_M": os.path.basename(dst), "sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
                      "slots": slots}))


if __name__ == "__main__":
    main()
