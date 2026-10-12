"""K13 pilot: does an edit to the carried state survive the next loops, and through which channel?

Facts the models know (country -> capital, city -> country). Pair (A, B) differs only in the subject tokens.
Edit: at the end of loop j, the carried state at the subject positions moves toward B's state at the same loop:
    h <- h + alpha * (h_B - h)
Arms: single edit at loop j (each j), repeated edit at loops j..K, last-loop edit, for alpha in ALPHAS.
IFM injection twins: also an input-channel arm (the re-read embedding at the subject positions set to B's, at
every loop).
Per arm and loop: effect e_j = (m_edit - m_A) / (m_B - m_A) where m = logit(B answer token) - logit(A answer
token) at the last position, read through the model's own head after loop j. Immediate = e at the edited
loop, final = e at loop K; flip = final top-1 equals B's answer token. Items are split even/odd ids into
dev and held-out halves.

    python k13_edit.py --model ouro --out OUT     (HF Ouro-1.4B)
    python k13_edit.py --model ifm_p1_ouro_inj --out OUT   (xllm venv)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

ALPHAS = [0.5, 1.0]
CAPITALS = {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Germany": "Berlin", "Spain": "Madrid",
            "Russia": "Moscow", "Egypt": "Cairo", "England": "London", "China": "Beijing", "Greece": "Athens",
            "Austria": "Vienna", "Norway": "Oslo", "Ireland": "Dublin", "Portugal": "Lisbon", "Poland": "Warsaw",
            "Sweden": "Stockholm", "Kenya": "Nairobi", "Peru": "Lima", "Cuba": "Havana", "Thailand": "Bangkok",
            "Hungary": "Budapest", "Denmark": "Copenhagen", "Finland": "Helsinki", "Turkey": "Ankara",
            "Iran": "Tehran", "Iraq": "Baghdad", "Syria": "Damascus", "Lebanon": "Beirut", "Chile": "Santiago",
            "Argentina": "Buenos", "Colombia": "Bogota", "Mexico": "Mexico", "Canada": "Ottawa",
            "Australia": "Canberra", "India": "New", "Pakistan": "Islamabad", "Bangladesh": "Dhaka",
            "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Philippines": "Manila", "Nigeria": "Abuja",
            "Ghana": "Accra", "Ethiopia": "Addis", "Morocco": "Rabat", "Belgium": "Brussels",
            "Netherlands": "Amsterdam", "Switzerland": "Bern", "Ukraine": "Kyiv", "Romania": "Bucharest"}
CITIES = {"Paris": "France", "Rome": "Italy", "Tokyo": "Japan", "Berlin": "Germany", "Madrid": "Spain",
          "Moscow": "Russia", "Cairo": "Egypt", "London": "England", "Beijing": "China", "Athens": "Greece",
          "Vienna": "Austria", "Oslo": "Norway", "Dublin": "Ireland", "Lisbon": "Portugal", "Warsaw": "Poland",
          "Stockholm": "Sweden", "Nairobi": "Kenya", "Lima": "Peru", "Havana": "Cuba", "Bangkok": "Thailand",
          "Toronto": "Canada", "Sydney": "Australia", "Mumbai": "India", "Chicago": "America",
          "Munich": "Germany", "Barcelona": "Spain", "Venice": "Italy", "Osaka": "Japan", "Seoul": "Korea",
          "Shanghai": "China", "Istanbul": "Turkey", "Manchester": "England", "Milan": "Italy"}


def build_pairs(enc):
    """enc(text) -> ids. Pairs with equal length that differ in a contiguous span, different answers."""
    out = []
    for tmpl, table in (("The capital of {} is", CAPITALS), ("{} is a city in the country of", CITIES)):
        keys = list(table)
        for i, a in enumerate(keys):
            for b in keys[i + 1:i + 6]:
                if table[a] == table[b]:
                    continue
                ia, ib = enc(tmpl.format(a)), enc(tmpl.format(b))
                ta, tb = enc(" " + table[a], ans=True), enc(" " + table[b], ans=True)
                if len(ia) != len(ib) or ta[0] == tb[0]:
                    continue
                diff = [k for k in range(len(ia)) if ia[k] != ib[k]]
                if not diff or diff[-1] - diff[0] + 1 != len(diff):
                    continue
                out.append({"A": ia, "B": ib, "pos": diff, "tA": ta[0], "tB": tb[0], "a": a, "b": b,
                            "rel": "capital" if table is CAPITALS else "city"})
    return out


OURO_14B_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"   # cluster copy: pinned (cluster/weights.json)


class OuroEditor(object):
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo = "ByteDance/Ouro-1.4B"
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=OURO_14B_REVISION)
        self.m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                      revision=OURO_14B_REVISION).cuda().eval()
        self.K = 4
        self.W = self.m.lm_head.weight

    def enc(self, t, ans=False):
        return self.tok(t, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def run(self, ids, edit=None):
        """edit: dict loops:set, pos:list, alpha, ref:list of K [1,T,H]. Returns (states list, readout logits K x V)."""
        inner = self.m.model
        states, cnt = [], [0]

        def hook(mod, i, o):
            j = cnt[0]
            cnt[0] += 1
            if edit and (j + 1) in edit["loops"]:
                o = o.clone()
                p = edit["pos"]
                o[:, p] = o[:, p] + edit["alpha"] * (edit["ref"][j][:, p] - o[:, p])
            states.append(o.detach())
            return o
        h = inner.norm.register_forward_hook(hook)
        try:
            self.m(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            h.remove()
        z = torch.stack([torch.nn.functional.linear(s[0, -1], self.W).float() for s in states])
        return states, z


class IfmEditor(object):
    def __init__(self, name):
        import k10_ifm
        self.r = k10_ifm.make_runner(name)
        self.r.set_depth(self.r.K_trained)
        self.K = self.r.k
        self.m = self.r.m
        self.L = int(self.m.loop_end_layers) - int(self.m.loop_start_layer)
        self.inj = getattr(self.m, "loop_input_injection_module", None)

    def enc(self, t, ans=False):
        ids = self.r.tok.hf(t, add_special_tokens=False)["input_ids"]
        return ids if ans else [self.r.tok.bos_token_id] + ids

    @torch.no_grad()
    def run(self, ids, edit=None):
        r, m = self.r, self.m
        states, cnt = [], [0]
        orig = r._layer
        end = int(m.loop_end_layers) - 1

        def lay(lid, h, freq, cache, slot):
            out = orig(lid, h, freq, cache, slot)
            if lid == end:
                j = cnt[0]
                cnt[0] += 1
                if edit and edit.get("kind", "state") == "state" and (j + 1) in edit["loops"]:
                    out = out.clone()
                    p = edit["pos"]
                    out[:, p] = out[:, p] + edit["alpha"] * (edit["ref"][j][:, p] - out[:, p])
                states.append(out.detach())
            return out
        r._layer = lay
        inj_orig = None
        if edit and edit.get("kind") == "input" and self.inj is not None:
            inj_orig = self.inj.forward
            embB = edit["embB"]

            def inj_fwd(hidden, emb):
                emb = emb.clone()
                emb[:, edit["pos"]] = emb[:, edit["pos"]] + edit["alpha"] * (embB[:, edit["pos"]] - emb[:, edit["pos"]])
                return inj_orig(hidden, emb)
            self.inj.forward = inj_fwd
        try:
            x = torch.tensor([ids], device=r.dev)
            r.fwd(x, self.K, cache=None, example_ids=torch.tensor([0], device=r.dev), all_logits=True)
        finally:
            r._layer = orig
            if inj_orig is not None:
                self.inj.forward = inj_orig
        z = torch.stack([m.output(s[:, -1:], None, None)[0, -1].float() for s in states])
        return states, z

    def emb(self, ids):
        return self.m.embed(torch.tensor([ids], device=self.r.dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vendor", default=None)
    a = ap.parse_args()
    if a.vendor:
        sys.path.insert(0, a.vendor)
    ed = OuroEditor() if a.model == "ouro" else IfmEditor(a.model)
    K = ed.K
    pairs = build_pairs(ed.enc)
    arms = []
    for al in ALPHAS:
        for j in range(1, K + 1):
            arms.append(("single_L%d_a%.1f" % (j, al), {"loops": {j}, "alpha": al}))
        for j in range(1, K):
            arms.append(("repeat_L%d-%d_a%.1f" % (j, K, al), {"loops": set(range(j, K + 1)), "alpha": al}))
    has_inj = a.model != "ouro" and getattr(ed, "inj", None) is not None
    if has_inj:
        for al in ALPHAS:
            arms.append(("input_all_a%.1f" % al, {"kind": "input", "loops": set(), "alpha": al}))
    rows = []
    t0 = time.time()
    for pi, p in enumerate(pairs):
        sA, zA = ed.run(p["A"])
        sB, zB = ed.run(p["B"])
        mA = (zA[:, p["tB"]] - zA[:, p["tA"]])
        mB = (zB[:, p["tB"]] - zB[:, p["tA"]])
        knowA = int(zA[-1].argmax()) == p["tA"]
        knowB = int(zB[-1].argmax()) == p["tB"]
        rec = {"i": pi, "rel": p["rel"], "a": p["a"], "b": p["b"], "knowA": knowA, "knowB": knowB,
               "mA": mA.tolist(), "mB": mB.tolist(), "arms": {}}
        for name, spec in arms:
            e = dict(spec)
            e["pos"] = p["pos"]
            e["ref"] = sB
            if spec.get("kind") == "input":
                e["embB"] = ed.emb(p["B"])
            _, z = ed.run(p["A"], e)
            m = (z[:, p["tB"]] - z[:, p["tA"]])
            rec["arms"][name] = {"m": m.tolist(), "flip": int(z[-1].argmax()) == p["tB"]}
        rows.append(rec)
        if pi % 20 == 0:
            print("[k13 %s] %d/%d %.0fs" % (a.model, pi + 1, len(pairs), time.time() - t0), flush=True)
    # summary on pairs where the model knows both answers
    good = [r_ for r_ in rows if r_["knowA"] and r_["knowB"]]
    summ = {}
    for half, sel in (("dev", [r_ for r_ in good if r_["i"] % 2 == 0]), ("heldout", [r_ for r_ in good if r_["i"] % 2 == 1])):
        out = {}
        for name, spec in arms:
            imm, fin, flips = [], [], []
            for r_ in sel:
                mA, mB, me = r_["mA"], r_["mB"], r_["arms"][name]["m"]
                jj = (min(spec["loops"]) - 1) if spec["loops"] else K - 1
                for lst, idx in ((imm, jj), (fin, K - 1)):
                    d = mB[idx] - mA[idx]
                    if abs(d) > 1e-3:
                        lst.append((me[idx] - mA[idx]) / d)
                flips.append(r_["arms"][name]["flip"])
            n = len(sel)
            by_loop = []
            for jl in range(K):
                v = [(r_["arms"][name]["m"][jl] - r_["mA"][jl]) / (r_["mB"][jl] - r_["mA"][jl]) for r_ in sel
                     if abs(r_["mB"][jl] - r_["mA"][jl]) > 1e-3]
                by_loop.append(round(sum(v) / max(1, len(v)), 3))
            out[name] = {"effect_by_loop": by_loop, "final": round(sum(fin) / max(1, len(fin)), 3),
                         "flip": round(sum(flips) / max(1, n), 3), "n": n}
        summ[half] = out
    res = {"model": a.model, "K": K, "n_pairs": len(pairs), "n_known_both": len(good), "summary": summ,
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"res": res, "rows": rows}, open(os.path.join(a.out, "k13_%s.json" % a.model), "w"))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
