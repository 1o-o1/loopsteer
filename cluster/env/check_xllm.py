"""Import check after the xllm build (cluster/env/build_xllm.sh): xllm and its CUDA extension import, every IFM
checkpoint of the job list loads through the pilot loader (cluster/pilot/k10_ifm.py, strict state dict), and one
short forward of each gives finite logits. Exit code 0 only if every check passes; one JSON line per checkpoint.

    $VENV/bin/python cluster/env/check_xllm.py [--models ifm_p1_ouro,...]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), "pilot")
MODELS = ["ifm_p1_ouro", "ifm_p1_ouro_inj", "ifm_p1_mid", "ifm_p1_mid_inj", "ifm_p2m_r5", "ifm_p2m_diag",
          "ifm_p2m_ortho", "ifm_d112"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    sys.path.insert(0, PILOT)
    import torch
    import k10_ifm
    sys.path.insert(0, k10_ifm.XLLM)
    import xllm                     # noqa: F401
    import xllm_extension           # noqa: F401  (the CUDA extension built by setup.py)
    import k15_smoke as S1
    ok = True
    for name in [x for x in a.models.split(",") if x]:
        rec = {"model": name}
        try:
            M = S1.load_model(name)                    # k15_smoke.IfmM over k10_ifm.make_runner (strict load)
            ids = M.enc("The capital of France is")
            with torch.no_grad():
                logits, _ = M.r.fwd(torch.tensor([ids], device=M.dev), M.r.k)
            rec["K"], rec["top1"] = int(M.r.k), int(logits[0, -1].float().argmax())
            rec["finite"] = bool(torch.isfinite(logits.float()).all())
            rec["ok"] = rec["finite"]
            del M
            torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001
            rec["ok"], rec["error"] = False, repr(ex)[:300]
        ok = ok and rec["ok"]
        print(json.dumps(rec), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
