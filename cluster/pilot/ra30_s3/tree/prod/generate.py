"""One (model, task, depth, shard) generation job.

  python -m prod.generate --model=ouro_1_4b_base --task=gsm8k --k=4 \
      [--protocol=natural|forced] [--n=N] [--shard=0 --shards=1] [--adapter=DIR] \
      [--caps=0,16,32,64,128,256,512] [--extra-caps=48,96,192,384] [--horizon=512] \
      [--batch-cap=32] [--mem-fraction=0.85] [--tag-loops --tag-tokens] [--out=DIR] \
      [--batch-width=N] [--no-mask] [--wall-clock] [--resume-from-chains]
      [--continue-chains=think_tag|horizon] [--protocol-tag=natural2]
      [--continue-from-cells=FILE] [--old-horizon=N]

Protocol (natural stop), ported from s32_run.py and s28_common:
  generate once at the horizon with the model's own stop rule, then cut(B) = trace[:min(B, natural
  stop)] and run ONE forced read-out per DISTINCT cut length, serving every budget that shares it.
  Both parses are stored per cap, and protocol v2 is computed from the stored fields.

Protocol (continuation of a stored chain), `--continue-chains=<mode>` on the natural protocol:
  the chains sidecar of a finished natural-stop job is replayed as the prefix (prefill, exactly as
  --resume-from-chains does) and greedy decoding CARRIES ON: think_tag continues the rows whose
  chain ended at the closing think tag past it, horizon continues the rows that reached the old
  horizon out to the new one. The cuts and the forced read-outs are recomputed at every cap for
  those rows; every other row, and every cap at or below a continued row's old stop, is copied from
  the old cells file, whose prefix is identical there. The output is a SEPARATE grid under
  --protocol-tag (default natural2), and neither the source cells file nor the chains sidecar is
  ever written.

Protocol (forced continuation), ported from s13_common.decode(stopper=...) / make_stopper:
  no row ever stops; a token that would stop is replaced by the "Wait," tokens, so every row emits
  exactly `horizon` tokens. GSM8K and MATH500 on the Ouro checkpoints only.

Pinned protocol rules:
  * batch width 16 in every production run, stored per row as `batch_width` with the
    actual group size of the generation and of the read-out beside it. `--batch-width=0` restores the
    adaptive KV-ceiling width and exists for the gates, which reproduce spikes that ran that way.
  * the forced read-out is truncated at its eos, KEEPING the eos token, in every family. Where the
    strip removed text, the row also carries `nostrip` = what the answer would have been without it,
    so a mismatch can be attributed to the strip rule with no second generation.
  * forced continuation defaults to a fixed N per task (GSM8K 300, MATH500 500) and `--n`
    overrides it. The full-N forced block is priced separately.

Every hyperparameter above is an argument; nothing is hard-coded in a driver.
Resumable at (problem, cut) granularity: traces and cells are append-only jsonl, re-read on start.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import deque

import torch

from . import config as cfgmod
from .common import (ART, Appender, BATCH_WIDTH, CAPS_EXTRA, CAPS_STANDARD, FORCED_BUDGETS,
                     FORCED_HORIZON, FORCED_N, HORIZON, env_report, gpu_procs, load_ckpt, nvsmi,
                     read_header, read_jsonl, save_json, sha256_text)
from .models import depths_for, get
from .tasks import (ans_eq, build_prompts, data_hashes, make_find_cut, parse_forced, parse_own,
                    row_kind, row_n_answer, rows as task_rows_of, split_labels, task_cfg)


# ------------------------------------------------------------------ protocol v2
def score_v2(row):
    """s32_common.score_v2, verbatim: the model's own answer if it parses inside the cut, else the
    forced read-out. Computed from stored fields only."""
    if row.get("trace_answer") is not None:
        return bool(row["trace_correct"])
    return bool(row["correct"])


# ------------------------------------------------------------------ argument parsing
def build_parser():
    p = argparse.ArgumentParser(prog="prod.generate")
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--protocol", default="natural", choices=("natural", "forced"))
    p.add_argument("--adapter", default=None, help="LoRA adapter directory (S32 tag convention)")
    p.add_argument("--n", type=int, default=None, help="use only the first N rows of the shard")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--only-rows", dest="only_rows", default=None,
                   help="a JSON file holding a list of row_idx values; generate exactly those "
                        "rows, in the task's own order (prod.live_check uses it to regenerate the "
                        "prompts the allocator chose a given cell for)")
    p.add_argument("--extra-caps", dest="extra_caps", default=",".join(str(c) for c in CAPS_EXTRA))
    p.add_argument("--no-extra-caps", dest="extra_caps", action="store_const", const="")
    p.add_argument("--mem-fraction", dest="mem_fraction", type=float, default=None)
    # every protocol default (caps, horizon, batch width, seed, forced block, depths, datasets)
    # comes from prod/config.py and is overridable here
    cfgmod.add_args(p)
    p.add_argument("--no-mask", dest="no_mask", action="store_true",
                   help="raven families only: never install the padding mask (batch-1 control)")
    p.add_argument("--tag-loops", dest="tag_loops", action="store_true",
                   help="S32 tag convention: prepend 'Loops: k.'")
    p.add_argument("--tag-tokens", dest="tag_tokens", action="store_true",
                   help="S32 tag convention: prepend 'Loops: k. Tokens: T.' (one pass per cap)")
    p.add_argument("--suffix", default=None, help="override the task's forced read-out suffix")
    p.add_argument("--out", default=None, help="artifacts directory (default prod artifacts/)")
    p.add_argument("--wall-clock", dest="wall_clock", action="store_true",
                   help="also write a wall-clock row for cost.py")
    p.add_argument("--resume-from-chains", dest="resume_from_chains", action="store_true",
                   help="forced protocol only: when chains_<model>_<task>_k<k>.jsonl "
                        "(written by a prior natural-stop job on the same model/task/k) exists, "
                        "re-prefill prompt + its stored chain_ids instead of regenerating that span, "
                        "then apply the existing Wait injection and continue to the horizon; falls "
                        "back to full generation for a row missing from the file")
    p.add_argument("--continue-chains", dest="continue_chains", default=None,
                   choices=CONTINUE_MODES,
                   help="natural protocol only: instead of generating, replay the stored "
                        "chains_<model>_<task>_k<k>.jsonl as the prefix and CONTINUE the rows this "
                        "mode selects -- think_tag: the rows whose chain ended at the closing "
                        "think tag (the tag is put back and decoding carries on past it to the "
                        "task's "
                        "stop strings, the eos or the horizon); horizon: the rows whose chain "
                        "reached the old horizon (decoding carries on to --horizon). The cuts and "
                        "forced read-outs are recomputed at every cap for those rows; every other "
                        "row, and every cap at or below the old stop, is copied from the old cells "
                        "file. Output goes to a NEW cells file named by --protocol-tag")
    p.add_argument("--protocol-tag", dest="protocol_tag", default="natural2",
                   help="the protocol tag in the output filenames of a --continue-chains job "
                        "(default natural2); ignored without --continue-chains, and it must differ "
                        "from --protocol so the source grid can never be overwritten")
    p.add_argument("--continue-from-cells", dest="continue_from_cells", default=None,
                   help="the source cells file of a --continue-chains job (default: the same job's "
                        "own cells_<model>_<task>_<protocol>_k<k>...jsonl in the artifacts dir)")
    p.add_argument("--old-horizon", dest="old_horizon", type=int, default=None,
                   help="the horizon the source grid ran at (default: the `horizon` its cells "
                        "file's _header row records)")
    p.add_argument("--save-traces", dest="save_traces", default="1")
    p.add_argument("--extend-forced-horizon", dest="extend_forced_horizon", action="store_true",
                   help="forced protocol only: carry a FINISHED forced job on to a longer "
                        "--forced-horizon from its own trace checkpoint (copied into --out with the "
                        "same --shards/--shard) instead of regenerating; refuses when the checkpoint "
                        "does not hold every row of the shard")
    p.add_argument("--no-eos-cut", dest="eos_cut", action="store_false", default=True,
                   help="use the older S9a cut rule with no eos branch, which S9c and S9f "
                        "copied; gate G1 needs it to reproduce those two spikes exactly")
    return p


def tag_line(k, T=None, loops=False, tokens=False):
    """s32_common.tag_line: the one control line between the few-shot prompt and the question."""
    if tokens and T is not None:
        return "Loops: %d. Tokens: %d.\n" % (int(k), int(T))
    if loops or tokens:
        return "Loops: %d.\n" % int(k)
    return ""


# ------------------------------------------------------------------ chains
# A sidecar per (model, task, k) -- not per tag, so every shard of a run resumes the SAME file and
# rows are keyed by the task-global problem id (`idx`), which is stable across shards; a shard's
# local row number is not. Written once by a natural-stop job; read by a later forced job's
# `--resume-from-chains`. Greedy decoding is deterministic, so the natural pass's trace up to (and
# including) its own stop marker is byte-identical to what a forced pass would produce up to that
# same point -- storing it lets the forced job skip regenerating that span.
def chain_path(out_dir, model, task, k):
    return os.path.join(out_dir, "chains_%s_%s_k%d.jsonl" % (model.replace("+", "-"), task, int(k)))


def write_chains(out_dir, model, task, k, rws, enc, cur, st, bw):
    """One row per problem, appended once, resumable exactly like the cells file (skip a problem
    already present). `chain_ids` = cur[i]['ids'][:c] where c = st[i][0] (the cut find_cut found --
    the tokens up to and including the stop marker, or the whole trace when it never found one,
    i.e. the horizon). `prompt_sha256` fingerprints the encoded prompt (enc[i], the token ids), so a
    later resume can detect a prompt that changed underneath it (a different suffix, tag line, or
    tokenizer) and fall back to full generation instead of silently continuing the wrong context."""
    path = chain_path(out_dir, model, task, k)
    have = load_ckpt(path, lambda r: int(r["idx"]))
    ap = Appender(path)
    n = 0
    try:
        for i, r in enumerate(rws):
            ridx = int(r["idx"])
            if ridx in have or not cur[i].get("done"):
                continue            # a truncated trace (unrecoverable OOM) is never a chain
            c = st[i][0]
            c = len(cur[i]["ids"]) if c is None else int(c)
            ap.write({"idx": ridx, "natural_stop": (None if st[i][0] is None else int(st[i][0])),
                     "n_prompt_tokens": len(enc[i]),
                     "chain_ids": [int(x) for x in cur[i]["ids"][:c]],
                     "prompt_sha256": sha256_text(json.dumps(enc[i])), "batch_width": int(bw or 0)})
            n += 1
    finally:
        ap.close()
    return n, path


def stopper_fire_index(ids, pieces, tail_window, stopper):
    """The index of the first token of `ids` at which the forced stopper fires, or None.

    The replay is the decoder's own rule (prod/models/ouro.py decode): `tail_old` is the decoded
    text of the last up-to-`tail_window` tokens BEFORE the candidate, and the candidate is the token
    at that index, so the index returned is the position the forced protocol would have injected at.
    """
    tail = deque(maxlen=tail_window)
    for j, x in enumerate(ids):
        if stopper(int(x), "".join(tail)) is not None:
            return j
        tail.append(pieces[int(x)])
    return None


def seed_from_chain(chain_row, pieces, tail_window, stopper=None):
    """The prefill-and-continue bookkeeping `--resume-from-chains` needs: the resumed trace, and the
    tail-window seed a forced state's stopper requires (the last up-to-`tail_window` decoded pieces
    of the chain) so a pattern straddling the chain/continuation boundary (e.g. a '####' line split
    across the two) is still seen whole by the stopper on the very first newly generated token.
    Positions fall out for free: the caller sets fstate['off'] = len(cur[i]['ids']) exactly as it
    already does for a fresh row (generate.py, the wave loop), so an absolute position computed as
    off + len(gen[b]) is correct whether the trace started empty or from a stored chain.

    With `stopper` given (the same make_stopper the job decodes with), the chain is replayed through
    it and truncated just BEFORE the first token at which it fires, so the decoder regenerates that
    token, the stopper fires on it, and the injection lands exactly where a run from the prompt
    would have put it. The truncation is not cosmetic: the chain was cut by the NATURAL rule (a stop
    string or eos), which fires LATER than the forced stopper -- after the '#### <number>' on gsm8k,
    after 'Final Answer:' on math500 -- so an untruncated replay both overshoots the injection point
    and seeds a tail that already holds the stop pattern, which makes the stopper's own
    "... and not in tail_old" guard false from the first generated token: the row then emits NO
    injection at all and reports a null natural stop while still being labelled forced.
    Returns (ids, tail_seed) with no stopper, else (ids, tail_seed, n_truncated) -- the same "the
    forced branch returns one field more" contract as OuroAdapter.decode.
    """
    ids = [int(x) for x in chain_row["chain_ids"]]
    if stopper is None:
        return ids, ([pieces[t] for t in ids[-tail_window:]] if pieces is not None else [])
    cut = None if pieces is None else stopper_fire_index(ids, pieces, tail_window, stopper)
    n_truncated = 0
    if cut is not None:
        n_truncated = len(ids) - cut
        ids = ids[:cut]
    tail_seed = [pieces[t] for t in ids[-tail_window:]] if pieces is not None else []
    return ids, tail_seed, n_truncated


# ------------------------------------------------------------------ continuation of a stored chain
# `--continue-chains=<mode>` (natural protocol): the cluster holds a chains sidecar for every
# (model, task, k) of the natural-stop grids, so a chain that stopped too early can be CONTINUED
# instead of regenerated. The stored ids are replayed as the prefix (prefill, exactly as
# --resume-from-chains does for the forced protocol) and greedy decoding carries on to a new stop
# rule. Output goes to its own cells file, named by --protocol-tag (default natural2); the source
# grid and the chains sidecar are only ever READ.
#:  think_tag  rows whose chain ended at the closing think tag: put the tag back (the stored ids end
#:             one token short of it, because write_chains cuts at the marker POSITION) and continue
#:             past it to the task's stop strings, the tokenizer eos, or the horizon.
#:  horizon    rows whose chain reached the old horizon: continue to the new --horizon.
CONTINUE_MODES = ("think_tag", "horizon")
#: characters of the cut text stored per row as `chain_tail`
CHAIN_TAIL_CHARS = 200


def reopen_for_longer_horizon(rec, hor, forced, extend):
    """A checkpointed forced row is done exactly when its trace reached the horizon it ran at (no
    forced row stops for any other reason). Under --extend-forced-horizon a finished row whose ids
    are shorter than the CURRENT horizon is re-opened, so the protocol carries on from its stored
    state (ids and fstate) to the new horizon instead of regenerating the first span."""
    done = bool(rec["done"])
    if extend and forced and done and len(rec["ids"]) < hor:
        return False
    return done


def iter_jsonl(path):
    """Stream a jsonl artifact row by row, header skipped. `common.read_jsonl` holds the whole file,
    and a chains sidecar (or a cells file) for a 2,290-problem job is tens of MB."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                 # noqa: BLE001
                continue                                      # truncated last line after a kill
            if isinstance(r, dict) and not r.get("_header"):
                yield r


def job_tag(model, task, protocol, k, tagstr="", shardstr=""):
    """The one place a job's artifact tag is spelled. `protocol` is the PROTOCOL TAG, which is the
    protocol itself for an ordinary job and --protocol-tag for a continuation job."""
    return "%s_%s_%s_k%d%s%s" % (model.replace("+", "-"), task, protocol, int(k), tagstr, shardstr)


def check_not_overwriting(new_path, old_path):
    """A continuation job must never write over the grid it reads."""
    if os.path.abspath(new_path) == os.path.abspath(old_path):
        raise SystemExit("--continue-chains would write over its own source, %s; pass a "
                         "--protocol-tag that differs from --protocol" % old_path)


def select_continuation_rows(mode, chain_stop, cells_marker, old_horizon, think_text="</think>"):
    """Which stored chains this job continues, and where each one's OLD stop was.

    `chain_stop`   {row_idx: the chains sidecar's `natural_stop`} -- the rows that HAVE a chain.
    `cells_marker` {row_idx: the old cells row's `stop_marker`}.

    The selection field differs by mode, and which one is used matters:
      think_tag  `stop_marker` from the OLD CELLS rows. The chains sidecar stores the stop POSITION
                 but not which marker produced it, and for a Thinking checkpoint the natural cut
                 rule (tasks.make_find_cut's find_cut_think) fires on the first token in the eos
                 set, which is either the tokenizer eos or `</think>`; only the cells row says
                 which. `stop_marker` is the decoded stop token, so it is compared to `think_text`.
      horizon    `natural_stop` from the CHAINS file: missing, or at/above the old horizon.

    Returns {"mode", "field", "selected": {row_idx: old_stop}, "n_candidates", "no_chain"}.
    """
    if mode not in CONTINUE_MODES:
        raise ValueError("--continue-chains must be one of %s, not %r" % (CONTINUE_MODES, mode))
    sel, cand, no_chain = {}, [], []
    if mode == "think_tag":
        field = "old cells stop_marker == %r" % think_text
        for ridx, mk in cells_marker.items():
            if mk is None or str(mk).strip() != str(think_text).strip():
                continue
            cand.append(ridx)
            if ridx in chain_stop:
                s = chain_stop[ridx]
                sel[ridx] = int(old_horizon) if s is None else int(s)
            else:
                no_chain.append(ridx)
    else:
        field = "chains natural_stop missing or >= the old horizon %d" % int(old_horizon)
        for ridx, s in chain_stop.items():
            if s is None or int(s) >= int(old_horizon):
                cand.append(ridx)
                sel[ridx] = int(old_horizon) if s is None else int(s)
    return {"mode": mode, "field": field, "selected": sel, "n_candidates": len(cand),
            "no_chain": sorted(no_chain)}


def find_cut_after(tok, ids, boundary, stop_strings, stop_ids):
    """The natural cut of a CONTINUED trace: the earliest stop at or after `boundary`.

    `tasks.make_find_cut` scans from 0, which on a continued trace re-finds the marker that stopped
    the OLD chain -- it sits at the end of the replayed prefix -- and cuts the continuation away
    again. This is that rule with a floor: the same eos-token scan and the same bisect from a stop
    string's character position onto a token count (s9a/s26/s28 find_cut_base), both restricted to
    the continuation. Returns (cut, marker), or (len(ids), None) when nothing stopped it.
    """
    boundary = max(0, int(boundary))
    best, mk = None, None
    sset = {int(x) for x in (stop_ids or ())}
    for j in range(boundary, len(ids)):
        if int(ids[j]) in sset:
            best, mk = j, tok.decode([int(ids[j])], clean_up_tokenization_spaces=False)
            break
    if stop_strings:
        head = 0 if not boundary else len(
            tok.decode(ids[:boundary], clean_up_tokenization_spaces=False))
        full = tok.decode(ids, clean_up_tokenization_spaces=False)
        hits = [full.find(s, head) for s in stop_strings]
        hits = [h for h in hits if h >= 0]
        if hits:
            pos = min(hits)
            lo, hi = boundary, len(ids)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(tok.decode(ids[:mid], clean_up_tokenization_spaces=False)) > pos:
                    hi = mid
                else:
                    lo = mid + 1
            c = max(boundary, lo - 1)
            if best is None or c < best:
                best, mk = c, full[pos:pos + 16]
    return (len(ids), None) if best is None else (best, mk)


def cont_seed(cont, rws, enc, cur, horizon):
    """Seed `cur` from the stored chains for a --continue-chains job, and fill `cont["boundary"]`.

    A selected row's trace starts as its stored ids (plus `cont["tag_ids"]` in think_tag mode, the
    stop token the stored ids end one short of), and its boundary is the length of that prefix: no
    stop before it counts, and the read-outs below the old stop are the OLD ones (copy_old_rows).
    A row whose stored prompt fingerprint no longer matches this job's prompt, or that has no chain,
    is NOT generated -- the prefix would be the wrong context -- and is copied from the old grid
    instead, exactly like a row this mode did not select (`_copy`). A row that already has ids (a
    killed continuation job resuming off its own trace checkpoint) is left alone, but its boundary
    is recomputed from the chain, never read back from the checkpoint.
    """
    sel = cont["selected"]
    tag_ids = [int(x) for x in (cont.get("tag_ids") or [])]
    cont.setdefault("boundary", {})
    out = {"continued": [], "prompt_mismatch": [], "no_chain": [], "not_selected": 0}
    for i, r in enumerate(rws):
        ridx = int(r["idx"])
        cr = (cont.get("chains") or {}).get(ridx)
        if ridx not in sel or cr is None:
            if ridx in sel:
                out["no_chain"].append(ridx)
            else:
                out["not_selected"] += 1
            cur[i] = {"ids": cur[i].get("ids") or [], "done": True, "_copy": True}
            continue
        if cr.get("prompt_sha256") != sha256_text(json.dumps(enc[i])):
            out["prompt_mismatch"].append(ridx)
            cur[i] = {"ids": [], "done": True, "_copy": True}
            continue
        ids = [int(x) for x in cr["chain_ids"]]
        if tag_ids and ids[-len(tag_ids):] != tag_ids:
            ids = ids + tag_ids
        cont["boundary"][i] = len(ids)
        if not cur[i].get("ids"):
            cur[i] = {"ids": ids, "done": len(ids) >= int(horizon)}
        out["continued"].append(ridx)
    return out


def fill_shared_cuts(ap, caps, rows_by_cap, is_extra):
    """Write the caps a copied problem is missing by SHARING an identical cut.

    A continuation job may run a wider cap set than the grid it copies from (horizon mode adds the
    new horizon as a cap). For a problem it did NOT continue, cut(B) = min(natural stop, B) is the
    same number at the new cap as at the highest old one, so the read-out already copied at that cut
    serves the new cap too -- the protocol's own rule of ONE read-out per distinct cut, serving
    every budget that shares it. Returns the rows written (the caller records them as done).
    """
    by_cut = {}
    for B in sorted(rows_by_cap):
        r = rows_by_cap[B]
        by_cut.setdefault(int(r["n_cut"]), r)
    # The problem's stop is the one its HIGHEST copied cap reports. A source grid that was itself
    # a continuation (natural2) carries the OLD stop on the caps it copied from below the old
    # boundary and the final stop on the caps it regenerated above it, so the lowest cap's
    # natural_stop can name a cut no row has, and every cap above the old horizon would be
    # left unwritten.
    top = max(rows_by_cap) if rows_by_cap else None
    stop = rows_by_cap[top].get("natural_stop") if top is not None else None
    out = []
    for B in caps:
        if int(B) in rows_by_cap:
            continue
        cut = int(B) if stop is None else min(int(stop), int(B))
        src = by_cut.get(cut)
        if src is None and top is not None and stop is not None \
                and int(stop) <= int(top) < int(B):
            # the top cap already reads out at the natural stop, whatever its n_cut bookkeeping
            src = rows_by_cap[top]
        if src is None:
            continue                       # no identical cut was copied: it must be regenerated
        r = dict(src)
        r["B"] = int(B)
        r["extra"] = bool(is_extra(int(B)))
        r["shared_cut_from_B"] = int(src["B"])
        ap.write(r)
        out.append(r)
    return out


def copy_old_rows(old_path, ap, idx_of, done, copy_all, copy_upto, extra_of):
    """Copy the rows a continuation job does not regenerate out of the OLD cells file.

    `copy_all`   row_idx values whose every cap is copied (the rows this job did not continue).
    `copy_upto`  {row_idx: old stop} for the rows it DID continue: a cap at or below the old
                 stop was produced from a byte-identical prefix (cut(B) = min(stop, B) = B on both
                 sides, and the chain's first B ids are the same ids), so its read-out is the old
                 row; a cap above it is regenerated.
    Every field is kept as it was, with `idx` re-stamped to this job's own local index (the resume
    key of the new cells file is (idx, B)) and `extra_of` adding the provenance fields.
    Streams the old file, so a 17 MB cells file is never held in memory.
    """
    n_copy, n_skip = 0, 0
    for row in iter_jsonl(old_path):
        try:
            ridx, B = int(row["row_idx"]), int(row["B"])
        except (KeyError, TypeError, ValueError):
            continue
        i = idx_of.get(ridx)
        if i is None:
            continue                                  # not a problem of this job (shard, --n)
        if ridx not in copy_all:
            lim = copy_upto.get(ridx)
            if lim is None or B > int(lim):
                n_skip += 1
                continue
        if (i, B) in done:
            continue
        out = dict(row)
        out["idx"] = int(i)
        out.update(extra_of(i, ridx, B, row) or {})
        ap.write(out)
        done[(i, B)] = out
        n_copy += 1
    return n_copy, n_skip


# ------------------------------------------------------------------ per-row text diagnostics
# Written on every row generate.py produces, so a wrong label can be diagnosed at text level
# without regenerating the job.
def stop_reason_of(marker, n_trace, horizon, stops=None, think_text=None, think_id=None,
                   eos_texts=None, eos_ids=None, forced=False):
    """Why the trace stopped, as one of eos / stop_string:<which> / think_tag / horizon.

    `marker` is the cut rule's own marker: the decoded stop token (find_cut_think), the literal
    "eos", or the first 16 characters at a stop string's match (find_cut_base) -- and under the
    forced protocol the stopper's reason ("tok_<id>", "hash_line", "final_answer", ...).
    """
    if marker is None:
        return "horizon" if int(n_trace or 0) >= max(1, int(horizon or 0)) else "none"
    mk = str(marker)
    eset = {int(x) for x in (eos_ids or ())}
    if forced:
        if mk.startswith("tok_"):
            try:
                t = int(mk[4:])
            except ValueError:
                t = None
            if t is not None and think_id is not None and t == int(think_id):
                return "think_tag"
            if t is not None and t in eset:
                return "eos"
        return "eos" if mk == "eos" else "stop_string:%s" % mk
    if mk == "eos":
        return "eos"
    if think_text is not None and mk.strip() == str(think_text).strip():
        return "think_tag"
    if eos_texts and mk in set(eos_texts):
        return "eos"
    for s in (stops or []):
        if mk.startswith(s) or s in mk:
            return "stop_string:%s" % s
    return "stop_string:%s" % mk[:16]


def own_answer_span(cut_text, own, own_marker):
    """Character offsets [start, end) of the parsed own answer inside the cut text.

    The parsers normalise (commas, "$", brackets), so the exact string is not always present; then
    the span is the region the parser read -- the last own marker to the end of its line -- which is
    what a text-level diagnosis needs. None when there is nothing to point at.
    """
    if own is None or not cut_text:
        return None
    own = str(own)
    p = cut_text.rfind(str(own_marker)) if own_marker else -1
    # the parsers read the LAST marker's line, so the offset is searched from there, never before it
    start = cut_text.find(own, p) if p >= 0 else cut_text.rfind(own)
    if start >= 0:
        return [int(start), int(start + len(own))]
    if p < 0:
        return None
    e = cut_text.find("\n", p)
    return [int(p), int(len(cut_text) if e < 0 else e)]


def chain_tail_of(cut_text, n=CHAIN_TAIL_CHARS):
    return "" if not cut_text else str(cut_text)[-int(n):]


# ------------------------------------------------------------------ forced state checkpointing
# A forced row's whole protocol state lives in its `fstate`: where its natural stop was and why, how
# many injections it has had and at which positions, the wait tokens still draining, the stopper's
# tail window, and the absolute offset of the next token. None of it can be recomputed from the ids
# (an injected "Wait," is indistinguishable from one the model wrote itself), so a job killed
# mid-wave must read it back out of the trace checkpoint. Without it the resumed row reports
# natural_stop_pos null and n_forced_continuations 0 for injections it had already made, and the
# empty tail window also loses the stopper's "not already in the tail" guard, so a '#### n' line
# emitted before the kill fires the stopper a SECOND time and records a stop position later than the
# true one. Written for the forced protocol only; a natural-stop checkpoint row is unchanged.
def fstate_to_json(fs):
    """The JSON-serialisable form of a forced state (`tail` is a deque, every other field is
    already JSON). None when the row has no state yet."""
    if not fs:
        return None
    return {"pending": [int(x) for x in fs.get("pending") or []],
            "natural_stop_pos": (None if fs.get("natural_stop_pos") is None
                                 else int(fs["natural_stop_pos"])),
            "natural_stop_reason": fs.get("natural_stop_reason"),
            "n_forced_continuations": int(fs.get("n_forced_continuations") or 0),
            "forced_positions": [list(p) for p in (fs.get("forced_positions") or [])],
            "tail": list(fs.get("tail") or []),
            "off": int(fs.get("off") or 0)}


def fstate_from_json(d):
    """Rebuild a forced state from a trace checkpoint row, with `tail` back as the bounded deque the
    stopper reads (a plain list would keep growing and widen the window the guard looks at)."""
    from .models.ouro import TAIL_WINDOW
    return {"pending": [int(x) for x in d.get("pending") or []],
            "natural_stop_pos": d.get("natural_stop_pos"),
            "natural_stop_reason": d.get("natural_stop_reason"),
            "n_forced_continuations": int(d.get("n_forced_continuations") or 0),
            "forced_positions": [list(p) for p in (d.get("forced_positions") or [])],
            "tail": deque(d.get("tail") or (), maxlen=TAIL_WINDOW),
            "off": int(d.get("off") or 0)}


# ------------------------------------------------------------------ the job
def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = cfgmod.from_args(a)
    forced = a.protocol == "forced"
    # ---- continuation of a stored chain (--continue-chains). The output is a SEPARATE grid under
    # its own protocol tag; the source cells file and the chains sidecar are only ever READ.
    cmode = a.continue_chains
    if cmode and forced:
        raise SystemExit("--continue-chains is for the natural protocol only; the forced protocol "
                         "resumes a stored chain with --resume-from-chains")
    ptag = a.protocol_tag if cmode else a.protocol
    if cmode and str(ptag) == str(a.protocol):
        raise SystemExit("--protocol-tag must differ from --protocol (%s), else the continuation "
                         "would be written over the grid it reads" % a.protocol)
    caps = sorted(set(cfg["forced_budgets"] if forced else cfg["caps"]))
    extra = [] if forced else sorted({int(x) for x in (a.extra_caps or "").split(",")
                                      if x.strip() != ""})
    # The pinned production width. 0 means "adaptive". When the caller did not pass an explicit
    # --batch-width, the per-(model, k) table in config.yaml applies (a.batch_width is the RAW cli
    # value, None when the flag was omitted, so an explicit --batch-width=N -- including the queue's
    # own, which the launcher always sets from this same table -- still wins).
    bw = int(cfg["batch_width"]) if a.batch_width is not None \
        else cfgmod.batch_width_for(a.model, a.k, cfg)
    bw = None if bw <= 0 else bw
    if forced:
        fn = cfg["forced_block"].get("n")
        if a.n is None and isinstance(fn, dict) and a.task in fn:
            a.n = int(fn[a.task])
        elif a.n is None and isinstance(fn, int):
            a.n = int(fn)
        # fn None means FULL N, which is the default.
    all_caps = sorted(set(caps) | set(extra))
    horizon = int(cfg["forced_horizon"] if forced else cfg["horizon"])
    # Is the closing think tag a natural stop (config `think_tag_is_stop`, default True)? With it
    # False a chat-template Thinking chain runs PAST the reasoning block to the task's stop strings
    # or the eos token, and the own answer is read through the tag. The flag is recorded in the
    # header of every cells file, so a grid says which rule produced it.
    think_stop = bool(cfg.get("think_tag_is_stop", True))
    # A cap above the horizon cannot be honoured: its cut would be the horizon, and the row would
    # claim a budget it never spent. Drop it loudly rather than write a mislabelled row. (Under the
    # natural-stop protocol a trace that STOPS before a cap is a different matter: the cut is the
    # natural stop and the row is correct at every cap above it.)
    over = [c for c in all_caps if c > horizon]
    if over and forced:
        print("[warn] dropping caps above the horizon %d: %s" % (horizon, over), flush=True)
        caps = [c for c in caps if c <= horizon]
        extra = [c for c in extra if c <= horizon]
        all_caps = sorted(set(caps) | set(extra))
    out_dir = a.out or ART
    os.makedirs(out_dir, exist_ok=True)

    tcfg = task_cfg(a.task)
    if forced and not tcfg["forced"]:
        raise SystemExit("task %s has no forced-continuation protocol of record" % a.task)
    if a.k not in depths_for(a.model, a.task):
        print("[warn] depth %d is not in the depth set of record for %s on %s (%s)"
              % (a.k, a.model, a.task, depths_for(a.model, a.task)), flush=True)

    kw = {}
    if cfg.get("batch_cap"):
        kw["batch_cap"] = int(cfg["batch_cap"])
    if a.mem_fraction:
        kw["mem_fraction"] = a.mem_fraction
    ad = get(a.model, adapter_dir=a.adapter, **kw)
    if forced and ad.family != "ouro":
        raise SystemExit("forced continuation is of record for the Ouro checkpoints only; "
                         "%s is %s" % (a.model, ad.family))

    # ---- rows, split, shard
    all_rows = task_rows_of(a.task)
    labels = split_labels(a.task, len(all_rows), cfg["split_seed"], cfg["n_cal"])
    sel = list(range(a.shard, len(all_rows), a.shards)) if a.shards > 1 else list(
        range(len(all_rows)))
    if a.only_rows:
        with open(a.only_rows, encoding="utf-8") as f:
            want = set(int(x) for x in json.load(f))
        sel = [i for i in sel if int(all_rows[i].get("row_idx", all_rows[i].get("idx", i))) in want]
    if a.n:
        sel = sel[:a.n]
    rws = [all_rows[i] for i in sel]
    spl = [labels[i] for i in sel]
    N = len(rws)

    tagged = bool(a.tag_tokens)
    tagstr = ("_tagT" if a.tag_tokens else ("_tagK" if a.tag_loops else ""))
    shardstr = "" if a.shards == 1 else "_s%dof%d" % (a.shard, a.shards)
    if a.only_rows:
        shardstr += "_sub%d" % N
    tag = job_tag(a.model, a.task, ptag, a.k, tagstr, shardstr)
    old_tag = job_tag(a.model, a.task, a.protocol, a.k, tagstr, shardstr)
    t0 = time.time()
    print("[%s] N=%d caps=%s horizon=%d | %s | %s"
          % (tag, N, all_caps, horizon, nvsmi(), gpu_procs()), flush=True)

    from .models.ouro import strip_tail    # the eos-keeping strip rule, family-agnostic
    gen_widths, ro_widths = {}, {}
    _t_load = time.time()
    ad.load()
    tok = ad.tok
    dinfo = ad.set_depth(a.k)
    load_seconds = round(time.time() - _t_load, 1)
    ast = bool(getattr(ad, "add_special_tokens", False))
    # Pooled BBH mixes answer kinds, and word_sorting writes a list, so the forced read-out
    # length is a PER-ROW quantity. A read-out group decodes max(n_answer) new tokens; the extra
    # tokens on a short-answer row are truncated at its eos and never enter the accounting, which
    # stores len(o[j]) after the strip.
    nans_row = [row_n_answer(a.task, r) for r in rws]
    nans = max(nans_row) if nans_row else task_cfg(a.task)["n_answer"]

    # ---- --continue-chains: which stored chains this job continues, and where each one's old stop
    # was. Read out of the source cells file and the chains sidecar, both read-only, BEFORE the
    # output file exists, so its header row can record the source, the mode, the old horizon and the
    # count continued.
    cont, cont_sel = None, None
    if cmode:
        if a.tag_loops or a.tag_tokens:
            raise SystemExit("--continue-chains cannot continue a tagged arm: its prompt carries "
                             "the cap, so one stored chain does not belong to the new prompt")
        cpath = chain_path(out_dir, a.model, a.task, a.k)
        old_cells = a.continue_from_cells or os.path.join(out_dir, "cells_%s.jsonl" % old_tag)
        for p, what in ((cpath, "chains sidecar"), (old_cells, "source cells file")):
            if not os.path.exists(p):
                raise SystemExit("--continue-chains needs the %s %s" % (what, p))
        check_not_overwriting(os.path.join(out_dir, "cells_%s.jsonl" % tag), old_cells)
        hdr = read_header(old_cells) or {}
        old_hor = int(a.old_horizon or hdr.get("horizon")
                      or (hdr.get("config") or {}).get("horizon") or 0)
        if not old_hor:
            raise SystemExit("%s records no horizon in its _header row; pass --old-horizon"
                             % old_cells)
        if cmode == "horizon" and int(horizon) <= old_hor:
            raise SystemExit("--continue-chains=horizon needs --horizon above the old "
                             "horizon (%d); at %d every selected row is already at its stop"
                             % (old_hor, horizon))
        if cmode == "horizon" and int(horizon) > max(all_caps):
            print("[warn] the new horizon %d is above every cap %s, so the continued tail is "
                  "generated but no cap scores it: put it in --caps" % (horizon, all_caps),
                  flush=True)
        think_id = tok.convert_tokens_to_ids("</think>") if ad.chat_template else None
        think_text = None if think_id is None else tok.decode([int(think_id)],
                                                              clean_up_tokenization_spaces=False)
        if cmode == "think_tag" and think_id is None:
            raise SystemExit("--continue-chains=think_tag needs a chat-template checkpoint; %s has "
                             "no closing think tag" % a.model)
        # the stop MARKER is only in the old cells rows, the stop POSITION only in the chains file
        cells_marker, chain_stop = {}, {}
        if cmode == "think_tag":
            for r in iter_jsonl(old_cells):
                if "row_idx" in r and "stop_marker" in r:
                    cells_marker[int(r["row_idx"])] = r.get("stop_marker")
        for r in iter_jsonl(cpath):
            chain_stop[int(r["idx"])] = r.get("natural_stop")
        cont_sel = select_continuation_rows(cmode, chain_stop, cells_marker, old_hor,
                                            think_text=think_text or "</think>")
        mine = {int(r["idx"]) for r in rws} & set(cont_sel["selected"])
        cont = {"mode": cmode, "old_horizon": old_hor, "old_cells": old_cells,
                "chains_path": cpath, "think_id": think_id, "think_text": think_text,
                "selected": {i: cont_sel["selected"][i] for i in mine}, "boundary": {},
                "tag_ids": [int(think_id)] if cmode == "think_tag" else [],
                # only the selected rows' ids are held: a whole 4096-token chains file for 2,290
                # problems is hundreds of MB of Python ints
                "chains": {int(r["idx"]): r for r in iter_jsonl(cpath) if int(r["idx"]) in mine}}
        print("[%s] --continue-chains=%s: %d of %d rows selected by %s; old horizon %d, new %d; "
              "source %s" % (tag, cmode, len(mine), N, cont_sel["field"], old_hor, horizon,
                             os.path.basename(old_cells)), flush=True)

    meta_path = os.path.join(out_dir, "meta_%s.json" % tag)
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    # `protocol` is the protocol TAG of this grid (a continuation grid is its own protocol: a
    # different stop rule over the same prompts), and `protocol_base` the protocol it decodes under.
    # Every meta-driven consumer -- cost.py's reference cell, checks.py's throughput table --
    # keys on `protocol`, and must see a continuation grid as separate from the grid it reads.
    meta.update({"tag": tag, "model": a.model, "task": a.task, "k": a.k, "protocol": ptag,
                 "protocol_base": a.protocol,
                 "adapter": a.adapter, "caps": caps, "extra_caps": extra, "horizon": horizon,
                 "n_problems": N, "shard": a.shard, "shards": a.shards,
                 "n_eval": sum(1 for s in spl if s == "eval"),
                 "n_cal": sum(1 for s in spl if s == "cal"),
                 "cut_rule": "min(natural stop, B)" if not forced else "B tokens of a forced trace",
                 "n_answer_tokens": nans, "add_special_tokens": ast,
                 "config": cfg, "config_sha256": cfgmod.digest(cfg),
                 "tag_loops": bool(a.tag_loops), "tag_tokens": bool(a.tag_tokens),
                 "only_rows": a.only_rows,
                 "model_meta": ad.meta(), "depth_info": dinfo, "env": env_report(),
                 "data_hashes": data_hashes(), "argv": list(sys.argv),
                 "passes_per_token": ad.passes_per_token(a.k),
                 "kv_bytes_per_token": ad.kv_bytes_per_token(a.k),
                 "load_seconds": load_seconds,
                 "batch_width_pinned": bw or "adaptive (KV ceiling)",
                 "strip_at_eos": True, "eos_cut": bool(a.eos_cut),
                 "resume_from_chains": bool(a.resume_from_chains),
                 "protocol_tag": ptag, "continue_chains": cmode})
    if cont:
        meta["continue"] = {"mode": cmode, "chains_path": cont["chains_path"],
                            "source_cells": cont["old_cells"], "old_horizon": cont["old_horizon"],
                            "new_horizon": horizon, "selection_field": cont_sel["field"],
                            "n_candidates": cont_sel["n_candidates"],
                            "n_candidates_without_a_chain": len(cont_sel["no_chain"]),
                            "n_selected": len(cont["selected"]),
                            "think_id": cont["think_id"], "think_text": cont["think_text"]}
    meta.setdefault("oom_events", [])
    meta.setdefault("errors", {})
    meta.setdefault("passes", {})

    def save_meta():
        save_json(meta_path, meta)

    cells_path = os.path.join(out_dir, "cells_%s.jsonl" % tag)
    done = load_ckpt(cells_path, lambda r: (int(r["idx"]), int(r["B"])))
    fresh = not os.path.exists(cells_path)
    apc = Appender(cells_path)
    if fresh:
        # The effective config is the FIRST LINE of every cells file. Every reader in the package
        # skips a row with `_header` (common.read_jsonl / load_ckpt / count_cells).
        hx = {"model": a.model, "task": a.task, "k": a.k,
              "protocol": ptag, "protocol_base": a.protocol, "n_problems": N,
              "horizon": horizon, "caps": all_caps, "data_hashes": data_hashes()}
        if cont:
            # what this grid is: the chains it continued, the rule that picked them, the horizon
            # they were produced at, and how many rows were continued
            hx.update({"continue_chains": cmode, "continue_chains_path": cont["chains_path"],
                       "continue_source_cells": cont["old_cells"],
                       "continue_old_horizon": cont["old_horizon"],
                       "continue_selection_field": cont_sel["field"],
                       "continue_n_continued": len(cont["selected"])})
        apc.write(cfgmod.header_row(cfg, tag, hx))

    # ---------------------------------------------------------------- phase A: traces
    def gen_pass(T_cap, hor, key):
        """Generate traces for one pass. Returns (enc, cur, plen, stop_table, suffix_ids)."""
        tl = tag_line(a.k, T_cap, a.tag_loops, a.tag_tokens)
        prompts, suf, stops, eos_ids, extra_meta = build_prompts(
            tok, a.task, rws, chat_template=ad.chat_template, suffix_text=a.suffix,
            think_tag_is_stop=think_stop)
        if tl:
            # the tag line goes between the few-shot prompt and the question (s32_common)
            prompts = [_insert_tag(p, tl, a.task, ad.chat_template) for p in prompts]
            extra_meta["tag_line"] = tl
        find_cut = make_find_cut(tok, stops or [], eos_ids, chat_template=ad.chat_template,
                                 eos_cut=bool(a.eos_cut), think_tag_is_stop=think_stop)
        if ad.family in ("huginn", "mcleish"):
            # the family's stop ids of record (S9c: end_text, end_turn, begin_text for Huginn) join
            # the tokenizer eos, for the decoder, the cut rule and the strip alike
            eos_ids = sorted(set(eos_ids) | set(ad.stop_ids()))
            find_cut = make_find_cut(tok, stops or [], eos_ids, chat_template=ad.chat_template,
                                     eos_cut=bool(a.eos_cut), think_tag_is_stop=think_stop)
        enc = [tok(p, add_special_tokens=ast)["input_ids"] for p in prompts]
        plen = [len(e) for e in enc]
        maxpos = getattr(ad, "max_positions", None)
        if maxpos:
            # Huginn indexes a rotary table of block_size (4096) entries by absolute position, so
            # prompt + trace + suffix + read-out must stay below it or the forward raises
            room = int(maxpos) - max(plen) - len(suf) - nans - 1
            if room < hor:
                print("[warn] %s: horizon %d clamped to %d (positions limited to %d; longest prompt "
                      "%d, read-out %d)" % (key, hor, max(16, room), maxpos, max(plen),
                                           len(suf) + nans), flush=True)
                hor = max(16, room)
        meta["passes"][key] = {"tag_line": tl, "horizon": hor, "horizon_requested": horizon,
                               "max_positions": maxpos, "suffix_text":
                               extra_meta["suffix_text"], "stop_strings": stops,
                               "eos_ids": eos_ids,
                               "end_think_id": extra_meta.get("end_think_id"),
                               "think_tag_is_stop": bool(think_stop),
                               "prompt_tokens_mean": sum(plen) / max(1, len(plen)),
                               "prompt_tokens_min_max": [min(plen), max(plen)]}
        tp = os.path.join(out_dir, "trace_%s_%s.jsonl" % (tag, key))
        cur = {i: {"ids": [], "done": False} for i in range(N)}
        loaded = load_ckpt(tp, lambda r: int(r["idx"]))
        if forced and a.extend_forced_horizon:
            have = sum(1 for i in loaded if i < N)
            if have < N:
                raise SystemExit("--extend-forced-horizon: %s holds %d of this shard's %d rows; copy the "
                                 "finished forced job's trace checkpoint into --out first, and use the "
                                 "same --shards and --shard it ran with" % (tp, have, N))
        for i, r in loaded.items():
            if i < N:
                cur[i] = {"ids": r["ids"],
                          "done": reopen_for_longer_horizon(r, hor, forced, a.extend_forced_horizon)}
                if forced and r.get("fstate"):
                    # the forced row resumes its protocol state, not only its ids (see
                    # fstate_to_json for what is lost without this)
                    cur[i]["fstate"] = fstate_from_json(r["fstate"])
        apt = Appender(tp) if a.save_traces != "0" else None
        gen_tokens, gen_seconds = 0, 0.0

        # ---- --continue-chains: replay the stored chains as the prefix, and give the CONTINUATION
        # its own stop rule. `dec_eos`/`dec_stops` are what the decoder stops on and what the strip
        # uses; the cut of a continued row is measured from its boundary (find_cut_after).
        dec_eos, dec_stops = eos_ids, stops
        if cont:
            tid = cont["think_id"]
            if cont["mode"] == "think_tag":
                # continue PAST the tag: it leaves the stop set, and the task's stop strings (which
                # the chat-template branch of build_prompts does not use) become the continuation's
                # stop rule beside the tokenizer eos.
                dec_eos = [e for e in eos_ids if tid is None or int(e) != int(tid)]
                dec_stops = list(task_cfg(a.task)["stops"] or [])
            cont["stop_ids"], cont["stops"] = dec_eos, dec_stops
            cs = cont_seed(cont, rws, enc, cur, hor)
            cont["chains"] = {}                       # the ids live in `cur` now
            cont["seeded"] = cs
            meta["passes"][key].update({
                "continue_chains": cont["mode"], "continue_rows": len(cont["boundary"]),
                "continue_stop_ids": dec_eos, "continue_stop_strings": dec_stops,
                "continue_not_selected": cs["not_selected"],
                "continue_prompt_mismatch": cs["prompt_mismatch"][:500],
                "continue_no_chain": cs["no_chain"][:500]})
            if cs["prompt_mismatch"]:
                print("[warn] %d stored chain(s) no longer match this job's prompt and are copied, "
                      "not continued: %s" % (len(cs["prompt_mismatch"]),
                                             cs["prompt_mismatch"][:10]), flush=True)

        def cut_of(i, idsx):
            """This pass's cut rule for row i: from the boundary on a continued row, so the marker
            that stopped the OLD chain -- the last token of the replayed prefix -- cannot cut the
            continuation away again."""
            if cont and i in cont["boundary"]:
                return find_cut_after(tok, idsx, cont["boundary"][i], cont["stops"],
                                      cont["stop_ids"])
            return find_cut(idsx)

        if forced:
            from .models.ouro import TAIL_WINDOW, make_stopper, new_forced_states
            pieces = ad._pieces                     # noqa: SLF001  (the adapter's own table)
            stopper = make_stopper(ad.chat_template, pieces, eos_ids, a.task)
            wait = ad.wait_ids()
            meta["passes"][key].update({"wait_ids": wait, "wait_text": tok.decode(wait)})
            # ---- resume from a stored natural-stop chain instead of regenerating it.
            # A row whose OWN trace checkpoint already has ids (a resumed/incomplete forced job) is
            # left alone -- the chain is only for a row this forced job has not touched yet.
            chains_loaded, cpath = 0, chain_path(out_dir, a.model, a.task, a.k)
            chains_truncated, chain_tokens_truncated, truncated_detail = 0, 0, []
            if a.resume_from_chains and os.path.exists(cpath):
                chain_by_idx = {int(r["idx"]): r for r in read_jsonl(cpath)}
                for i in range(N):
                    if cur[i]["ids"]:
                        continue
                    cr = chain_by_idx.get(int(rws[i]["idx"]))
                    if cr is None:
                        continue
                    if cr.get("prompt_sha256") != sha256_text(json.dumps(enc[i])):
                        print("[warn] chain prompt mismatch idx=%s (%s); full generation"
                              % (rws[i]["idx"], key), flush=True)
                        continue
                    ids, tail_seed, ntr = seed_from_chain(cr, pieces, TAIL_WINDOW, stopper)
                    cur[i] = {"ids": ids, "done": len(ids) >= hor, "_tail_seed": tail_seed}
                    chains_loaded += 1
                    if ntr:
                        chains_truncated += 1
                        chain_tokens_truncated += ntr
                        truncated_detail.append([int(rws[i]["idx"]), int(ntr)])
            meta["passes"][key].update({"resume_from_chains_requested": bool(a.resume_from_chains),
                                        "chains_path": cpath, "chains_file_found": os.path.exists(cpath),
                                        "chains_loaded": chains_loaded,
                                        # the stored chain is cut by the natural rule, which fires
                                        # later than the forced stopper, so each chain is truncated
                                        # back to the stopper's own first fire before it is replayed
                                        # (seed_from_chain). These say how many chains were cut and
                                        # by how many tokens in total.
                                        "chains_truncated": chains_truncated,
                                        "chain_tokens_truncated": chain_tokens_truncated,
                                        "chains_truncated_detail": truncated_detail[:500]})

        waves = [w for w in (128, 256, 512, 1024, 2048) if w < hor] + [hor]
        for T_next in waves:
            stuck = 0
            while True:
                todo = [i for i in range(N) if not cur[i]["done"]
                        and len(cur[i]["ids"]) < T_next]
                if not todo or stuck > 2:
                    break
                Tn = min(len(cur[i]["ids"]) for i in todo)
                if cont:
                    # A continuation replays chains of MANY different lengths, and the exact-length
                    # grouping below would then decode at width one or two and cost more than the
                    # regeneration it replaces. Continued rows are grouped by the wave's own target
                    # instead: every row of the group is asked for the same T_next - Tn tokens and
                    # each row's output is trimmed to its OWN remaining budget just below, so no
                    # trace ever runs past the wave. Greedy decoding is deterministic, so a trimmed
                    # tail is simply regenerated by the next wave when it is needed.
                    at_T = sorted(todo, key=lambda i: plen[i] + len(cur[i]["ids"]))
                else:
                    at_T = sorted([i for i in todo if len(cur[i]["ids"]) == Tn],
                                  key=lambda i: plen[i])
                before = sum(len(cur[i]["ids"]) for i in range(N))
                b = bw or ad.batch_cap
                gi = 0
                while gi < len(at_T):
                    grp = at_T[gi:gi + b]
                    max_seq = max(plen[i] for i in grp) + T_next + len(suf) + nans
                    if bw is None:
                        nb, _ = ad.batch_for(a.k, max_seq)
                        if nb < len(grp):
                            b = nb
                            grp = at_T[gi:gi + b]
                    try:
                        seqs = [enc[i] + cur[i]["ids"] for i in grp]
                        if forced:
                            # s13_common carries the stopper state ACROSS segments, so the decoded
                            # tail is not cleared at a segment boundary and a stop pattern spanning
                            # it cannot be missed. The state lives on the problem, not the call.
                            for i in grp:
                                if "fstate" not in cur[i]:
                                    cur[i]["fstate"] = new_forced_states(1)[0]
                                    cur[i]["fstate"]["off"] = len(cur[i]["ids"])
                                    # A row resumed from a stored chain seeds the tail deque
                                    # with the chain's own trailing pieces, so the stopper sees a
                                    # pattern straddling the chain/continuation boundary whole.
                                    seed = cur[i].pop("_tail_seed", None)
                                    if seed:
                                        cur[i]["fstate"]["tail"].extend(seed)
                            states = [cur[i]["fstate"] for i in grp]
                            g, states, dt = ad.decode(
                                seqs, T_next - Tn, a.k, eos_ids, stop_strings=None,
                                row_ids=[rws[i]["idx"] for i in grp],
                                stopper=stopper, wait_ids=wait, states=states)
                        else:
                            kwd = {}
                            if ad.family in ("huginn", "mcleish") and a.no_mask:
                                kwd["use_mask"] = False
                            g, dt = ad.decode(seqs, T_next - Tn, a.k, dec_eos,
                                              stop_strings=dec_stops,
                                              row_ids=[rws[i]["idx"] for i in grp], **kwd)
                    except torch.OutOfMemoryError as e:
                        line = "OOM A %s b=%d %d->%d: %s" % (key, len(grp), Tn, T_next,
                                                             str(e).split("\n")[0])
                        meta["oom_events"].append(line)
                        print("[OOM] " + line, flush=True)
                        traceback.clear_frames(e.__traceback__)
                        e.__traceback__ = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                        if b == 1:
                            meta["errors"]["A_%s" % key] = line
                            save_meta()
                            break
                        b = max(1, b // 2)
                        continue
                    gen_seconds += dt
                    # Strip at the eos, KEEPING the eos token, in EVERY family. Both decoders
                    # already stop a row at its eos, so this is belt and braces rather than a
                    # second rule; it makes the rule explicit and uniform instead of Ouro-only. A
                    # forced trace is never stripped: it has no eos to stop at, by construction
                    # (the forced stopper suppresses every stop).
                    gen_widths.setdefault(len(grp), 0)
                    gen_widths[len(grp)] += 1
                    if not forced:
                        g = strip_tail(g, dec_eos)
                    if cont:
                        # each continued row keeps only its own remaining budget for this wave
                        g = [list(gj)[:max(0, T_next - len(cur[i]["ids"]))]
                             for gj, i in zip(g, grp)]
                    for j, i in enumerate(grp):
                        if forced:
                            # decode works on COPIES of the forced states and returns the new ones
                            # (prod/models/ouro.py), so the row's state is replaced only HERE, on a
                            # call that returned: an attempt that raised OOM left cur[i]["fstate"]
                            # untouched, and the retry at half the batch cannot count the injections
                            # of the failed attempt a second time.
                            cur[i]["fstate"] = states[j]
                        cur[i]["width"] = max(cur[i].get("width") or 0, len(grp))
                        cur[i]["ids"] = cur[i]["ids"] + g[j]
                        gen_tokens += len(g[j])
                        c, mk = cut_of(i, cur[i]["ids"])
                        cur[i]["done"] = bool(forced is False and (mk is not None)) or \
                            len(cur[i]["ids"]) >= hor
                        if apt is not None:
                            ck = {"idx": i, "row_idx": rws[i]["idx"], "ids": cur[i]["ids"],
                                  "done": cur[i]["done"], "natural_stop": c, "marker": mk,
                                  "wave": T_next}
                            if forced:
                                # the forced row's protocol state travels WITH its ids, so a kill
                                # mid-wave resumes the stop position, the injection count, the
                                # pending wait tokens and the stopper's tail window
                                ck["fstate"] = fstate_to_json(cur[i].get("fstate"))
                            apt.write(ck)
                    gi += len(grp)
                    gc.collect()
                    torch.cuda.empty_cache()
                stuck = stuck + 1 if sum(len(cur[i]["ids"]) for i in range(N)) == before else 0
            if all(cur[i]["done"] for i in cur):
                break
        if apt is not None:
            apt.close()
        st = {}
        for i in range(N):
            if forced:
                # S13's semantics for a forced trace: `natural_stop` is the position of the FIRST
                # suppressed stop (the stopper's `natural_stop_pos`), not a cut-rule position -- the
                # forced trace never stops, so the cut rule has nothing to find. s13_merge.py writes
                # natural_stop == natural_stop_pos for exactly this reason.
                fs = cur[i].get("fstate") or {}
                st[i] = (fs.get("natural_stop_pos"), fs.get("natural_stop_reason"))
            else:
                c, mk = cut_of(i, cur[i]["ids"])
                st[i] = (min(c, hor), mk)
        nat = [st[i][0] for i in range(N) if st[i][0] is not None]
        meta["passes"][key].update({
            "natural_stop_mean": (sum(nat) / len(nat)) if nat else None,
            "natural_stop_min_max": [min(nat), max(nat)] if nat else None,
            "natural_stop_hit_frac": sum(1 for i in st if st[i][1] is not None) / max(1, N),
            "trace_tokens": sum(len(cur[i]["ids"]) for i in cur),
            "generated_tokens": gen_tokens, "generate_seconds": round(gen_seconds, 2),
            "tokens_per_s": (gen_tokens / gen_seconds) if gen_seconds > 0 else None})
        save_meta()
        print("  pass %s: natural stop mean %s hit %.3f, %d tokens in %.0fs (%.1f tok/s) [%.0fs]"
              % (key, ("%.1f" % meta["passes"][key]["natural_stop_mean"])
                 if meta["passes"][key]["natural_stop_mean"] is not None else "n/a",
                 meta["passes"][key]["natural_stop_hit_frac"], gen_tokens, gen_seconds,
                 meta["passes"][key]["tokens_per_s"] or 0.0, time.time() - t0), flush=True)
        return enc, cur, plen, st, suf, eos_ids, find_cut

    # ---------------------------------------------------------------- phase B: forced read-out
    def readout_pass(enc, cur, plen, st, suf, eos_ids, jobs, key):
        jobs = [(i, c, bs) for (i, c, bs) in jobs if any((i, B) not in done for B in bs)]
        jobs.sort(key=lambda t: plen[t[0]] + t[1])
        b = bw or ad.batch_cap
        gi, nrows = 0, 0
        # what the three text-level diagnostic fields need: this pass's stop rule, its horizon, the
        # think tag, and the task's own-answer marker
        pm = meta["passes"].get(key) or {}
        p_stops, p_hor = pm.get("stop_strings") or [], int(pm.get("horizon") or 0)
        p_think = pm.get("end_think_id")
        p_think_txt = None if p_think is None else tok.decode(
            [int(p_think)], clean_up_tokenization_spaces=False)
        eos_txt = [tok.decode([int(e)], clean_up_tokenization_spaces=False)
                   for e in (eos_ids or [])]
        own_mk = task_cfg(a.task)["own_marker"]
        while gi < len(jobs):
            grp = jobs[gi:gi + b]
            max_seq = max(plen[i] + c for (i, c, _x) in grp) + len(suf) + nans
            if bw is None:
                nb, _ = ad.batch_for(a.k, max_seq)
                if nb < len(grp):
                    b = nb
                    grp = jobs[gi:gi + b]
            try:
                kwd = {}
                if ad.family in ("huginn", "mcleish") and a.no_mask:
                    kwd["use_mask"] = False
                o, dt = ad.decode([enc[i] + cur[i]["ids"][:c] + suf for (i, c, _x) in grp],
                                  nans, a.k, eos_ids, stop_strings=None,
                                  row_ids=[rws[i]["idx"] for (i, _c, _x) in grp], **kwd)
            except torch.OutOfMemoryError as e:
                line = "OOM B %s b=%d: %s" % (key, len(grp), str(e).split("\n")[0])
                meta["oom_events"].append(line)
                print("[OOM] " + line, flush=True)
                traceback.clear_frames(e.__traceback__)
                e.__traceback__ = None
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                if b == 1:
                    meta["errors"]["B_%s" % key] = line
                    save_meta()
                    break
                b = max(1, b // 2)
                continue
            # the group decoded max(n_answer) new tokens; each row keeps only its OWN read-out length
            # (a letter row 8, word_sorting 48), so the parse and the token accounting see exactly
            # the read-out the protocol defines for that row
            o = [list(oj)[:nans_row[i]] for oj, (i, _c, _x) in zip(o, grp)]
            raw = [list(x) for x in o]          # before the eos strip, for cause attribution
            o = strip_tail(o, eos_ids)          # the eos-keeping strip, every family
            ro_widths.setdefault(len(grp), 0)
            ro_widths[len(grp)] += 1
            for j, (i, c, bs) in enumerate(grp):
                # the eos token is KEPT in o[j] for the token accounting but must not reach
                # the parsed text: "42<|endoftext|>" is not "42" for the math and free-form parsers
                atxt = tok.decode([t for t in o[j] if t not in eos_ids],
                                  clean_up_tokenization_spaces=False)
                ctxt = tok.decode(cur[i]["ids"][:c], clean_up_tokenization_spaces=False)
                opts = rws[i].get("options") or None
                gold = rws[i]["target"]
                kind = row_kind(a.task, rws[i])
                pred = parse_forced(atxt, a.task, opts, kind)
                own = parse_own(ctxt, a.task, opts, kind, think_tag_is_stop=think_stop)
                # what the answer would have been WITHOUT the eos strip, stored only when the strip
                # actually removed something, so a mismatch can be attributed to the strip rule
                # without a second generation.
                nostrip = None
                if len(raw[j]) != len(o[j]):
                    rtxt = tok.decode([t for t in raw[j] if t not in eos_ids],
                                      clean_up_tokenization_spaces=False)
                    nostrip = {"n_answer_tokens": len(raw[j]),
                               "pred": parse_forced(rtxt, a.task, opts, kind)}
                    nostrip["correct"] = bool(ans_eq(nostrip["pred"], gold, a.task, kind))
                ngen = c + len(o[j])
                ppt = ad.passes_per_token(a.k)
                # every row says why the chain stopped, what its last 200 characters before the cut
                # were, and where in the cut text the own answer was parsed from, so a wrong label
                # can be diagnosed without regenerating the job.
                sreason = stop_reason_of(st[i][1], len(cur[i]["ids"]), p_hor, stops=p_stops,
                                         think_text=p_think_txt, think_id=p_think,
                                         eos_texts=eos_txt, eos_ids=eos_ids, forced=forced)
                ctail = chain_tail_of(ctxt)
                ospan = own_answer_span(ctxt, own, own_mk)
                for B in bs:
                    if (i, B) in done:
                        continue
                    row = {"idx": i, "row_idx": rws[i]["idx"], "split": spl[i],
                           "model": a.model, "adapter": a.adapter, "task": a.task, "k": a.k,
                           "B": B, "extra": bool(B in extra and B not in caps),
                           "protocol": ptag, "protocol_base": a.protocol,
                           "tag_line": meta["passes"][key]["tag_line"],
                           "pred": pred, "gold": gold, "kind": kind,
                           "subtask": rws[i].get("subtask"),
                           "subject": rws[i].get("subject"),
                           "correct": bool(ans_eq(pred, gold, a.task, kind)),
                           "trace_answer": own,
                           "trace_correct": bool(ans_eq(own, gold, a.task, kind)),
                           "n_cut": c, "natural_stop": st[i][0], "stop_marker": st[i][1],
                           "n_trace": len(cur[i]["ids"]), "n_generated": ngen,
                           "n_prompt_tokens": plen[i], "n_answer_tokens": len(o[j]),
                           "n_suffix_tokens": len(suf),
                           # the budget is k L (P + T + R) with R = suffix + answer tokens;
                           # n_generated counts the cut and the answer, the suffix is added here
                           "layer_passes": ppt * (plen[i] + ngen + len(suf)),
                           "layer_passes_promptfree": ppt * (ngen + len(suf)),
                           "cost_fields": "prompt+cut+suffix+answer",
                           "answer_text": atxt,
                           # the width is pinned and STORED PER ROW, because a per-problem label is
                           # only comparable to another taken at the same width
                           "batch_width": bw or 0,
                           "batch_width_readout": len(grp),
                           "batch_width_gen": cur[i].get("width"),
                           "strip_at_eos": True,
                           "stop_reason": sreason, "chain_tail": ctail,
                           "own_answer_span": ospan}
                    if cont:
                        row.update({"continue_chains": cont["mode"],
                                    "continued": bool(i in cont["boundary"]),
                                    "continue_boundary": cont["boundary"].get(i),
                                    "continue_old_stop": cont["selected"].get(
                                        int(rws[i]["idx"]))})
                    if nostrip is not None:
                        row["nostrip"] = nostrip
                    if forced and "fstate" in cur[i]:
                        fs = cur[i]["fstate"]
                        row.update({"natural_stop_pos": fs["natural_stop_pos"],
                                    "natural_stop_reason": fs["natural_stop_reason"],
                                    "n_forced_continuations": fs["n_forced_continuations"],
                                    "forced_positions": fs["forced_positions"]})
                        # S13 writes natural_stop == natural_stop_pos on a forced row
                        row["natural_stop"] = fs["natural_stop_pos"]
                        row["stop_marker"] = fs["natural_stop_reason"]
                    row["correct_v2"] = score_v2(row)
                    apc.write(row)
                    done[(i, B)] = row
                    nrows += 1
            gi += len(grp)
            gc.collect()
            torch.cuda.empty_cache()
            if gi % (16 * max(1, b)) == 0 or gi >= len(jobs):
                print("    readout %s %d/%d jobs b=%d rows %d [%.0fs] peak %.2f GB"
                      % (key, gi, len(jobs), len(grp), nrows, time.time() - t0,
                         torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)
                save_meta()
        return nrows

    # ---------------------------------------------------------------- drive
    if tagged:
        # A tagged arm's prompt contains the cap, so one generation cannot serve every cap
        # (s32_run.py): one pass per cap at that cap's horizon, plus a generation-free B=0 pass.
        prompts0, suf0, _s, eos0, ex0 = build_prompts(tok, a.task, rws,
                                                      chat_template=ad.chat_template,
                                                      suffix_text=a.suffix)
        tl0 = tag_line(a.k, 0, a.tag_loops, a.tag_tokens)
        enc0 = [tok(_insert_tag(p, tl0, a.task, ad.chat_template),
                    add_special_tokens=ast)["input_ids"] for p in prompts0]
        plen0 = [len(e) for e in enc0]
        meta["passes"]["T0"] = {"tag_line": tl0, "horizon": 0, "suffix_text": ex0["suffix_text"],
                                "natural_stop_mean": 0.0,
                                "prompt_tokens_mean": sum(plen0) / max(1, len(plen0)),
                                "note": "Tokens: 0. never appears in the S32 training pool"}
        cur0 = {i: {"ids": [], "done": True} for i in range(N)}
        st0 = {i: (0, None) for i in range(N)}
        readout_pass(enc0, cur0, plen0, st0, suf0, eos0,
                     [(i, 0, [0]) for i in range(N)], "T0")
        for T in [c for c in all_caps if c > 0]:
            enc, cur, plen, st, suf, eos_ids, _fc = gen_pass(T, T, "T%d" % T)
            readout_pass(enc, cur, plen, st, suf, eos_ids,
                         [(i, min(st[i][0], T), [T]) for i in range(N)], "T%d" % T)
    else:
        enc, cur, plen, st, suf, eos_ids, _fc = gen_pass(None, horizon, "single")
        if not forced and not cont:
            # The natural-stop pass's own trace is the chain a later forced job on the same
            # (model, task, k) can resume from instead of regenerating it.
            n_chain, chain_p = write_chains(out_dir, a.model, a.task, a.k, rws, enc, cur, st, bw)
            meta["chains_written"] = meta.get("chains_written", 0) + n_chain
            meta["chains_path"] = chain_p
            save_meta()
        elif cont:
            # A continuation job never writes chains: the sidecar is keyed by problem id, so a
            # continued trace would either be dropped (the id is already there) or stored beside
            # chains cut by a different rule. It is read-only here, and cleanup never deletes it.
            meta["chains_written"] = 0
            meta["chains_path"] = cont["chains_path"]
        jobs = []
        incomplete = {i for i in range(N) if not cur[i]["done"]}
        if incomplete:
            meta["n_incomplete_rows"] = len(incomplete)
            meta["incomplete_row_idx"] = sorted(int(rws[i]["idx"]) for i in incomplete)[:500]
            print("[warn] %d row(s) have truncated traces (OOM or stalled wave) and are left "
                  "unscored; the job will exit 3" % len(incomplete), flush=True)
            save_meta()
        cont_i = set(cont["boundary"]) if cont else set()
        if cont:
            # The copy runs BEFORE the read-outs, so a job killed in phase B has already written
            # every row it was never going to regenerate.
            idx_of = {int(rws[i]["idx"]): i for i in range(N)}
            copy_all = {int(rws[i]["idx"]) for i in range(N) if i not in cont_i}
            copy_upto = {int(rws[i]["idx"]): cont["selected"][int(rws[i]["idx"])] for i in cont_i}
            oldbase = os.path.basename(cont["old_cells"])
            own_mk_c = task_cfg(a.task)["own_marker"]
            eos_txt_c = [tok.decode([int(e)], clean_up_tokenization_spaces=False)
                         for e in (eos_ids or [])]

            def extra_of(i, ridx, B, row):
                """The provenance a copied row carries, and the text fields that CAN be recomputed
                for it. A row of a continued problem below the old stop has its prefix in memory (it
                is the stored chain), so its chain tail and own-answer offsets are filled; a
                row of a problem this job did not continue keeps them null, because its chain text
                exists in no artifact this job read."""
                # the row is copied field for field; only its provenance is added, and `protocol`
                # becomes this grid's tag (it is a row OF this grid now) with the old value kept
                ex = {"protocol": ptag, "protocol_base": a.protocol,
                      "copied_protocol": row.get("protocol"),
                      "continue_chains": cmode, "copied_from": oldbase,
                      "continued": False, "continued_problem": bool(i in cont_i),
                      "stop_reason": stop_reason_of(row.get("stop_marker"), row.get("n_trace"),
                                                    cont["old_horizon"],
                                                    stops=task_cfg(a.task)["stops"],
                                                    think_text=cont["think_text"],
                                                    think_id=cont["think_id"],
                                                    eos_texts=eos_txt_c, eos_ids=eos_ids),
                      "chain_tail": None, "own_answer_span": None}
                if i in cont_i:
                    ex["continue_old_stop"] = cont["selected"].get(ridx)
                    nc = int(row.get("n_cut", B) or 0)
                    ctx = tok.decode(cur[i]["ids"][:nc], clean_up_tokenization_spaces=False)
                    ex["chain_tail"] = chain_tail_of(ctx)
                    ex["own_answer_span"] = own_answer_span(ctx, row.get("trace_answer"), own_mk_c)
                return ex

            n_copy, n_skip = copy_old_rows(cont["old_cells"], apc, idx_of, done, copy_all,
                                           copy_upto, extra_of)
            # a cap this job runs that the source grid never had (horizon mode adds the new horizon)
            n_share = 0
            for i in range(N):
                if i in cont_i:
                    continue
                have = {int(B): r for (ii, B), r in done.items() if ii == i}
                if not have:
                    continue
                for r in fill_shared_cuts(apc, all_caps, have,
                                          lambda B: bool(B in extra and B not in caps)):
                    done[(i, int(r["B"]))] = r
                    n_share += 1
            n_copy += n_share
            meta["continue"].update({"cells_shared_cut": n_share,
                                     "rows_continued": len(cont_i), "cells_copied": n_copy,
                                     "cells_above_the_old_stop": n_skip,
                                     "rows_copied_whole": len(copy_all)})
            save_meta()
            print("  continue: %d row(s) continued, %d copied whole; %d cell(s) copied from %s, "
                  "%d to regenerate" % (len(cont_i), len(copy_all), n_copy, oldbase, n_skip),
                  flush=True)
        for i in range(N):
            if i in incomplete:
                continue
            if cont and i not in cont_i:
                continue                          # copied verbatim, never regenerated
            old_stop = int(cont["selected"].get(int(rws[i]["idx"]), -1)) if cont else -1
            m = {}
            for B in all_caps:
                if cont and B <= old_stop:
                    # cut(B) = min(stop, B) = B under BOTH stop rules and the first B ids are the
                    # same ids, so the read-out is byte-identical to the old one: it was copied, not
                    # regenerated (and it stays comparable to the published grid, which a
                    # regeneration at a different batch composition would not be).
                    continue
                cut = min(st[i][0], B) if not forced else min(len(cur[i]["ids"]), B)
                m.setdefault(cut, []).append(B)
            for c, bs in m.items():
                jobs.append((i, c, bs))
        readout_pass(enc, cur, plen, st, suf, eos_ids, jobs, "single")

    apc.close()
    meta["cells_written"] = len(done)
    meta["cells_expected"] = N * len(all_caps)
    got = {}
    for B in all_caps:
        rowsB = [done[(i, B)] for i in range(N) if (i, B) in done]
        if not rowsB:
            continue
        got[str(B)] = {
            "n": len(rowsB),
            "parse_rate_forced": sum(1 for r in rowsB if r["pred"] is not None) / len(rowsB),
            "parse_rate_own": sum(1 for r in rowsB if r["trace_answer"] is not None) / len(rowsB),
            "acc_forced": sum(r["correct"] for r in rowsB) / len(rowsB),
            "acc_own": sum(r["trace_correct"] for r in rowsB) / len(rowsB),
            "acc_v2": sum(r["correct_v2"] for r in rowsB) / len(rowsB),
            "mean_layer_passes": sum(r["layer_passes"] for r in rowsB) / len(rowsB),
            "mean_n_cut": sum(r["n_cut"] for r in rowsB) / len(rowsB)}
        for sp in ("eval", "cal"):
            sub = [r for r in rowsB if r["split"] == sp]
            if sub:
                got[str(B)]["acc_v2_" + sp] = sum(r["correct_v2"] for r in sub) / len(sub)
                got[str(B)]["n_" + sp] = len(sub)
    meta["by_budget"] = got
    meta["batch_width_histogram"] = {"generation": gen_widths, "readout": ro_widths}
    meta["parse_rate_min"] = min([v["parse_rate_forced"] for v in got.values()] or [None])
    meta["seconds"] = round(time.time() - t0, 1)
    meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) \
        if torch.cuda.is_available() else None
    n_inc = int(meta.get("n_incomplete_rows") or 0)
    if meta["errors"] or n_inc or len(done) < meta["cells_expected"]:
        meta["complete"] = False
        save_meta()
        print("INCOMPLETE %s %d/%d cells (%d row(s) unscored) in %.0fs peak %s GB; errors %s"
              % (tag, len(done), meta["cells_expected"], n_inc, meta["seconds"], meta["peak_gb"],
                 meta["errors"]), flush=True)
        sys.exit(3)
    meta["complete"] = True
    save_meta()
    print("DONE %s %d/%d cells in %.0fs peak %s GB | acc_v2 %s"
          % (tag, len(done), meta["cells_expected"], meta["seconds"], meta["peak_gb"],
             {B: round(v["acc_v2"], 3) for B, v in got.items()}), flush=True)
    return meta


def _insert_tag(prompt, tl, task, chat_template):
    """Insert the S32 control line between the few-shot prompt and the question.

    The question begins at the task's `q_prefix` after the exemplar block, so the insertion point is
    the LAST occurrence of the exemplar block's trailing separator before that prefix. Implemented
    the way s32_common.build_prompts does it: the prompt is rebuilt as prefix + tag + question, so
    here the tag is spliced in front of the final `q_prefix`.
    """
    from .tasks import task_cfg as _cfg
    qp = _cfg(task)["q_prefix"]
    i = prompt.rfind(qp)
    if i < 0:
        return prompt
    return prompt[:i] + tl + prompt[i:]


if __name__ == "__main__":
    main()
