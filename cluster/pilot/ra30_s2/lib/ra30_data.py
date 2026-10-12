"""RA-30 data layer: pools, prompts, the Table 2 cut and parser, the precompute split (plan_ra30.md section 3).

Prompts and the parser are prod's (tree/prod/tasks, unchanged): both datasets use the GSM8K exemplar block, the
"Question: " prefix and the "\nAnswer:" cue; GSM8K under task gsm8k, ASDiv under task svamp (the cluster's
steer_gen_math registers ASDiv with SVAMP's prompt, stops and parser). The cut is prod's
make_find_cut (stops "\n\nQuestion:" and "Final Answer:", eos), the own answer prod's parse_own (number after the
last "####"), as in G.main.
"""
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TREE = os.path.join(ROOT, "tree")
DATA = os.path.join(ROOT, "data")

DATASETS = ("gsm8k", "asdiv")       # RA brief update: SVAMP and MAWPS dropped (too few multi-step chains)
BASE_TASK = {"gsm8k": "gsm8k", "asdiv": "svamp"}
PAIRS = {"gsm8k": ("varbench", "pairs_gsm8k.jsonl"), "asdiv": ("asdiv", "pairs_asdiv.jsonl")}
HORIZON = 512                 # setting a1 of steer_gen (Ouro-1.4B base, k = 4)
K = 4
SPLIT_N = {"main": 50, "pilot": 10}
SPLIT_KEY = "ra30:split:%s"
DEFAULT_MODEL = "ouro_1_4b_base"
MODELS = ("ouro_1_4b_base", "ouro_2_6b_base")     # the 2.6B axis: GSM8K and ASDiv (plan section 6b)


def model_suffix(model):
    """'' for the 1.4B files of record (their names never change), '_<model>' for every other model."""
    return "" if model in (None, DEFAULT_MODEL) else "_%s" % model


def sources_name(model=None):
    return "sources%s.json" % model_suffix(model)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def pairs_path(ds):
    return os.path.join(TREE, "data_steer", *PAIRS[ds])


def load_pool(ds, model=None):
    """data/pool_<ds>[_<model>].json written by make_data.py: {"items": [...], "source": ...} in the pool's order."""
    with open(os.path.join(DATA, "pool_%s%s.json" % (ds, model_suffix(model))), encoding="utf-8") as f:
        return json.load(f)


def load_split(ds, model=None):
    with open(os.path.join(DATA, "split_%s%s.json" % (ds, model_suffix(model))), encoding="utf-8") as f:
        return json.load(f)


def order_key(ds, model=None):
    return (SPLIT_KEY % ds) + model_suffix(model).replace("_", ":", 1)


def draw_order(ids, ds, model=None):
    """The run order of a pool: random.Random(SPLIT_KEY % ds) shuffle of the sorted problem ids. Drawn once, before
    any GPU step, and stored. The precompute split of the dataset-level neuron set is the first SPLIT_N[stage]
    eligible items (after the screen) in this order; the scored items are the eligible items after them."""
    xs = sorted(int(i) for i in ids)
    random.Random(order_key(ds, model)).shuffle(xs)
    return xs


def prompt_text(tok, ds, question):
    from prod.tasks import build_prompts
    rows = [{"input": question, "target": "", "options": [], "gold_text": "", "idx": 0}]
    prompts, _s, _st, _e, _x = build_prompts(tok, BASE_TASK[ds], rows, chat_template=False)
    return prompts[0]


def prompt_ids(tok, ds, question):
    return tok(prompt_text(tok, ds, question), add_special_tokens=False)["input_ids"]


def stops_and_eos(tok, ds):
    from prod.tasks import task_cfg
    return list(task_cfg(BASE_TASK[ds])["stops"]), [tok.eos_token_id]


def cut_parse(tok, ds, chain_ids):
    """(cut, cut_text, own): G.main's strip at eos, natural cut and own-answer parse on chain ids (prompt
    excluded)."""
    from prod.models.ouro import strip_tail
    from prod.tasks import make_find_cut, parse_own
    stops, eos = stops_and_eos(tok, ds)
    ids = strip_tail([list(chain_ids)], eos)[0]
    find_cut = make_find_cut(tok, stops, eos, chat_template=False, eos_cut=True)
    c, _mk = find_cut(ids)
    txt = tok.decode(ids[:c], clean_up_tokenization_spaces=False)
    return c, txt, parse_own(txt, BASE_TASK[ds])
