# Hash map of the scrubbed inputs (2026-10-09)

The public repository must hold no person name, host name or home path. The PI decided to scrub the inputs below.
The person's name became "the PI". The host names became role words: fast host, shared machine, second shared
machine. Home paths became `<home>`. The GPU model name GB10 stays. No other byte changed. This file maps every old
hash to the new one, so that records written this week with the old hashes can still be traced.

## Old hash to new hash

| file in the bundle | old sha256 | new sha256 | what changed |
|---|---|---|---|
| `plans/k17_varbench_plan.md` | `16dd5706e84c1070cf58212e5eaec92d03c6db5262d2cb0d65200fd9c9b1549c` | `cf92be3aa318d676eccddefadf295c2fda81c406b51a99eeeddc838a16d7a332` | person name on lines 49, 50, 565, 1066 and 1134 |
| `plans/k17_varbench_plan.md`, frozen part (`plan_frozen_sha256`: the file without section 13) | `35b379a5f7998f95...` | `198ff7e4a85e070a...` | lines 49, 50 and 565 are outside section 13 |
| `plans/k17_public_plan.md` | `ae1847d948bd992e91b506b6bbb655be4571387162e653ff274ddc68fc3e9e3a` | `681f5dacfff96e1b6e6fe6e347623227c72bf04f359759db371a052d2ac9606a` | 43 host names (shared machine), 1 host name (fast host), 1 home path, 1 person name; then (second scrub, `9f27da60...` -> `681f5dac...`) two attribution mentions removed on lines 778 and 791, and two role words after a line break set in lower case (lines 708 and 790) |
| `plans/k17_parcae_plan.md` (copy of the notes plan) | `fc8801e3bad7a63080a5ec346a807e4dafbaa629d849cbf7eb0ca6d5499ca2d9` | not shipped | the PI chose the plan that the pilot full run hashed |
| `plans/k17_parcae_plan.md` (plan the pilot full run hashed, `k17_parcae/full/plan_at_launch.md`) | `d055dd4d0255e7447554acf59bf451e45e724b5f8035f447a6037a42aac04abd` | `f697c6b9ea96e1868f27d02180928b2b44fc76fc17fe1dd1ac84b866317a61a9` | 8 host names, 4 home paths, 6 person names |
| `data/cells_ouro_1_4b_think_gsm8k_natural2_k4.jsonl` (SETTLE) | `fca99f85e15b8803ff0d42265b2a746bfae69cfd910743df0400aaa5babe22e2` | `90ae08698e391f0c3a3e184920b061c4eac138ca7f920138c9d963dc2e1f3b5a` | header line only: `continue_chains_path` and `continue_source_cells` now start with `<home>/loopedcot/artifacts/`. Lines 2-13191 are byte-identical (sha256 of lines 2 onward `c0b74c7bff3d30f17758dde33f89f7d9055f2628ee8804fc7c327fb127a0d75a`, before and after) |
| `data/cells_ouro_2_6b_base_gsm8k_natural_k4.jsonl` (SETTLE, added 2026-10-10) | `a446b64b14b937046603e11691018ac262f61504cfa26a82ad6a749497add354` | `a446b64b14b937046603e11691018ac262f61504cfa26a82ad6a749497add354` | unchanged; the same hash as the settle value in the a2_gsm8k hash line of the run on our cards |
| `data/cells_ouro_1_4b_think_svamp_natural2_k4.jsonl` (SETTLE, added 2026-10-10) | `a285e75fe758a07e0d4f1acaa2094014abeaef88dac2c709ea554dc9d614b7e8` | `f574a138718439942b848ac3e8c862fefc3fd9de4a8788247ec6f0b2b1512bb2` | header line only: the home path in `continue_chains_path` now starts with `<home>/loopedcot/artifacts/`. Lines 2-3001 are byte-identical (sha256 of lines 2 onward `5f86ba629ac3b30578283396034061818c4915dd866ba63b055bda11db13d148`, before and after) |
| `SMOKE_HASHES.txt` | `c40f04ec7bb5af084dcb684d24d0d61a7e297a3400fac71aace839beb4f9bb0b` | `a203441ecd02a200dc3a9f4ca774a3c7617f9a691b449942963b472f374d90cb` | home path in the comment on line 489. Every hash line is unchanged |

`plans/k17_freeze_plan.md` (`de416161...`) and `plans/k17_freeze_raven_plan.md` (`ee8c583f...`) needed no change.

## Where the old hashes appear (this week's records)

Paths below are relative to the pilot artifact folder (`DONE_SRC`). Line numbers refer to `LEDGER.md`.

- `16dd5706` (varbench plan, whole file):
  - `cluster_local/g1_copy/B/k17_varbench/{ouro,ouro26}`, 3 files each
  - `cluster_local/g1_copy/hashes_g1_copy.txt`
  - `cluster_local/ra20_c0t/B/k17_varbench/ouro_think` (3 files) and `ouro26_think` (1 file)
  - `cluster_local/ra20_c0t/_hashes` (2 files)
  - `cluster_local/ra20_cl/B/k17_varbench/{ouro,ouro26}`, 1 file each
  - LEDGER lines 428 and 450
- `35b379a5` (varbench plan, frozen part): the `plan_launch` records (`vb_<model>.json`, shard files, `checks.json`,
  `results.json`) of the same four VarBench models in `cluster_local/g1_copy`, `cluster_local/ra20_c0t` and
  `cluster_local/ra20_cl`.
- `ae1847d9` (public plan): `k17_public/fill/` files `checks.json`, `checks_fillA_VB.json`, `checks_fillB_D4.json`,
  `hash_log.json`, `inputs_huginn.json` and `meta_huginn.json`. LEDGER lines 400 and 434.
- `d055dd4d` (Parcae plan of the full run): `k17_parcae/full/` and `k17_parcae/full_final/` files `checks.json` and
  `summary.md`. LEDGER lines 360, 464 and 466.
- `fc8801e3` (notes Parcae plan): no record. LEDGER line 464.
- `fca99f85` (SETTLE file): no record in the pilot artifact folder and no LEDGER line.

## What the cluster run records

- **VarBench.** Cluster phase-W lines pass `--plan $VB_PLAN --plan_sha $VB_PLAN_SHA`. `setup_cluster.sh` computes
  `VB_PLAN_SHA` from the copy (`cf92be3a...`), so cluster records carry the new plan hash and the new frozen hash.
  Records from this week carry the old ones.
  - The launch check (`k17_varbench.plan_launch_check`) was run on the new plan with the pinned code. Phase W accepts
    `cf92be3a...` and refuses `16dd5706...`. The `Launch code sha256` line is unchanged. A phase-M plan copy written
    by `vb_slots.py` passes the frozen-part rule.
  - The frozen-part rule refuses a phase W on the old plan followed by a phase M on the new one. No job does this.
    The one partly done pipeline (`ouro26_think`: window and scan ran on our cards) runs phase M on its staged
    `plan_M_ouro26_think.md`, built from the old plan. Its phase-W record and plan copy therefore agree.
- **Public.** Cluster k17_public records carry the new public plan hash (`681f5dac...`; the intermediate `9f27da60...` was never used by a run).
- **Parcae.** Cluster k17_parcae records carry `f697c6b9...` in place of `d055dd4d...`.
- **SETTLE.** `prod/steer_gen.py` (prep, screen, select) writes only the file name (`settle_file` in `prep.json`). It
  writes no sha256 of a SETTLE input, and `steer_job.json` and the meta files hold none either. So no generation
  record carries the old or the new hash of the scrubbed file. `cluster/pilot/steer_gen_raven.py` writes
  `settle_sha256`, but only for its own McLeish SETTLE file, which is not in this bundle and was not changed. The
  reader (`steer_gen.settle_cells`) skips the header line, so the rows the runs read are byte-identical to this
  week's.

## Amendment of 2026-10-11 (B-EXT2 item 5, ordered by the PI)

Reason: --vbd_v1 flag added for the IFM V1 rerun; default unchanged.

| file in the bundle | old sha256 | new sha256 | what changed |
|---|---|---|---|
| `pilot/k17_varbench.py` | `a161b00c06238fc00653853306488318ae4e42a09d57564b56c579ee18714468` | `43808060ac3948b6c3c0cd91f0edc06bacfe4254f982ece05c0253dca9f2c0d1` | the optional per-model flag --vbd_v1 (3 added lines and a comment) |
| `plans/k17_varbench_plan.md` | `cf92be3aa318d676eccddefadf295c2fda81c406b51a99eeeddc838a16d7a332` | `10beefdc3992c9d637d7b2add0f2963c1cafd5357efe9fd58e4bfffd612a9f1f` | the section 9 "Launch code sha256" line names the new code sha256 |
| `plans/k17_varbench_plan.md`, frozen part (`plan_frozen_sha256`) | `198ff7e4a85e070a109b21b0776d8a2f1759b914c4ed5483ae8eda86450c3222` | `0b7cae853b866f22c011c8014acce2e8d51e08018ae9ecdbfd4f62998e55b2ae` | the same line (outside section 13) |

`VB_PLAN_SHA` changes with the plan: setup (run_all.sh) must run again before the next chain. The finished
k17_varbench pipelines keep their records and their plan copies (old hashes); the IFM V1 pipelines (group
g12_ifmv1, folders `B/k17_varbench/<model>_v1`) run window, scan, rows, merge and summarize on the new plan and code.

