# K17 VarBench placement (B-K17-VARBENCH, code v6)

Copy-read target (A3). Source cells: e_LO = full-span log-odds effect; answer cells: e = first-token margin effect (plan section 5 v3). Both only where the clean gap >= 2.0. Cells: primary mean [95% cluster bootstrap CI] / flip. '*' = the RND cell at that site fails T3: the O1 value is cited, and a lower-bound claim there also needs the O1 - RND CI lower bound > 0 (checks.json placement.<cell>.rnd_pair, T3b v6, review C-F2).

## ifm_p1_ouro_inj (K 4, layers per loop 28, set red 17 sites, cluster)

Items: {"scanned": 913, "kept": 814, "A_fail": 88, "B_fail": 11, "subset_W": 20}. Hook checks all_ok True. PC-final exact {'exact': 814, 'n': 814}, SZ exact {'exact': 1628, 'n': 1628}.

| op pos (endpoint) | L0@1 | Lmid@1 | Llast@1 | L0@2 | Lmid@2 | Llast@2 | L0@3 | Lmid@3 | Llast@3 | L0@4 | Lmid@4 | Llast@4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O1 src (e_LO) | 0.00 [0.00, 0.01] / 0.00 | 0.00 [0.00, 0.01] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.02 [0.01, 0.02] / 0.00 | 0.01 [0.01, 0.02] / 0.00 | 0.01 [0.01, 0.01] / 0.00 | 0.05 [0.05, 0.06] / 0.00 | 0.05 [0.04, 0.05] / 0.00 | 0.03 [0.03, 0.03] / 0.00 | 0.97 [0.97, 0.97] / 0.99 | 0.77 [0.76, 0.78] / 0.93 | 0.00 [0.00, 0.00] / 0.00 |
| O1 ans (e) | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 | 0.03 [0.02, 0.03] / 0.00 | 0.03 [0.02, 0.03] / 0.00 | 0.22 [0.22, 0.23] / 0.02 | 1.00 [1.00, 1.00] / 1.00 |
| RND src (e) | -0.00 [-0.00, -0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.01 [0.01, 0.01] / 0.00 | 0.02 [0.01, 0.02] / 0.00 | 0.02 [0.01, 0.02] / 0.00 | 0.00 [0.00, 0.00] / 0.00 |
| RND ans (e) | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | -0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.00 [-0.00, 0.00] / 0.00 | 0.05 [0.04, 0.05] / 0.00 |
| O2 src (e) | - | - | 0.00 [0.00, 0.00] / 0.00 | - | - | - | - | - | 0.03 [0.03, 0.03] / 0.00 | - | - | - |
| O2 ans (e) | - | - | 0.00 [-0.00, 0.00] / 0.00 | - | - | - | - | - | 0.03 [0.03, 0.03] / 0.00 | - | - | - |

Answer classes (a = 1): {"L0@1": "gate fails (absent or overwritten within loop j)", "Lmid@1": "not carried", "Llast@1": "not carried", "L0@2": "not carried", "Lmid@2": "not carried", "Llast@2": "not carried", "L0@3": "not carried", "Lmid@3": "not carried", "Llast@3": "not carried"}

Window dep (subset n 20): {"dir|full": 0.76646, "dir|after_Llast@1": 0.76551, "dir|after_Lmid@2": 0.76491, "dir|after_Llast@2": 0.7664, "dir|after_Lmid@3": 0.76349, "dir|after_Llast@3": 0.75949, "dir|after_Lmid@4": 0.41931, "rel|full": 0.83829, "rel|after_Llast@1": 0.82759, "rel|after_Lmid@2": 0.80426, "rel|after_Llast@2": 0.79671, "rel|after_Lmid@3": 0.77948, "rel|after_Llast@3": 0.77878, "rel|after_Lmid@4": 0.43767, "rand_rel|full": -9e-05}
Undo slots selected: []; slot table {}

Carry gain (dev full set at a = 1: hit / band chance / regret; test: predicted - dev-tuned, - uniform):
- VarBench: dev eligible 0, hit None, hit - latest None, hit - Llast@K-1 (loops < K) None, chance None, regret None, FN None; verified-9 eligible 0/0; working-site fraction None; dev-tuned None; pred-dev None; pred-uniform None
- VBD: dev eligible 0, hit None, hit - latest None, hit - Llast@K-1 (loops < K) None, chance None, regret None, FN None; verified-9 eligible 0/0; working-site fraction None; dev-tuned None; pred-dev None; pred-uniform None
- V6 gate (plan critic 3 F4): CGCHECK pass None, family FD pass None (null), bf16 vs fp32 G null -> interpretable False
- Undo v5 (scored on M-only items): {}
- VBD (derived-use, first-token e): items {"scanned": 848, "kept": 797, "A_fail": 37, "B_fail": 14}; PC-final {'exact': 797, 'n': 797, 'e': [1.0, 1.0, 1.0, 797, 343]}; SZ {'exact': 1594, 'n': 1594}; src Llast@1 [0.006, 'O1']; ans Llast@1 [0.0007, 'O1'], Llast@2 [0.0084, 'O1'], Llast@3 [0.1718, 'O1'], Lmid@4 [0.969, 'O1 (RND fails T3: lower bounds gated on O1 - RND)']

## Predictions

Tally (harness rows apart): {"tallied": {"held": 25, "missed": 12, "not scored": 511}, "harness": {"held": 6, "missed": 0, "not scored": 18}}
