"""local_debug_v21_4 — VALIDAÇÃO LOCAL EM ESCALA REDUZIDA (SÓ DEBUG).

Regra 1.3: número local NÃO vale para ranking. Este script existe para provar que
o código novo (tasks_v21 + stab_v21 + runner_v21) RODA e para medir a propriedade
que motivou a v21.4: o harness antigo ('mem') satura e não discrimina; o harness
novo ('induction') discrimina atenção viva (layer_scale=0.1) de atenção morta (=0.0).

Config reduzida (NÃO é a config do projeto — é um microscópio):
    LAYERS=2, D_MODEL=64, VOCAB=128, SEQ_LEN=32, tarefas V=64 L=32 B=4.

Contribuição: DeepSeek (v21.4 — noite 2026-09-30).
"""
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))

# --- microscópio: reduz o modelo ANTES de instanciar (não altera model_v21.py) ---
import model_v21
model_v21.LAYERS = 2
model_v21.D_MODEL = 64
model_v21.VOCAB_SIZE = 128
model_v21.SEQ_LEN = 32

from model_v21 import ConsciousV21  # noqa: E402
from runner_v21 import context_test, set_seeds, train_run  # noqa: E402
from stab_v21 import StabilityLogger  # noqa: E402
from tasks_v21 import make_env  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")

STEPS = 400
LS_INIT = {"morta": 0.0, "viva": 0.1}
TASKS = ["mem", "lag", "induction"]
V_DEBUG, L_DEBUG, B_DEBUG = 16, 32, 8
OUT = REPO / "bundles" / "v21.4-harness" / "local_debug"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for task in TASKS:
    for name, ls in LS_INIT.items():
        set_seeds(0)
        m = ConsciousV21(coupling=True, langevin=False, layer_scale_init=ls)
        env = make_env(task, V=V_DEBUG, L=L_DEBUG, B=B_DEBUG, seed=0, n_val=4)
        log = StabilityLogger(OUT / f"stab_{task}_{name}.csv")
        r = train_run(m, env, STEPS, V=V_DEBUG, clip_norm=1.0, lr=1e-3, log=log,
                      resid_every=0, grad_stats_every=100, snap_steps=(0, 1, STEPS - 1),
                      verbose_every=50, label=f"{task}/{name}")
        r["context_test"] = context_test(m, V=V_DEBUG, L=L_DEBUG)
        r["_model"] = m
        log.flush()
        results[f"{task}::{name}"] = r
        print(f"[{task}/{name}] loss_train={r['loss_train']:.5f} loss_val={r['loss_val']:.5f} "
              f"clip={r['clip_rate']:.2f} r_attn_L0={r['residual_gain']['end'][0]['r_attn_mean']:.4f} "
              f"ctx={r['context_test']['usa_contexto']}", flush=True)

print("\n=== COMPARAÇÃO (atenção morta vs viva) ===")
for task in TASKS:
    a = results[f"{task}::morta"]
    b = results[f"{task}::viva"]
    d = a["loss_val"] - b["loss_val"]
    print(f"{task:10s} mem: {a['loss_val']:.5f} | viva: {b['loss_val']:.5f} | "
          f"delta(viva-morta)={-d:+.5f} | sat(morta)={a['sat_heuristic']}")

serial = {k: {kk: vv for kk, vv in v.items() if kk != "_model"} for k, v in results.items()}
(OUT / "local_debug.json").write_text(json.dumps(serial, indent=2, default=str))
print(f"\nok -> {OUT}/local_debug.json")
