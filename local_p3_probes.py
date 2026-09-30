"""local_p3_probes — P3 da fila: sondas copy + assoc + tabela final (SÓ DEBUG).

Pergunta: qual capacidade existe hoje? `copy`/`assoc` exigem atenção por conteúdo;
a v21 não tem Q/K/V -> devem falhar (resultado esperado = medir a lacuna).
Roda copy+assoc × morta/viva e junta com `induction` já medido na P1
(bundles/v21.4-harness/local_debug/local_debug.json) na tabela final.

Mesmo microscópio e hiperparâmetros da P1. Regra 1.3: número local NÃO vale p/ ranking.
Contribuição: execução da fila PROXIMOS_PASSOS.md (P3).
"""
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))

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
TASKS = ["copy", "assoc"]
V_DEBUG, L_DEBUG, B_DEBUG = 16, 32, 8
OUT = REPO / "bundles" / "v21.4-harness" / "p3_probes"
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
        log.flush()
        results[f"{task}::{name}"] = {k: v for k, v in r.items() if k != "_model"}
        print(f"[{task}/{name}] loss_train={r['loss_train']:.5f} loss_val={r['loss_val']:.5f} "
              f"clip={r['clip_rate']:.2f} ctx={r['context_test']['usa_contexto']}", flush=True)

# tabela final: copy+assoc (agora) + induction (P1, mesmo desenho)
table = {}
p1 = json.loads((REPO / "bundles" / "v21.4-harness" / "local_debug" / "local_debug.json").read_text())
for task in ["copy", "assoc"]:
    for name in ["morta", "viva"]:
        r = results[f"{task}::{name}"]
        table[f"{task}::{name}"] = {"loss_val": round(r["loss_val"], 5), "fonte": "P3"}
for name in ["morta", "viva"]:
    r = p1[f"induction::{name}"]
    table[f"induction::{name}"] = {"loss_val": round(r["loss_val"], 5), "fonte": "P1"}

uniforme = float(np.log(V_DEBUG))
print(f"\n=== P3: tabela de sondas (uniforme ln{V_DEBUG}={uniforme:.5f}) ===")
for k, v in table.items():
    status = "FALHA-esperada" if v["loss_val"] >= uniforme - 0.05 else "aprendeu?!"
    print(f"{k:18s} loss_val={v['loss_val']:.5f} [{status}] ({v['fonte']})")

(OUT / "p3_probes.json").write_text(json.dumps(results, indent=2, default=str))
(OUT / "probe_table.json").write_text(json.dumps(
    {"uniforme": uniforme, "tabela": table}, indent=2))
print(f"ok -> {OUT}/probe_table.json")
