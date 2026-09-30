"""local_p2_coupling — P2 da fila: acoplamento isolado no harness novo (SÓ DEBUG).

Pergunta: `w_hu/w_uh/w_uu` mudam o resultado ALÉM do `layer_scale`?
Desenho: tarefa `lag`, LS fixo 0.1, coupling False vs True. Todo o resto igual à P1
(microscópio 2L/d64, V=16 L=32 B=8, 400 passos, lr 1e-3, clip 1.0, seed 0).

Regra 1.3: número local NÃO vale para ranking.
Contribuição: execução da fila PROXIMOS_PASSOS.md (P2).
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
TASK = "lag"
LS_FIXED = 0.1
COUPLING = {"sem": False, "com": True}
V_DEBUG, L_DEBUG, B_DEBUG = 16, 32, 8
OUT = REPO / "bundles" / "v21.4-harness" / "p2_coupling"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for name, co in COUPLING.items():
    set_seeds(0)
    m = ConsciousV21(coupling=co, langevin=False, layer_scale_init=LS_FIXED)
    env = make_env(TASK, V=V_DEBUG, L=L_DEBUG, B=B_DEBUG, seed=0, n_val=4)
    log = StabilityLogger(OUT / f"stab_lag_co_{name}.csv")
    r = train_run(m, env, STEPS, V=V_DEBUG, clip_norm=1.0, lr=1e-3, log=log,
                  resid_every=0, grad_stats_every=100, snap_steps=(0, 1, STEPS - 1),
                  verbose_every=50, label=f"lag/co_{name}")
    r["context_test"] = context_test(m, V=V_DEBUG, L=L_DEBUG)
    log.flush()
    results[f"lag::co_{name}"] = {k: v for k, v in r.items() if k != "_model"}
    print(f"[lag/co_{name}] loss_train={r['loss_train']:.5f} loss_val={r['loss_val']:.5f} "
          f"clip={r['clip_rate']:.2f} ctx={r['context_test']['usa_contexto']}", flush=True)

a, b = results["lag::co_sem"], results["lag::co_com"]
print(f"\n=== P2: lag sem vs com acoplamento (LS=0.1) ===")
print(f"sem: {a['loss_val']:.5f} | com: {b['loss_val']:.5f} | delta(com-sem)={b['loss_val'] - a['loss_val']:+.5f}")
(OUT / "p2_coupling.json").write_text(json.dumps(results, indent=2, default=str))
print(f"ok -> {OUT}/p2_coupling.json")
