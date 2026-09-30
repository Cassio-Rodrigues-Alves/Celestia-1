"""probe_qkv_induction — teste de falsificação da hipótese "falta roteamento por conteúdo".

Hipótese:
    A v21 não aprende `induction` porque a mistura é um grafo FIXO (sem Q/K/V).
    -> adicionar Q/K/V (1 variável, arquivo `model_qkv_probe.py`) deve tirar a loss do uniforme.

Critério (pré-registrado, para não fazer análise retroativa):
    - Se com a sonda QKV a loss_val de `induction` cair ABAIXO de (uniforme - 5%), a hipótese
      é CONFIRMADA (a lacuna era roteamento por conteúdo).
    - Se ficar em ~uniforme, a hipótese é FALSIFICADA e a causa é outra (otimização/escala).

Escala reduzida, CPU local -> DEBUG (regra 1.3: não vale para ranking).

Contribuição: DeepSeek (noite 2026-09-30).
"""
import json
import math
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
from model_qkv_probe import ConsciousV21QKVProbe  # noqa: E402
from runner_v21 import set_seeds, train_run  # noqa: E402
from tasks_v21 import make_env  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")

V, L, B, STEPS, LS = 16, 32, 8, 400, 0.1
UNIFORME = math.log(V)
OUT = REPO / "bundles" / "v21.4-harness" / "probe_qkv"
OUT.mkdir(parents=True, exist_ok=True)

res = {}
for tag, cls in [("v21_fixo", ConsciousV21), ("v21_qkv", ConsciousV21QKVProbe)]:
    for task in ("induction", "lag"):
        set_seeds(0)
        m = cls(coupling=True, langevin=False, layer_scale_init=LS)
        env = make_env(task, V=V, L=L, B=B, seed=0, n_val=4)
        r = train_run(m, env, STEPS, V=V, clip_norm=1.0, lr=1e-3, log=None,
                      grad_stats_every=0, snap_steps=(0, STEPS - 1), verbose_every=100,
                      label=f"{tag}/{task}")
        res[f"{tag}::{task}"] = {k: v for k, v in r.items() if k != "_model"}
        print(f"[{tag}/{task}] train={r['loss_train']:.5f} val={r['loss_val']:.5f} "
              f"r_attn_L0={r['residual_gain']['end'][0]['r_attn_mean']:.4f}", flush=True)

print(f"\nuniforme (ln {V}) = {UNIFORME:.5f}")
verdict = {}
for task in ("induction", "lag"):
    a = res[f"v21_fixo::{task}"]["loss_val"]
    b = res[f"v21_qkv::{task}"]["loss_val"]
    ganho = a - b
    conf = b < UNIFORME * 0.95
    verdict[task] = {"fixo_val": round(a, 5), "qkv_val": round(b, 5), "uniforme": round(UNIFORME, 5),
                     "ganho_qkv": round(ganho, 5), "abaixo_do_uniforme": bool(conf)}
    print(f"{task:10s} fixo={a:.5f} qkv={b:.5f} ganho={ganho:+.5f} "
          f"qkv<uniforme={conf}")

(OUT / "probe_qkv.json").write_text(json.dumps({"uniforme": UNIFORME, "verdict": verdict,
                                                "resultados": res}, indent=2, default=str))
print(f"\nok -> {OUT}/probe_qkv.json")
