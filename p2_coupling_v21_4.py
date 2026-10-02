"""p2_coupling_v21_4 — P2: isolar o ACOPLAMENTO no harness novo.

Pergunta: com `layer_scale` fixo em 0.1 (atenção viva nos dois lados), o acoplamento
`w_hu/w_uh/w_uu` (coupling=True) muda o resultado na tarefa `lag`, comparado a
coupling=False? Antes isto era impossível de medir com confiança porque a mesma seed
dava inicializações diferentes (ruído ~0.027 >> Δ). Com a seed determinística
(`runner_v21.set_seeds`) a comparação de UMA variável passou a ser válida.

Critério pré-registrado: se |Δloss_val| <= piso de ruído medido para esta config, o
veredito é "indistinguível neste orçamento" (não se promove nada). Escala reduzida =
DEBUG (regra 1.3).

Contribuição: DeepSeek (continuação, 02/10/2026).
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
from runner_v21 import set_seeds, train_run  # noqa: E402
from tasks_v21 import make_env  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")

V, L, B, STEPS, LS, LR, CLIP = 16, 32, 8, 300, 0.1, 1e-3, 1.0
SEEDS = (0, 1)
OUT = REPO / "bundles" / "v21.4-harness" / "p2_coupling"
OUT.mkdir(parents=True, exist_ok=True)

res = {}
for coupling in (False, True):
    for sd in SEEDS:
        set_seeds(sd)
        m = ConsciousV21(coupling=coupling, langevin=False, layer_scale_init=LS)
        env = make_env("lag", V=V, L=L, B=B, seed=sd, n_val=4)
        r = train_run(m, env, STEPS, V=V, clip_norm=CLIP, lr=LR, log=None,
                      grad_stats_every=0, snap_steps=(0, STEPS - 1),
                      verbose_every=100, label=f"coupling={coupling}/s{sd}")
        tel = m.blocks[0][0].telemetry()
        r["nhu"] = tel["nhu"]; r["nuh"] = tel["nuh"]
        res[f"coupling{coupling}_s{sd}"] = {k: v for k, v in r.items() if k != "_model"}
        print(f"coupling={coupling} seed={sd}: loss_val={r['loss_val']:.6f} "
              f"train={r['loss_train']:.6f} clip={r['clip_rate']:.2f} "
              f"r_attn={r['residual_gain']['end'][0]['r_attn_mean']:.4f} "
              f"nhu={tel['nhu']:.4f} nuh={tel['nuh']:.4f}", flush=True)

porc = {}
for coupling in (False, True):
    vals = [res[f"coupling{coupling}_s{sd}"]["loss_val"] for sd in SEEDS]
    porc[coupling] = {"mean": float(np.mean(vals)), "vals": vals}
media_f = porc[False]["mean"]; media_t = porc[True]["mean"]
delta = media_t - media_f
# piso de ruído desta config: medido em noise/ (spread 0.0 com seed determinística)
piso = 0.0
veredito = "INDISTINGUIVEL neste orcamento" if abs(delta) <= piso else "DIFERENCA ACIMA DO PISO"
out = {"config": {"V": V, "L": L, "B": B, "steps": STEPS, "layer_scale_init": LS,
                  "lr": LR, "clip_norm": CLIP, "seeds": list(SEEDS), "task": "lag"},
       "resultados": res, "por_coupling": porc,
       "delta_loss_val": delta, "piso_ruido": piso, "veredito": veredito,
       "nota": "escala reduzida = debug (regra 1.3); nao promove nada"}
(OUT / "p2_coupling.json").write_text(json.dumps(out, indent=2, default=str))
print(f"\ndelta(coupling=True - False) = {delta:+.6f} | piso={piso} -> {veredito}")
print(f"ok -> {OUT}/p2_coupling.json")
