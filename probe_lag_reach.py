"""probe_lag_reach — até onde o grafo fixo enxerga? (sonda de alcance posicional)

Tarefa lagJ (eco com atraso nominal J = J−2 efetivo) × seeds, modelo fixo
(v21.1: coupling, LS 0.1, sem Langevin). 1 variável contra a baseline: só J muda.
Pergunta: onde fica o penhasco (dentro da janela k=6 vs além, via shortcuts/multi-hop)?

Uso:
    python probe_lag_reach.py --delays 4 --seeds 0 --steps 400   # smoke local
    python probe_lag_reach.py                                    # matriz cheia
Saída em bundles/v21.4-harness/probe_lag_reach/probe_lag_reach.json.
Escala reduzida = DEBUG (regra 1.3).
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import tensorflow as tf

REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))

import model_v21  # noqa: E402

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

V, L, B, LS = 16, 32, 8, 0.1
UNIFORME = math.log(V)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", nargs="*", type=int, default=[2, 4, 6, 8, 10, 12, 16, 20])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=str(REPO / "bundles" / "v21.4-harness" / "probe_lag_reach"))
    return ap.parse_args()


def main():
    a = parse()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"alcance posicional: atrasos={a.delays} (efetivos = nominal−2) seeds={a.seeds} "
          f"passos={a.steps} uniforme={UNIFORME:.5f}", flush=True)
    rows = {}
    for j in a.delays:
        for sd in a.seeds:
            t0 = time.time()
            set_seeds(sd)
            m = ConsciousV21(coupling=True, langevin=False, layer_scale_init=LS)
            env = make_env(f"lag{j}", V=V, L=L, B=B, seed=sd, n_val=4)
            r = train_run(m, env, a.steps, V=V, clip_norm=1.0, lr=1e-3, log=None,
                          grad_stats_every=0, snap_steps=(0, a.steps - 1), verbose_every=0,
                          label=f"lag{j}/s{sd}")
            rec = {k: v for k, v in r.items() if k != "_model"}
            rec["wall_s"] = round(time.time() - t0, 1)
            rows[f"lag{j}::s{sd}"] = rec
            print(f"[lag{j:2d}/s{sd}] train={r['loss_train']:.5f} val={r['loss_val']:.5f} "
                  f"nan={r['g_nan_total']} ({rec['wall_s']}s)", flush=True)
    summary = {}
    for j in a.delays:
        vals = [rows[f"lag{j}::s{sd}"]["loss_val"] for sd in a.seeds]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[f"lag{j}"] = {"efetivo": j - 2, "mean": round(mean, 5), "std": round(std, 5),
                              "aprendeu": bool(mean < 0.95 * UNIFORME)}
    (out / "probe_lag_reach.json").write_text(json.dumps(
        {"uniforme": UNIFORME, "args": vars(a), "resumo": summary, "runs": rows},
        indent=2, default=str))
    print(f"\n=== alcance (nominal → efetivo; janela k=6) ===")
    for k, r in summary.items():
        mark = "OK " if r["aprendeu"] else "FALHOU"
        print(f"{k:8s} (ef {r['efetivo']:2d}): {r['mean']:.5f}±{r['std']:.5f} [{mark}]")
    print(f"ok -> {out}/probe_lag_reach.json")


if __name__ == "__main__":
    main()
