"""noise_floor_v21_4 — qual é a resolução real do instrumento? (regra 0: sem número comparável, sem claim)

Motivação: a verificação de reprodutibilidade (verify_v21_4.py) re-rodou a MESMA config,
mesma seed, e obteve loss_val 1.49745 contra 1.51270 registrado — Δ = 0,0152. Isso é 15x
a terceira casa decimal que a regra 2.1 usa para ranking. Se o ruído run-to-run for dessa
ordem, qualquer Δ de 0,003–0,010 (ex.: os que embasaram v21.1/v21.2/v21.3) está DENTRO do
ruído — ou seja, não é evidência.

Este script mede:
  - ruído com 2 threads (config atual): N repetições idênticas -> dispersão
  - determinismo com 1 thread: 2 repetições idênticas -> reproduz bit-a-bit?

Uso:  python noise_floor_v21_4.py --threads 2 --reps 3 --steps 150
Escala reduzida = DEBUG (regra 1.3). O que vale generalizar é o PRINCÍPIO: medir o piso
de ruído do harness antes de aceitar qualquer Δ.

Contribuição: DeepSeek (noite 2026-09-30).
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")

    OUT = REPO / "bundles" / "v21.4-harness" / "noise"
    OUT.mkdir(parents=True, exist_ok=True)

    loss_vals, loss_trains = [], []
    for rep in range(a.reps):
        set_seeds(0)                      # MESMA seed em todas as repetições
        m = ConsciousV21(coupling=True, langevin=False, layer_scale_init=0.1)
        env = make_env("lag", V=16, L=32, B=8, seed=0, n_val=4)
        r = train_run(m, env, a.steps, V=16, clip_norm=1.0, lr=1e-3, log=None,
                      grad_stats_every=0, snap_steps=(0, a.steps - 1),
                      verbose_every=0, label=f"noise/t{a.threads}/r{rep}")
        loss_vals.append(r["loss_val"]); loss_trains.append(r["loss_train"])
        print(f"threads={a.threads} rep={rep} loss_val={r['loss_val']:.8f} "
              f"loss_train={r['loss_train']:.8f}", flush=True)

    d = float(max(loss_vals) - min(loss_vals))
    out = {
        "threads": a.threads, "reps": a.reps, "steps": a.steps,
        "loss_val_lista": loss_vals, "loss_train_lista": loss_trains,
        "spread_loss_val": d,
        "bit_exato": bool(d == 0.0),
        "nota": "delta menor ou igual a spread_loss_val nao e evidencia neste harness",
    }
    (OUT / f"noise_t{a.threads}.json").write_text(json.dumps(out, indent=2))
    print(f"\nthreads={a.threads} -> spread loss_val = {d:.8f} bit_exato={out['bit_exato']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
