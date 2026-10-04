"""run_v21_5_qkv — SONDA Q/K/V NO HARNESS OFICIAL (números que VALEM, regra 1.3).

Pergunta (falsificação pré-registrada, mesma do probe local):
    A v21 não aprende `induction` por falta de roteamento por conteúdo?
    -> Q/K/V (1 variável, `model_qkv_probe.py`) deve tirar a loss_val do uniforme.

Desenho: fixo vs qkv × induction+lag × 3 seeds × 300 steps, mesmos
hiperparâmetros do `run_v21_4_harness.py` (V=512 L=32 B=2 lr=1e-4 clip=1.5).
`lag` entra como guarda contra regressão (QKV não pode quebrar o que funciona).

Critério: qkv/induction < uniforme*0.95 (uniforme ln512=6.238) = CONFIRMADA.
Isto NÃO promove nada sozinho — se confirmada, a QKV vira v21.5-candidata.

Uso (Colab, 1 célula): !python run_v21_5_qkv.py
Saídas em bundles/v21.5-qkv/: metrics_3seeds.csv, resumo_v21_5.json, diag.json.
"""
import argparse
import csv
import json
import math
import pathlib
import time

import numpy as np
import tensorflow as tf

from diag_v21 import diag_model
from model_v21 import ConsciousV21
from model_qkv_probe import ConsciousV21QKVProbe
from runner_v21 import context_test, set_seeds, train_run
from stab_v21 import StabilityLogger
from tasks_v21 import make_env

VER = "v21.5-qkv"
OUT = pathlib.Path(f"bundles/{VER}")
UNIFORME = math.log(512)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["fixo", "qkv"])
    ap.add_argument("--tasks", nargs="*", default=["induction", "lag"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--V", type=int, default=512)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ls-init", type=float, default=0.1)
    ap.add_argument("--freeze-ls", action="store_true",
                    help="sonda gate-fixo: layer_scale congelado (não entra no otimizador)")
    return ap.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clip_norm = 1.5
    print(f"== {VER} == models={args.models} tasks={args.tasks} seeds={args.seeds} "
          f"steps={args.steps} V={args.V} (uniforme={UNIFORME:.5f}) "
          f"ls_init={args.ls_init} freeze_ls={args.freeze_ls}", flush=True)

    log = StabilityLogger(OUT / "stab_per_step.csv")
    rows, console = [], []
    last_model = None

    for model_tag in args.models:
        cls = ConsciousV21QKVProbe if model_tag == "qkv" else ConsciousV21
        for task in args.tasks:
            for sd in args.seeds:
                t0 = time.time()
                set_seeds(sd)
                m = cls(coupling=True, langevin=False, layer_scale_init=args.ls_init,
                        ls_trainable=not args.freeze_ls)
                env = make_env(task, V=args.V, L=args.L, B=args.B, seed=sd, n_val=8)
                label = f"{model_tag}/{task}"
                r = train_run(m, env, args.steps, V=args.V, clip_norm=clip_norm, lr=args.lr,
                              log=log, grad_stats_every=50, snap_steps=(0, 1, args.steps - 1),
                              verbose_every=100, label=f"{label}/s{sd}")
                ct = context_test(m, V=args.V, L=args.L)
                wall = time.time() - t0
                rows.append([model_tag, task, sd, f"{r['loss_train']:.5f}", f"{r['loss_val']:.5f}",
                             f"{r['clip_rate']:.4f}", f"{r['grad_norm_final']:.5f}",
                             r["g_nan_total"], r["g_inf_total"], ct["usa_contexto"], f"{wall:.1f}"])
                console.append(f"{model_tag} {task} s={sd}: loss_val={r['loss_val']:.5f} "
                               f"train={r['loss_train']:.5f} clip={r['clip_rate']:.2f} "
                               f"nan={r['g_nan_total']} ctx={ct['usa_contexto']}")
                print("  " + console[-1], flush=True)
                last_model = m

    log.flush()
    with open(OUT / "metrics_3seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "seed", "loss_train", "loss_val", "clip_rate",
                    "grad_norm_final", "g_nan", "g_inf", "usa_contexto", "wall_s"])
        w.writerows(rows)
    (OUT / "teste_contexto.log").write_text("\n".join(console) + "\n")
    if last_model is not None:
        diag_model(last_model, VER, OUT, extra={"steps": args.steps})

    summ = {}
    for model_tag in args.models:
        for task in args.tasks:
            vals = [float(r[4]) for r in rows if r[0] == model_tag and r[1] == task]
            summ[f"{model_tag}|{task}"] = {"n": len(vals), "mean": round(float(np.mean(vals)), 5),
                                           "std": round(float(np.std(vals)), 5),
                                           "list": [round(v, 5) for v in vals]}
    verdict = {}
    for task in args.tasks:
        a = summ.get(f"fixo|{task}")
        b = summ.get(f"qkv|{task}")
        if a and b:
            verdict[task] = {"fixo_mean": a["mean"], "qkv_mean": b["mean"],
                             "ganho_qkv": round(a["mean"] - b["mean"], 5),
                             "qkv_abaixo_uniforme": bool(b["mean"] < UNIFORME * 0.95),
                             "hipotese": "CONFIRMADA" if b["mean"] < UNIFORME * 0.95 else "FALSIFICADA"}
    (OUT / "resumo_v21_5.json").write_text(json.dumps(
        {"summary": summ, "verdict": verdict, "uniforme": round(UNIFORME, 5)}, indent=2))

    print("\n=== VEREDITO ===")
    print(json.dumps(verdict, indent=2))
    print(f"ok -> {OUT}/metrics_3seeds.csv + resumo_v21_5.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
