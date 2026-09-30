"""run_v21_4_harness — ENTRADA OFICIAL NO COLAB T4 (números que VALEM, regra 1.3).

Pergunta que esta versão responde:
    O harness antigo (lote fixo) mediu MEMORIZAÇÃO e saturou no chão 0.003 -> Δ=0 e
    nenhuma arquitetura se distinguiu (v21.2, v21.3). Este run troca o INSTRUMENTO
    de medição (não a arquitetura): tarefa de INDUÇÃO com validação congelada, que
    exige atenção e NÃO pode ser memorizada. Se a atenção viva (layer_scale=0.1)
    ganhar da morta (=0.0) aqui, o teste recuperou poder de discriminação.

Isto NÃO é uma mudança de arquitetura: nenhuma variável da Mycelium-LM muda.
Portanto NÃO promove nem rebaixa nada — gera evidência para decidir o harness oficial.

Uso (Colab, 1 célula):
    !python run_v21_4_harness.py                    # induction+mem, LS 0.0/0.1, 3 seeds
    !python run_v21_4_harness.py --steps 500
    !python run_v21_4_harness.py --tasks induction --variants 0.0 0.1 --seeds 0 1 2

Escreve em bundles/v21.4-harness/: metrics_3seeds.csv, stab_per_step.csv,
diag.json, diag_geometry.csv, teste_contexto.log, veredito.md.

Contribuição: DeepSeek (v21.4 — noite 2026-09-30).
"""
import argparse
import csv
import json
import pathlib
import time

import numpy as np
import tensorflow as tf

from diag_v21 import diag_model
from model_v21 import ConsciousV21
from runner_v21 import context_test, set_seeds, train_run
from stab_v21 import StabilityLogger
from tasks_v21 import eval_loss, make_env

VER = "v21.4-harness"
OUT = pathlib.Path(f"bundles/{VER}")
CFG_PATH = OUT / "config.json"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["induction", "mem"])
    ap.add_argument("--variants", nargs="*", type=float, default=[0.0, 0.1])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--V", type=int, default=512)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    return ap.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else {"seeds": [0, 1, 2], "clip_norm": 1.5}
    seeds = args.seeds if args.seeds else cfg["seeds"]
    clip_norm = cfg.get("clip_norm", 1.5)

    print(f"== {VER} == tasks={args.tasks} variants={args.variants} seeds={seeds} "
          f"steps={args.steps} V={args.V} L={args.L} B={args.B} lr={args.lr} clip={clip_norm}", flush=True)

    log = StabilityLogger(OUT / "stab_per_step.csv")
    rows, console = [], []
    last_model = None

    for task in args.tasks:
        for ls in args.variants:
            for sd in seeds:
                t0 = time.time()
                set_seeds(sd)
                m = ConsciousV21(coupling=True, langevin=False, layer_scale_init=ls)
                env = make_env(task, V=args.V, L=args.L, B=args.B, seed=sd, n_val=8)
                label = f"{task}/ls{ls}"
                r = train_run(m, env, args.steps, V=args.V, clip_norm=clip_norm, lr=args.lr,
                              log=log, grad_stats_every=50, snap_steps=(0, 1, args.steps - 1),
                              verbose_every=100, label=f"{label}/s{sd}")
                ct = context_test(m, V=args.V, L=args.L)
                r["context_test"] = ct
                r["wall_s"] = time.time() - t0
                rows.append([task, ls, sd, f"{r['loss_train']:.5f}", f"{r['loss_val']:.5f}",
                             f"{r['perplexidade_val']:.5f}", f"{r['overfitting_gap']:.5f}",
                             f"{r['clip_rate']:.4f}", f"{r['grad_norm_final']:.5f}",
                             r["g_nan_total"], r["g_inf_total"],
                             f"{r['residual_gain']['end'][0]['r_attn_mean']:.5f}",
                             ct["usa_contexto"], f"{r['wall_s']:.1f}"])
                console.append(f"{task} ls={ls} s={sd}: loss_val={r['loss_val']:.5f} "
                               f"train={r['loss_train']:.5f} clip={r['clip_rate']:.2f} "
                               f"r_attn={r['residual_gain']['end'][0]['r_attn_mean']:.4f} "
                               f"nan={r['g_nan_total']} ctx={ct['usa_contexto']}")
                print("  " + console[-1], flush=True)
                last_model = m

    log.flush()
    with open(OUT / "metrics_3seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "layer_scale_init", "seed", "loss_train", "loss_val", "ppl_val",
                    "overfitting_gap", "clip_rate", "grad_norm_final", "g_nan", "g_inf",
                    "r_attn_L0", "usa_contexto", "wall_s"])
        w.writerows(rows)

    (OUT / "teste_contexto.log").write_text("\n".join(console) + "\n")

    if last_model is not None:
        diag_model(last_model, VER, OUT, extra={"steps": args.steps, "tasks": args.tasks})

    # resumo estatístico honesto: média por (task, ls) e Δ
    summ = {}
    for task in args.tasks:
        for ls in args.variants:
            vals = [float(r[4]) for r in rows if r[0] == task and abs(r[1] - ls) < 1e-12]
            summ[f"{task}|ls{ls}"] = {"n": len(vals), "loss_val_mean": float(np.mean(vals)) if vals else None,
                                      "loss_val_std": float(np.std(vals)) if vals else None,
                                      "loss_val_list": [round(v, 5) for v in vals]}
    verdict = {}
    for task in args.tasks:
        a = summ.get(f"{task}|ls{args.variants[0]}")
        b = summ.get(f"{task}|ls{args.variants[-1]}")
        if a and b and a["loss_val_mean"] is not None and b["loss_val_mean"] is not None:
            verdict[task] = {"delta_ls_last_minus_first": round(b["loss_val_mean"] - a["loss_val_mean"], 5)}
    (OUT / "resumo_v21_4.json").write_text(json.dumps({"summary": summ, "delta": verdict,
                                                       "args": vars(args)}, indent=2))

    print("\n=== RESUMO ===")
    print(json.dumps({"summary": summ, "delta": verdict}, indent=2))
    print(f"\nok -> {OUT}/metrics_3seeds.csv + stab_per_step.csv + diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
