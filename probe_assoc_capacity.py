"""probe_assoc_capacity — a memória associativa causal (CEL-2.001) resolve o roteamento por
CONTEÚDO que o grafo fixo da v21 não resolve?

STATUS: PROPOSTA (arquivo novo). Escala reduzida, CPU local -> DEBUG (regra 1.3: não vale
para ranking nem promoção).

Mesmo microscópio das sondas anteriores (P1/P3/probe_qkv): LAYERS=2, D_MODEL=64, V=16, L=32,
B=8, 400 passos, lr 1e-3, clip 1.0, layer_scale_init=0.1, coupling=True, langevin=False,
`tf.keras.utils.set_random_seed` (via runner_v21.set_seeds). Diferença: 3 seeds por célula
(as sondas anteriores usaram 1) — com 1 seed não há como separar efeito de variância.

## Critérios PRÉ-REGISTRADOS (definidos antes de rodar, sem análise retroativa)

C1 (capacidade)  Uma tarefa conta como "aprendida" por um braço se a média de loss_val
                 nas seeds < 0.95 * ln(V) (mesmo critério de probe_qkv_induction.py).
C2 (alvo)        Braço `replace` aprende `induction` (C1). A referência Q/K/V é 1.30733 e
                 é só REFERÊNCIA (não é limiar): atenção linear com cabeça única pode
                 ficar entre o uniforme e o softmax, e isso é informação.
C3 (sem regressão) Braço `hybrid` não pode piorar `lag` além de 0.05 sobre o grafo fixo
                 (referência 1.51304 do probe_qkv; o grafo fixo é o que resolve `lag`
                 por posição). Se piorar, o gate alpha está prejudicando o roteamento
                 posicional.
C4 (gate vivo)   Em `hybrid`, alpha médio final fora de [0.02, 0.98] indica colapso do
                 gate para um dos lados (a mistura virou "só grafo" ou "só memória").
C5 (honestidade) Variância entre seeds é reportada junto de toda média. Diferença entre
                 braços só é lida como efeito se maior que o desvio entre seeds.

Uso:
    python probe_assoc_capacity.py                         # 2 braços x 4 tarefas x 3 seeds
    python probe_assoc_capacity.py --arms replace --tasks induction --seeds 0
    python probe_assoc_capacity.py --steps 400 --out bundles/v21.4-harness/probe_assoc

Contribuição: Claude (proposta CEL-2.001).
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

from model_assoc_probe import ConsciousV21AssocProbe, selftest  # noqa: E402
from runner_v21 import set_seeds, train_run  # noqa: E402
from tasks_v21 import make_env  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")

V, L, B, LS = 16, 32, 8, 0.1
UNIFORME = math.log(V)


def load_references():
    """Referências já medidas no repo (single seed, mesmo microscópio). Opcionais."""
    ref = {}
    bdir = REPO / "bundles" / "v21.4-harness"
    p = bdir / "probe_qkv" / "probe_qkv.json"
    if p.exists():
        for task, v in json.loads(p.read_text())["verdict"].items():
            ref[task] = {"v21_fixo": v["fixo_val"], "v21_qkv": v["qkv_val"]}
    p = bdir / "p3_probes" / "probe_table.json"
    if p.exists():
        for k, v in json.loads(p.read_text())["tabela"].items():
            task, name = k.split("::")
            if name == "viva":
                ref.setdefault(task, {}).setdefault("v21_fixo", v["loss_val"])
    return ref


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["induction", "lag", "copy", "assoc"])
    ap.add_argument("--arms", nargs="*", default=["replace", "hybrid"], choices=["replace", "hybrid"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(REPO / "bundles" / "v21.4-harness" / "probe_assoc"))
    return ap.parse_args()


def main():
    a = parse()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("selftest (paralelo == scan, sem vazamento causal):", selftest(d_model=64), flush=True)
    refs = load_references()
    print(f"uniforme ln({V}) = {UNIFORME:.5f} | tarefas={a.tasks} braços={a.arms} seeds={a.seeds} "
          f"passos={a.steps}", flush=True)

    runs = {}
    for arm in a.arms:
        for task in a.tasks:
            for sd in a.seeds:
                t0 = time.time()
                set_seeds(sd)
                m = ConsciousV21AssocProbe(coupling=True, langevin=False, layer_scale_init=LS, mode=arm)
                env = make_env(task, V=V, L=L, B=B, seed=sd, n_val=4)
                r = train_run(m, env, a.steps, V=V, clip_norm=1.0, lr=a.lr, log=None,
                              grad_stats_every=0, snap_steps=(0, a.steps - 1), verbose_every=0,
                              label=f"{arm}/{task}/s{sd}")
                alpha = [blk.last_alpha for blk, _ in m.blocks] if arm == "hybrid" else None
                rec = {k: v for k, v in r.items() if k not in ("_model",)}
                rec["alpha_final_por_camada"] = alpha
                rec["wall_s"] = round(time.time() - t0, 1)
                runs[f"{arm}::{task}::s{sd}"] = rec
                print(f"[{arm:7s}/{task:9s}/s{sd}] train={r['loss_train']:.5f} val={r['loss_val']:.5f} "
                      f"nan={r['g_nan_total']} alpha={None if alpha is None else [round(x, 3) for x in alpha]} "
                      f"({rec['wall_s']}s)", flush=True)

    # ---- resumo por (braço, tarefa) -------------------------------------------------------
    summary = {}
    for arm in a.arms:
        for task in a.tasks:
            vals = [runs[f"{arm}::{task}::s{sd}"]["loss_val"] for sd in a.seeds]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            ref = refs.get(task, {})
            row = {"loss_val_seeds": [round(v, 5) for v in vals], "mean": round(mean, 5),
                   "std": round(std, 5), "aprendeu_C1": bool(mean < 0.95 * UNIFORME),
                   "ref_v21_fixo": ref.get("v21_fixo"), "ref_v21_qkv": ref.get("v21_qkv")}
            if arm == "hybrid":
                alphas = [runs[f"{arm}::{task}::s{sd}"]["alpha_final_por_camada"] for sd in a.seeds]
                am = float(np.mean([x for al in alphas for x in al]))
                row["alpha_medio"] = round(am, 4)
                row["gate_vivo_C4"] = bool(0.02 <= am <= 0.98)
                if task == "lag" and ref.get("v21_fixo") is not None:
                    row["sem_regressao_lag_C3"] = bool(mean <= ref["v21_fixo"] + 0.05)
            summary[f"{arm}::{task}"] = row

    verdict = {}
    if "replace" in a.arms and "induction" in a.tasks:
        verdict["C2_replace_aprende_induction"] = summary["replace::induction"]["aprendeu_C1"]
    (out / "probe_assoc.json").write_text(json.dumps(
        {"uniforme": UNIFORME, "args": vars(a), "criterios": "ver docstring (pré-registrados)",
         "veredito": verdict, "resumo": summary, "runs": runs}, indent=2, default=str))

    print(f"\n=== resumo (uniforme ln{V} = {UNIFORME:.5f}; limiar C1 = {0.95 * UNIFORME:.5f}) ===")
    print(f"{'braço::tarefa':22s} {'média±std':>16s}  C1   ref_fixo  ref_qkv  extras")
    for k, r in summary.items():
        extra = ""
        if "alpha_medio" in r:
            extra = f"alpha={r['alpha_medio']} C4={r['gate_vivo_C4']}"
            if "sem_regressao_lag_C3" in r:
                extra += f" C3={r['sem_regressao_lag_C3']}"
        print(f"{k:22s} {r['mean']:9.5f}±{r['std']:.5f}  {str(r['aprendeu_C1'])[0]}   "
              f"{str(r['ref_v21_fixo']):>8s}  {str(r['ref_v21_qkv']):>7s}  {extra}")
    print(f"\nveredito: {verdict}\nok -> {out}/probe_assoc.json")


if __name__ == "__main__":
    main()
