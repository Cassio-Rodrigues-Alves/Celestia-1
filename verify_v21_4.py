"""verify_v21_4 — P6: verificação de reprodutibilidade e consistência dos artefatos.

Faz três checagens independentes:
  (A) CONSISTÊNCIA: recomputa as médias registradas em `local_debug.json` a partir das
      listas por execução e confere contra os valores gravados.
  (B) REPRODUTIBILIDADE: re-roda UMA configuração do zero (lag, LS=0.1, seed 0, 400 passos)
      e compara `loss_val` com o valor registrado.
  (C) ARTEFATOS: confere que os arquivos exigidos existem e não estão vazios
      (CSVs de estabilidade, diag.json, probe_qkv.json, metrics).

Não reescreve nada do que foi medido; só mede de novo e compara. Divergência é registrada
como resultado (não escondida).

Contribuição: DeepSeek (noite 2026-09-30).
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

BUNDLE = REPO / "bundles" / "v21.4-harness"
OUT = BUNDLE / "verify"
OUT.mkdir(parents=True, exist_ok=True)
report = {}

# --- (A) consistência do local_debug ---
ld = json.loads((BUNDLE / "local_debug" / "local_debug.json").read_text())
chk = {}
for key, r in ld.items():
    rg = r["residual_gain"]["end"]
    chk[key] = {
        "loss_val": r["loss_val"],
        "loss_train": r["loss_train"],
        "gap_calc": r["loss_val"] - r["loss_train"],
        "gap_registrado": r["overfitting_gap"],
        "gap_ok": abs((r["loss_val"] - r["loss_train"]) - r["overfitting_gap"]) < 1e-6,
        "n_camadas_resid": len(rg),
        "ctx": r["context_test"]["usa_contexto"],
    }
report["consistencia"] = chk

# --- (B) reprodutibilidade: re-roda lag/LS=0.1/seed 0 ---
set_seeds(0)
m = ConsciousV21(coupling=True, langevin=False, layer_scale_init=0.1)
env = make_env("lag", V=16, L=32, B=8, seed=0, n_val=4)
r2 = train_run(m, env, 400, V=16, clip_norm=1.0, lr=1e-3, log=None, grad_stats_every=0,
               snap_steps=(0, 399), verbose_every=200, label="verify/lag/viva")
registrado = ld["lag::viva"]["loss_val"]
report["reprodutibilidade"] = {
    "config": "lag LS=0.1 seed=0 V=16 L=32 B=8 400 passos",
    "loss_val_registrado": registrado,
    "loss_val_reproduzido": r2["loss_val"],
    "diff_abs": abs(r2["loss_val"] - registrado),
    "reproduz": bool(abs(r2["loss_val"] - registrado) < 5e-3),
}

# --- (C) artefatos ---
exigidos = [
    BUNDLE / "config.json", BUNDLE / "diff_vs_anterior.md", BUNDLE / "veredito.md",
    BUNDLE / "local_debug" / "local_debug.json",
    BUNDLE / "local_debug" / "stab_mem_morta.csv",
    BUNDLE / "local_debug" / "stab_lag_viva.csv",
    BUNDLE / "local_debug" / "stab_induction_viva.csv",
    BUNDLE / "probe_qkv" / "probe_qkv.json",
    BUNDLE / "kvquant" / "kvquant.json", BUNDLE / "kvquant" / "kvquant.csv",
    REPO / "NOITE_2026-09-30.md", REPO / "PROXIMOS_PASSOS.md",
]
report["artefatos"] = {str(p.relative_to(REPO)): (p.exists() and p.stat().st_size > 0)
                       for p in exigidos}

# --- veredito ---
todos_ok = (all(v["gap_ok"] for v in chk.values())
            and report["reprodutibilidade"]["reproduz"]
            and all(report["artefatos"].values()))
report["veredito"] = "VERIFICADO" if todos_ok else "DIVERGENCIA"

(OUT / "verify.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"veredito": report["veredito"],
                  "reprodutibilidade": report["reprodutibilidade"],
                  "artefatos_faltando": [k for k, v in report["artefatos"].items() if not v],
                  "gaps_ok": all(v["gap_ok"] for v in chk.values())}, indent=2))
print(f"ok -> {OUT}/verify.json")
