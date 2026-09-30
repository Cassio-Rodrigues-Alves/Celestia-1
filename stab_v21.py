"""stab_v21 — instrumentação de estabilidade (regra 2 / inst.ger.1 prioridade 1).

Métricas implementadas (todas exigidas pela regra 2.1 + inst.ger.1):
  - r_t^(l):  ganho residual por camada, por token -> mean/max; separado attn vs ffn.
  - mu_e, sigma_e^2: estatística dos EXPOENTES dos gradientes (log2|g|) por variável/camada.
  - ||dtheta||_2 / ||theta||_2: deslocamento relativo dos pesos por passo.
  - integra com diag_v21 (geometria) e com o CSV por época.

Princípio: OBSERVAÇÃO PURA. Nada aqui altera o grafo, a loss ou os gradientes.
`residual_gain` reexecuta a recursão do forward e VERIFICA que os logits batem com
`model(xb)`; se divergir, levanta exceção — instrumentação que mente não vale.

Contribuição: DeepSeek (v21.4 — noite 2026-09-30).
"""
from __future__ import annotations

import csv
import numpy as np
import tensorflow as tf

EPS = 1e-8


def _f32(x):
    return tf.cast(x, tf.float32)


def _row_norm(t):
    """Norma L2 por token -> vetor [N]."""
    n = tf.shape(t)[-1]
    return tf.linalg.norm(tf.reshape(_f32(t), [-1, n]), axis=-1)


def _flat_l2(t):
    shape = tf.shape(t)
    return tf.linalg.norm(tf.reshape(_f32(t), [tf.reduce_prod(shape)]))


def residual_gain(model, xb, verify=True):
    """r_t^(l) = ||a(x) + f(x)||_2 / (||x_in||_2 + eps) por camada.

    Reexecuta a recursão de `ConsciousV21.call` capturando entrada/saída de cada bloco.
    Requer langevin=False no modelo (o loop de Langevin só roda em training=True e
    injeta ruído — comparação bit-a-bit seria impossível). Levanta se langevin.
    verify=True: confere logits vs model(xb) e levanta AssertionError se divergir.
    """
    for a, _ in model.blocks:
        if bool(getattr(a, "langevin", False)):
            raise ValueError("residual_gain: modelo com langevin=True não suportado (ruído)")

    x = model.embed(xb) + model.pos(tf.range(tf.shape(xb)[1]))
    rows = []
    for i, (a, f) in enumerate(model.blocks):
        x_in = x
        a_out = a(x, training=False)
        f_out = f(x)
        x = x_in + a_out + f_out
        den = _row_norm(x_in) + EPS
        r = _row_norm(a_out + f_out) / den
        r_attn = _row_norm(a_out) / den
        r_ffn = _row_norm(f_out) / den
        rows.append({
            "layer": i,
            "r_mean": float(tf.reduce_mean(r)),
            "r_max": float(tf.reduce_max(r)),
            "r_attn_mean": float(tf.reduce_mean(r_attn)),
            "r_ffn_mean": float(tf.reduce_mean(r_ffn)),
            "x_in_norm_mean": float(tf.reduce_mean(_row_norm(x_in))),
        })

    logits = model.head(model.ln(x))
    if verify:
        ref = model(xb, training=False)
        diff = float(np.max(np.abs(logits.numpy() - ref.numpy())))
        if not np.allclose(logits.numpy(), ref.numpy(), atol=1e-4, rtol=1e-3):
            raise AssertionError(
                f"residual_gain: forward instrumentado divergiu do modelo (maxdiff={diff:.3e})"
            )
    return rows


def _as_np(g):
    """Gradiente -> numpy. Embedding produz IndexedSlices (que não tem .numpy())."""
    if isinstance(g, tf.IndexedSlices):
        g = tf.convert_to_tensor(g)
    return g.numpy()


def grad_exponent_stats(model, grads):
    """mu_e = E[log2|g|], sigma_e^2 = Var[log2|g|].

    Retorna (por_variavel, por_camada). Gradiente None ou todo-zero é registrado com
    n=0 e mu_e=None — ausência de gradiente é informação (conectividade), não erro.
    """
    per_var, per_layer = {}, {}
    for v, g in zip(model.trainable_variables, grads):
        name = v.name
        if g is None:
            per_var[name] = {"mu_e": None, "sigma_e2": None, "n": 0, "zero": True}
            continue
        ga = np.abs(_as_np(g).astype(np.float64)).ravel()
        nz = ga[ga > 0]
        if nz.size == 0:
            per_var[name] = {"mu_e": None, "sigma_e2": None, "n": 0, "zero": True}
            continue
        le = np.log2(nz)
        per_var[name] = {"mu_e": float(le.mean()), "sigma_e2": float(le.var()), "n": int(le.size), "zero": False}
        # bucket por bloco: 'attn_3' / 'ffn_3' / 'embedding' / 'head' ...
        parts = name.split("/")
        bucket = parts[0] if len(parts) > 1 else name.split(":")[0]
        per_layer.setdefault(bucket, []).append(le)
    agg = {}
    for k, arrs in per_layer.items():
        allv = np.concatenate(arrs)
        agg[k] = {"mu_e": float(allv.mean()), "sigma_e2": float(allv.var()), "n": int(allv.size)}
    return per_var, agg


def snapshot(model):
    """Cópia numpy dos pesos traináveis (para ||dtheta||/||theta||)."""
    return [v.numpy().copy() for v in model.trainable_variables]


def delta_ratio(before, after):
    """||dtheta||_2 / ||theta||_2 (relativo, global)."""
    num = np.sqrt(sum(float(np.sum((a - b) ** 2)) for a, b in zip(after, before)))
    den = np.sqrt(sum(float(np.sum(b ** 2)) for b in before)) + EPS
    return float(num / den)


def clip_overflow_stats(model, grads, clip_norm):
    """clip_rate + overflow fp16 + NaN/Inf dos gradientes. Retorna dict do passo."""
    norms, gnorm = [], 0.0
    nan = inf = 0
    for g in grads:
        if g is None:
            continue
        ga = _as_np(g)
        nan += int(np.isnan(ga).sum())
        inf += int(np.isinf(ga).sum())
        n = float(np.linalg.norm(ga.ravel()))
        gnorm += n * n
        norms.append(n)
    gnorm = float(np.sqrt(gnorm))
    return {"grad_norm": gnorm, "clip": int(gnorm > clip_norm),
            "g_nan": nan, "g_inf": inf, "n_zero_grad": sum(1 for g in grads if g is None)}


class StabilityLogger:
    """Grava métricas por passo de instrumentação em CSV (append-safe)."""

    FIELDS_STEP = ["step", "split", "loss", "grad_norm", "clip", "g_nan", "g_inf",
                   "n_zero_grad", "delta_theta_ratio"]

    def __init__(self, path):
        self.path = str(path)
        self.rows = []

    def log_step(self, **kw):
        self.rows.append({k: kw.get(k, "") for k in self.FIELDS_STEP})

    def log_residual(self, step, rows, split="train"):
        for r in rows:
            self.rows.append({"step": step, "split": f"r_{split}_L{r['layer']}",
                              "loss": "", "grad_norm": f"{r['r_mean']:.6e}",
                              "clip": f"attn={r['r_attn_mean']:.6e}|ffn={r['r_ffn_mean']:.6e}",
                              "g_nan": "", "g_inf": f"{r['r_max']:.6e}",
                              "n_zero_grad": "", "delta_theta_ratio": ""})

    def log_layer_stats(self, step, agg, prefix="mu"):
        for layer, st in agg.items():
            self.rows.append({"step": step, "split": f"{prefix}_{layer}", "loss": "",
                              "grad_norm": f"{st['mu_e']:.6f}" if st["mu_e"] is not None else "",
                              "clip": f"{st['sigma_e2']:.6f}" if st["sigma_e2"] is not None else "",
                              "g_nan": "", "g_inf": "", "n_zero_grad": st["n"],
                              "delta_theta_ratio": ""})

    def flush(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS_STEP)
            w.writeheader()
            w.writerows(self.rows)
        return self.path


def summary_str(rows):
    """Linha compacta p/ console: r medio por camada + top-3."""
    if not rows:
        return ""
    rs = [r["r_mean"] for r in rows]
    return f"r_mean[min={min(rs):.4f} max={max(rs):.4f}] r_attn_L0={rows[0]['r_attn_mean']:.4f}"
