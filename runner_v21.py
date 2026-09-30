"""runner_v21 — loop de treino/avaliação com TODAS as métricas obrigatórias da regra 2
e a instrumentação de estabilidade de `stab_v21`.

Usado por:
  - `run_v21_4_harness.py`  (Colab T4, config congelada — números que VALEM)
  - `local_debug_v21_4.py`  (reduzido, CPU local — só debug, regra 1.3)

Saídas por passo:   loss, grad_norm, clip, g_nan, g_inf, n_zero_grad, delta_theta_ratio
Por camada:         mu_e, sigma_e2 (expoentes de gradiente), r_t^(l) (ganho residual)
Validação:          loss_val, perplexidade_val, overfitting_gap

Contribuição: DeepSeek (v21.4 — noite 2026-09-30).
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import tensorflow as tf

import stab_v21
from stab_v21 import StabilityLogger, clip_overflow_stats, delta_ratio, grad_exponent_stats, residual_gain, snapshot
from tasks_v21 import eval_loss


def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def train_run(model, env, steps, V, clip_norm, lr=1e-4,
              log=None, resid_every=0, grad_stats_every=0, snap_steps=(0, 1),
              verbose_every=50, label=""):
    """Treina `steps` passos na env. Retorna dict de métricas (regra 2 + instrumentação).

    log: StabilityLogger opcional (CSV por passo).
    resid_every / grad_stats_every: 0 = só no fim (barato).
    snap_steps: passos em que ||dtheta||/||theta|| é medido.
    """
    batch_fn, val_x, val_y, meta = env
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=clip_norm)

    # PITFALL (Keras 3): trainable_variables só existe COMPLETO depois do 1º forward.
    # Snapshot antes disso compara listas de tamanhos diferentes -> métrica-lixo.
    model(val_x, training=False)
    n_trainable = len(model.trainable_variables)

    n_clip = 0
    last_loss = float("nan")
    prev = snapshot(model)                     # para ||dtheta||/||theta||
    resid_start = stab_v21.residual_gain(model, val_x)
    per_step = []
    final_agg, final_var = {}, {}
    nan_total = inf_total = 0

    for s in range(steps):
        xb, yb = batch_fn(s)
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb[:, 1:], logits[:, :-1, :V]) + sum(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        st = clip_overflow_stats(model, grads, clip_norm)
        n_clip += st["clip"]
        nan_total += st["g_nan"]
        inf_total += st["g_inf"]
        last_loss = float(loss)

        if s in snap_steps:
            opt.apply_gradients(zip(grads, model.trainable_variables))
            cur = snapshot(model)
            dr = delta_ratio(prev, cur)
            prev = cur
        else:
            opt.apply_gradients(zip(grads, model.trainable_variables))
            dr = float("nan")

        if grad_stats_every and (s % grad_stats_every == 0):
            _, agg = grad_exponent_stats(model, grads)
            if log is not None:
                log.log_layer_stats(s, agg, prefix="mu")
        if log is not None:
            log.log_step(step=s, split=label or "train", loss=f"{last_loss:.6f}",
                         grad_norm=f"{st['grad_norm']:.6f}", clip=st["clip"],
                         g_nan=st["g_nan"], g_inf=st["g_inf"],
                         n_zero_grad=st["n_zero_grad"], delta_theta_ratio=f"{dr:.6e}")
        per_step.append({"step": s, "loss": last_loss, "grad_norm": st["grad_norm"],
                         "clip": st["clip"], "delta_theta_ratio": dr})
        if verbose_every and (s % verbose_every == 0 or s == steps - 1):
            print(f"  [{label}] step {s:4d} loss={last_loss:.5f} gn={st['grad_norm']:.4f} "
                  f"clip={st['clip']} dtheta={dr:.3e}", flush=True)

    resid_end = stab_v21.residual_gain(model, val_x)

    loss_train = last_loss
    loss_val = eval_loss(model, val_x, val_y, V, loss_fn)
    out = {
        "label": label, "steps": steps, "V": V,
        "n_trainable_vars": n_trainable,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "perplexidade_val": float(math.exp(min(loss_val, 20.0))),
        "overfitting_gap": loss_val - loss_train,
        "clip_rate": n_clip / max(steps, 1),
        "grad_norm_final": per_step[-1]["grad_norm"] if per_step else None,
        "g_nan_total": nan_total, "g_inf_total": inf_total,
        "delta_theta_ratio_first": per_step[0]["delta_theta_ratio"] if per_step else None,
        "residual_gain": {"start": resid_start, "end": resid_end},
        "task_meta": meta,
        "sat_heuristic": bool(loss_val < 1e-2),
    }
    if log is not None:
        log.log_residual(steps - 1, resid_end, split="val")
    return out


def context_test(model, V=64, L=32, seed=12345):
    """Regra 3.2 — layer_scale=1.0 vs 0.0 PRECISA mudar o output. Restaura o valor original."""
    xb = tf.random.uniform((2, L), 0, V, dtype=tf.int32)
    saved = [float(a.layer_scale.numpy()) for a, _ in model.blocks]
    try:
        for a, _ in model.blocks:
            a.layer_scale.assign(tf.constant(0.0, tf.float32))
        out0 = model(xb, training=False).numpy()
        for a, _ in model.blocks:
            a.layer_scale.assign(tf.constant(1.0, tf.float32))
        out1 = model(xb, training=False).numpy()
    finally:
        for (a, _), v in zip(model.blocks, saved):
            a.layer_scale.assign(tf.constant(v, tf.float32))
    rel = float(np.linalg.norm(out1 - out0) / (np.linalg.norm(out0) + 1e-8))
    return {"layer_scale_0_vs_1_rel_diff": rel, "usa_contexto": bool(rel > 1e-4)}
