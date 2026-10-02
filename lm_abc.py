"""lm_abc — utilidades do protocolo A/B/C em português real (DR-ITEM-6.x).

- count_nonemb: parâmetros não-embedding (embed/pos_embed fora, head dentro).
- calibrate_ffn: acha ffn_mult p/ o braço C igualar N do braço B (<1%).
- slice_copy_general:prefix_match_frac por janela + split cópia-vs-geral.
- split_windows: janelas train/val determinísticas de ids planos.

Contribuição: execução do TODO_deep-research-agent.md (DR-ITEM-6.x).
"""
from __future__ import annotations

import numpy as np


def count_nonemb(model):
    """Total de params treináveis fora de embed/pos_embed (padrão Kaplan)."""
    tot = 0
    for v in model.trainable_variables:
        if "/embed" in v.path or "/pos" in v.path:
            continue
        tot += int(v.shape.num_elements())
    return tot


def calibrate_ffn(build_fn, n_target, seq_len=32, tol=0.01):
    """Acha ffn_mult tal que N_C ≈ N_B. Dá forward dummy antes de contar
    (Keras 3 cria Dense preguiçosamente — sem forward a contagem sai lixo)."""
    import tensorflow as tf

    def n_of(mult):
        m = build_fn(ffn_mult=mult)
        m(tf.zeros((1, seq_len), tf.int32), training=False)
        return count_nonemb(m)

    lo, hi = 4.0, 12.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if n_of(mid) < n_target:
            lo = mid
        else:
            hi = mid
    mult = round((lo + hi) / 2, 3)
    m = build_fn(ffn_mult=mult)
    m(tf.zeros((1, seq_len), tf.int32), training=False)
    n = count_nonemb(m)
    assert abs(n - n_target) / n_target < tol, f"C fora de 1%: {n} vs {n_target}"
    return mult, n


def split_windows(ids, seq_len, val_frac=0.1, seed=0):
    """Janelas (x=seq[:-1], y=seq[1:]) de ids planos. Val = primeiros 10%
    (determinístico dado o arquivo; sem fronteira de doc — limitação registrada:
    o TODO pede split por documento com hash, o que exige corpus com docs)."""
    ids = np.asarray(ids, dtype=np.int64)
    n_win = len(ids) // (seq_len + 1)
    ids = ids[:n_win * (seq_len + 1)].reshape(n_win, seq_len + 1)
    n_val = max(1, int(n_win * val_frac))
    val = ids[:n_val]
    train = ids[n_val:]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    return (train[idx][:, :-1], train[idx][:, 1:], val[:, :-1], val[:, 1:])


def prefix_match_frac(val_x, val_y):
    """Fração de alvos que já ocorreram no contexto (por janela). Elhage/Olsson."""
    x = np.asarray(val_x)
    frac = []
    for i in range(len(x)):
        seen = set(x[i, :-1].tolist())
        frac.append(np.mean([t in seen for t in np.asarray(val_y)[i].tolist()]))
    return np.asarray(frac)


def slice_losses(model, val_x, val_y, V, loss_fn):
    """Loss média nas fatias cópia (alvo visto no contexto) vs geral. TF."""
    import tensorflow as tf
    logits = model(val_x, training=False)
    per_tok = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=val_y[:, 1:], logits=logits[:, :-1, :V])
    seen = np.zeros_like(np.asarray(val_y)[:, 1:], dtype=bool)
    xa = np.asarray(val_x)
    for i in range(len(xa)):
        s = set(xa[i, :-1].tolist())
        seen[i] = [t in s for t in np.asarray(val_y)[i, 1:]]
    per_tok = per_tok.numpy()
    m_copia, m_geral = seen, ~seen
    return {"loss_copia": float(per_tok[m_copia].mean()) if m_copia.any() else None,
            "loss_geral": float(per_tok[m_geral].mean()) if m_geral.any() else None,
            "frac_copia": float(m_copia.mean())}
