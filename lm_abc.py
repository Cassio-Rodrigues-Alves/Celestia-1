"""lm_abc — utilidades do protocolo A/B/C em português real (DR-ITEM-6.x).

Convenção de alinhamento (padrão next-token LM, SEM o deslocamento +2 do
harness sintético): janela seq (len L+1) -> x=seq[:-1], y=seq[1:];
alvo(p) = y[p] com contexto x[0..p]. Cópia(p) = y[p] visto em x[0..p].

Contribuição: execução do TODO_deep-research-agent.md (DR-ITEM-6.x).
"""
from __future__ import annotations

import hashlib

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


def split_recipe(corpus_path, size, seq_len, val_frac, seed):
    """Receita do split documentada (TODO exige hash; ids planos não têm docs —
    limitação registrada: val = primeiros 10%, determinístico dado o arquivo)."""
    h = hashlib.sha256()
    h.update(f"{corpus_path}|{size}|{seq_len}|{val_frac}|{seed}".encode())
    with open(corpus_path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def split_windows(ids, seq_len, val_frac=0.1, seed=0, val_cap=512):
    """Janelas next-token de ids planos. Val = início do arquivo, CAPADO
    (val_cap janelas — val gigante explode o custo de eval)."""
    ids = np.asarray(ids, dtype=np.int64)
    n_win = len(ids) // (seq_len + 1)
    ids = ids[:n_win * (seq_len + 1)].reshape(n_win, seq_len + 1)
    n_val = max(1, min(int(n_win * val_frac), val_cap))
    val = ids[:n_val]
    train = ids[n_val:]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    train = train[idx]
    return (train[:, :-1], train[:, 1:], val[:, :-1], val[:, 1:])


def prefix_match_frac(val_x, val_y):
    """Fração de alvos vistos no contexto ESTRITO (x[0..p-1] para o alvo y[p]).
    Definição exigente de propósito: eco da posição atual não conta como cópia."""
    xa, ya = np.asarray(val_x), np.asarray(val_y)
    tot = hit = 0
    for i in range(len(xa)):
        seen = set()
        for p in range(xa.shape[1]):
            hit += ya[i, p] in seen
            tot += 1
            seen.add(int(xa[i, p]))
    return hit / max(tot, 1)


def slice_losses(per_tok, val_x, val_y):
    """Divide loss por token em cópia (alvo visto no contexto até p) vs geral."""
    per_tok = np.asarray(per_tok)
    xa, ya = np.asarray(val_x), np.asarray(val_y)
    mc = np.zeros_like(per_tok, dtype=bool)
    for i in range(len(xa)):
        seen = set()
        for p in range(xa.shape[1]):
            mc[i, p] = ya[i, p] in seen
            seen.add(int(xa[i, p]))
    c, g = per_tok[mc], per_tok[~mc]
    return {"loss_copia": float(c.mean()) if c.size else None,
            "loss_geral": float(g.mean()) if g.size else None,
            "frac_copia": float(mc.mean())}
