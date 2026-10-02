"""run_lm_abc — protocolo A/B/C em português real (DR-ITEM-6.x, pré-registro).

Braços (1 variável por versão): A=fixo (v21.1-coupled), B=qkv (só a mistura troca),
C=FFN alargado iso-parâmetros SEM roteamento (controle de capacidade).
Langevin OFF nos 3 (isolar mistura). Mesmo LR/schedule/clip/batch/steps/seeds.
Alinhamento next-token padrão (alvo(p)=y[p]; SEM o deslocamento +2 do sintético).

Uso:
  python run_lm_abc.py --smoke                      # fiação + mini-treino sintético (CPU)
  python run_lm_abc.py --corpus IDS.npy --steps N   # oficial (T4): A/B/C x 3 seeds + piso
  python run_lm_abc.py --corpus IDS.npy --arms A B --steps N --seeds 0  (piloto)

Salvaguardas (lições dos runs anteriores, automáticas):
  - clip_rate final > 0.95 → linha marcada TRANSIENTE (veredito inválido).
  - repete A/seed[0] no fim → piso same-seed publicado ao lado de todo Δ.
  - asserts: ids < vocab (tokenizer errado explode aqui, não no T4 remoto).

Decisão: adota QKV sse (i) loss_val(B)<loss_val(A) com |Δ|>piso nas 3 seeds,
(ii) B bate C, (iii) zero NaN/Inf e sem regressão de clip_rate >2pp.
"""
import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--corpus", default=None, help="npy de ids (Drive no Colab)")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--val-cap", type=int, default=512)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-floor", action="store_true", help="pula repetição same-seed")
    ap.add_argument("--out", default="bundles/lm-abc")
    return ap.parse_args()


def build_arm(tag, ffn_mult=4, ls=0.1):
    from model_v21 import ConsciousV21
    from model_qkv_probe import ConsciousV21QKVProbe
    kw = dict(coupling=True, langevin=False, layer_scale_init=ls, ffn_mult=ffn_mult)
    if tag == "B":
        return ConsciousV21QKVProbe(**kw)
    if tag in ("A", "C"):
        return ConsciousV21(**kw)
    raise ValueError(tag)


def _dense(g):
    import tensorflow as tf
    return tf.convert_to_tensor(g) if g is not None else None


def train_windows(m, tr_x, tr_y, va_x, va_y, V, steps, lr, clip, batch, seed,
                  label="", verbose_every=200):
    """Loop next-token padrão. Retorna dict de métricas + flags de convergência."""
    import tensorflow as tf
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=clip)
    m(va_x[:1], training=False)
    n_clip = nan_tot = 0
    for s in range(steps):
        i = np.random.default_rng(seed * 7919 + s).integers(0, len(tr_x), size=batch)
        xb, yb = tf.constant(tr_x[i]), tf.constant(tr_y[i])
        with tf.GradientTape() as tape:
            logits = m(xb, training=True)
            loss = loss_fn(yb, logits[..., :V]) + sum(m.losses)
        grads = tape.gradient(loss, m.trainable_variables)
        dense = [_dense(g) for g in grads]
        if any(g is not None and bool(tf.reduce_any(~tf.math.is_finite(g))) for g in dense):
            nan_tot += 1
        gn = float(tf.linalg.global_norm([g for g in dense if g is not None]))
        if gn > clip:
            n_clip += 1
        opt.apply_gradients(zip(grads, m.trainable_variables))
        if verbose_every and (s % verbose_every == 0 or s == steps - 1):
            print(f"  [{label}] step {s}: loss={float(loss):.4f} gn={gn:.3f}", flush=True)
    lv_logits = m(va_x, training=False)
    per_tok = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=va_y, logits=lv_logits[..., :V]).numpy()
    lv = float(per_tok.mean())
    from lm_abc import slice_losses
    sl = slice_losses(per_tok, va_x, va_y)
    clip_rate = n_clip / max(steps, 1)
    return {"loss_train": float(loss), "loss_val": lv, "ppl": float(math.exp(min(lv, 20.0))),
            "clip_rate": clip_rate, "gn": gn, "nan_steps": nan_tot,
            "transiente": bool(clip_rate > 0.95), **sl}


def main():
    import tensorflow as tf
    from lm_abc import (calibrate_ffn, count_nonemb, prefix_match_frac, split_recipe,
                        split_windows)
    from runner_v21 import set_seeds
    a = parse()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    if a.smoke:  # fiação + mini-treino real (3 steps/braço, sintético, microscópio)
        import model_v21
        model_v21.LAYERS, model_v21.D_MODEL = 2, 64
        maid = {}
        for tag in ["A", "B", "C"]:
            set_seeds(0)
            m = build_arm(tag)
            m(tf.zeros((1, 16), tf.int32), training=False)
            maid[tag] = count_nonemb(m)
        mult, n_c = calibrate_ffn(lambda ffn_mult: build_arm("C", ffn_mult), maid["B"], seq_len=16)
        rng = np.random.default_rng(0)
        sx = rng.integers(0, 64, size=(32, 15))
        sy = rng.integers(0, 64, size=(32, 15))
        print(f"params A={maid['A']} B={maid['B']} C={n_c}(mult={mult}) "
              f"copy_frac={prefix_match_frac(sx, sy):.3f}")
        set_seeds(0)
        m = build_arm("A")
        r = train_windows(m, sx, sy, sx[:8], sy[:8], 64, 3, 1e-3, 1.0, 2, 0, label="smoke/A")
        print(f"mini-treino ok: loss {r['loss_train']:.4f}->val {r['loss_val']:.4f} "
              f"transiente={r['transiente']}")
        (out / "smoke.json").write_text(json.dumps(
            {"params": maid, "ffn_mult_C": mult, "N_C": n_c,
             "mini_train": {k: v for k, v in r.items() if k != "gn"}}, indent=2))
        print(f"ok -> {out}/smoke.json")
        return 0

    assert a.corpus, "oficial exige --corpus (npy de ids)"
    assert os.path.exists(a.corpus), a.corpus
    ids = np.load(a.corpus, mmap_mode="r")
    assert int(ids.max()) < a.vocab, f"ids até {ids.max()} >= vocab {a.vocab} (tokenizer errado?)"
    recipe = split_recipe(a.corpus, os.path.getsize(a.corpus), a.seq_len, 0.1, 0)
    print(f"corpus: {a.corpus} ({len(ids) / 1e6:.1f}M ids) split={recipe}", flush=True)

    set_seeds(0)
    m_b = build_arm("B", ls=a.ls)
    m_b(tf.zeros((1, a.seq_len), tf.int32), training=False)
    n_b = count_nonemb(m_b)
    ffn_c, n_c = calibrate_ffn(lambda ffn_mult: build_arm("C", ffn_mult, ls=a.ls), n_b, seq_len=a.seq_len)
    m_a = build_arm("A", ls=a.ls)
    m_a(tf.zeros((1, a.seq_len), tf.int32), training=False)
    n_a = count_nonemb(m_a)
    print(f"N_A={n_a} N_B={n_b} N_C={n_c} (ffn_mult_C={ffn_c})", flush=True)
    audit = {"N_A": n_a, "N_B": n_b, "N_C": n_c, "ffn_mult_C": ffn_c, "corpus": a.corpus,
             "split_recipe": recipe, "steps": a.steps, "lr": a.lr, "clip": a.clip,
             "batch": a.batch, "seeds": a.seeds, "seq_len": a.seq_len, "ls": a.ls}

    rows = []
    for tag in a.arms:
        fm = ffn_c if tag == "C" else 4
        for sd in a.seeds:
            t0 = time.time()
            set_seeds(sd)
            m = build_arm(tag, ffn_mult=fm, ls=a.ls)
            tr_x, tr_y, va_x, va_y = split_windows(ids, a.seq_len, seed=sd, val_cap=a.val_cap)
            r = train_windows(m, tr_x, tr_y, va_x, va_y, a.vocab, a.steps, a.lr,
                              a.clip, a.batch, sd, label=f"{tag}/s{sd}")
            rows.append([tag, sd, f"{r['loss_train']:.5f}", f"{r['loss_val']:.5f}",
                         f"{r['ppl']:.2f}", f"{r['clip_rate']:.4f}", f"{r['gn']:.4f}",
                         r["nan_steps"],
                         f"{r['loss_copia']:.5f}" if r["loss_copia"] else "NA",
                         f"{r['loss_geral']:.5f}" if r["loss_geral"] else "NA",
                         f"{r['frac_copia']:.3f}", "TRANSIENTE" if r["transiente"] else "ok",
                         f"{time.time() - t0:.0f}"])
            print(f"[{tag}/s{sd}] val={r['loss_val']:.5f} copia={r['loss_copia']} "
                  f"geral={r['loss_geral']} {'TRANSIENTE' if r['transiente'] else ''}", flush=True)
    if not a.no_floor and a.seeds:  # piso same-seed: repete A/seed[0]
        sd = a.seeds[0]
        set_seeds(sd)
        m = build_arm("A", ls=a.ls)
        tr_x, tr_y, va_x, va_y = split_windows(ids, a.seq_len, seed=sd, val_cap=a.val_cap)
        r = train_windows(m, tr_x, tr_y, va_x, va_y, a.vocab, a.steps, a.lr,
                          a.clip, a.batch, sd, label=f"A-floor/s{sd}", verbose_every=0)
        audit["piso_same_seed"] = {"val_repeat": round(r["loss_val"], 5)}
        print(f"[piso] A/s{sd} repetido: val={r['loss_val']:.5f}", flush=True)
    with open(out / "metrics_abc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "loss_train", "loss_val", "ppl", "clip_rate", "gn",
                    "nan_steps", "loss_copia", "loss_geral", "frac_copia", "status", "wall_s"])
        w.writerows(rows)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"ok -> {out}/metrics_abc.csv + audit.json (piso: {audit.get('piso_same_seed')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
