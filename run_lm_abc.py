"""run_lm_abc — protocolo A/B/C em português real (DR-ITEM-6.x, pré-registro).

Braços (1 variável por versão): A=fixo (v21.1-coupled), B=qkv (só a mistura troca),
C=FFN alargado iso-parâmetros SEM roteamento (controle de capacidade).
Langevin OFF nos 3 (isolar mistura). Mesmo LR/schedule/clip/batch/steps/seeds.

Uso:
  python run_lm_abc.py --smoke                      # pipeline sem corpus (CPU, ~min)
  python run_lm_abc.py --corpus IDS.npy --steps N   # oficial (T4): A/B/C x 3 seeds
  python run_lm_abc.py --corpus IDS.npy --arms A B --steps N --seeds 0  (piloto)

Decisão (verbatim do critério): adota QKV sse (i) loss_val(B)<loss_val(A) com
|Δ|>piso nas 3 seeds, (ii) B bate C, (iii) zero NaN/Inf e sem regressão de
clip_rate >2pp. B≈A ou B≈C pós-convergência ⇒ arquiva com motivo (informação).
"""
import argparse
import csv
import json
import math
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
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="bundles/lm-abc")
    return ap.parse_args()


def build_arm(tag, ffn_mult=4):
    import model_v21
    from model_v21 import ConsciousV21
    from model_qkv_probe import ConsciousV21QKVProbe
    kw = dict(coupling=True, langevin=False, layer_scale_init=0.1)
    if tag == "A":
        return ConsciousV21(ffn_mult=ffn_mult, **kw)
    if tag == "B":
        return ConsciousV21QKVProbe(ffn_mult=ffn_mult, **kw)
    if tag == "C":
        return ConsciousV21(ffn_mult=ffn_mult, **kw)
    raise ValueError(tag)


def main():
    import tensorflow as tf
    from lm_abc import calibrate_ffn, count_nonemb, prefix_match_frac, slice_losses, split_windows
    from runner_v21 import set_seeds
    a = parse()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    if a.smoke:  # valida fiação sem corpus: build A/B/C + auditoria + 1 forward
        import model_v21
        model_v21.LAYERS, model_v21.D_MODEL = 2, 64
        maid = {}
        for tag in ["A", "B", "C"]:
            set_seeds(0)
            m = build_arm(tag)
            m(tf.zeros((1, 16), tf.int32), training=False)
            maid[tag] = count_nonemb(m)
        print(f"params A={maid['A']} B={maid['B']} C(base)={maid['C']}")
        mult, n_c = calibrate_ffn(lambda ffn_mult: build_arm("C", ffn_mult), maid["B"], seq_len=16)
        print(f"C calibrado: ffn_mult={mult} N_C={n_c} (|d|<1% de N_B={maid['B']})")
        xx = np.random.default_rng(0).integers(0, 64, size=(8, 32))
        print(f"prefix_match_frac demo: {prefix_match_frac(xx, xx).mean():.3f} (esperado ~1.0)")
        (out / "smoke.json").write_text(json.dumps(
            {"params": maid, "ffn_mult_C": mult, "N_C": n_c}, indent=2))
        print(f"ok -> {out}/smoke.json (fiação válida; ciência exige --corpus no T4)")
        return 0

    assert a.corpus, "oficial exige --corpus (npy de ids)"
    ids = np.load(a.corpus, mmap_mode="r")
    print(f"corpus: {a.corpus} ({len(ids) / 1e6:.1f}M ids)", flush=True)

    # auditoria de capacidade (full scale, 1 build por braço + forward p/ materializar)
    set_seeds(0)
    m_b = build_arm("B")
    m_b(tf.zeros((1, a.seq_len), tf.int32), training=False)
    n_b = count_nonemb(m_b)
    ffn_c, n_c = calibrate_ffn(lambda f: build_arm("C", f), n_b, seq_len=a.seq_len)
    m_a = build_arm("A")
    m_a(tf.zeros((1, a.seq_len), tf.int32), training=False)
    n_a = count_nonemb(m_a)
    print(f"N_A={n_a} N_B={n_b} N_C={n_c} (ffn_mult_C={ffn_c})", flush=True)
    audit = {"N_A": n_a, "N_B": n_b, "N_C": n_c, "ffn_mult_C": ffn_c,
             "corpus": a.corpus, "steps": a.steps, "lr": a.lr, "seeds": a.seeds}

    import tensorflow as tf  # noqa
    from runner_v21 import train_run as _  # noqa (usa loop próprio p/ stream de corpus)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    rows = []
    for tag in a.arms:
        fm = ffn_c if tag == "C" else 4
        for sd in a.seeds:
            t0 = time.time()
            set_seeds(sd)
            if tag == "B":
                from model_qkv_probe import ConsciousV21QKVProbe as Cls
            else:
                from model_v21 import ConsciousV21 as Cls
            m = Cls(coupling=True, langevin=False, layer_scale_init=a.ls, ffn_mult=fm)
            tr_x, tr_y, va_x, va_y = split_windows(ids, a.seq_len, seed=sd)
            opt = tf.keras.optimizers.Adam(a.lr, clipnorm=a.clip)
            m(va_x[:1], training=False)
            n_clip = 0
            for s in range(a.steps):
                i = np.random.default_rng(sd * 7919 + s).integers(0, len(tr_x), size=a.batch)
                xb, yb = tf.constant(tr_x[i]), tf.constant(tr_y[i])
                with tf.GradientTape() as tape:
                    logits = m(xb, training=True)
                    loss = loss_fn(yb[:, 1:], logits[:, :-1, :a.vocab]) + sum(m.losses)
                grads = tape.gradient(loss, m.trainable_variables)
                gn = float(tf.linalg.global_norm([g for g in grads if g is not None]))
                if gn > a.clip:
                    n_clip += 1
                opt.apply_gradients(zip(grads, m.trainable_variables))
                if s % 200 == 0 or s == a.steps - 1:
                    print(f"  [{tag}/s{sd}] step {s}: loss={float(loss):.4f} gn={gn:.3f}", flush=True)
            lv = float(loss_fn(va_y[:, 1:], m(va_x, training=False)[:, :-1, :a.vocab]))
            sl = slice_losses(m, va_x, va_y, a.vocab, loss_fn)
            rows.append([tag, sd, f"{float(loss):.5f}", f"{lv:.5f}", f"{math.exp(lv):.2f}",
                         f"{n_clip / a.steps:.4f}", f"{gn:.4f}",
                         f"{sl['loss_copia']:.5f}" if sl["loss_copia"] else "NA",
                         f"{sl['loss_geral']:.5f}" if sl["loss_geral"] else "NA",
                         f"{sl['frac_copia']:.3f}", f"{time.time() - t0:.0f}"])
            print(f"[{tag}/s{sd}] loss_val={lv:.5f} copia={sl['loss_copia']} geral={sl['loss_geral']}", flush=True)
    with open(out / "metrics_abc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "loss_train", "loss_val", "ppl", "clip_rate", "gn",
                    "loss_copia", "loss_geral", "frac_copia", "wall_s"])
        w.writerows(rows)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"ok -> {out}/metrics_abc.csv (decisão pelo critério pré-registrado, com piso same-seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
