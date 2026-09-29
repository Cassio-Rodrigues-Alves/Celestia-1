"""v21.3 1-variavel: layer_scale init 0->0.1 (atencao aparece desde o passo 0)."""
import json, csv, pathlib
import numpy as np, tensorflow as tf
from model_v21 import ConsciousV21
ver = "v21.3-layerscale01"
cfg = json.loads(pathlib.Path(f"bundles/{ver}/config.json").read_text())
seeds = cfg["seeds"]
out = pathlib.Path(f"bundles/{ver}/metrics_3seeds.csv")
rows = []
for sd in seeds:
    tf.random.set_seed(sd); np.random.seed(sd)
    m = ConsciousV21(coupling=True, langevin=True)
    _ = m(tf.zeros((1, 8), tf.int32), training=False)  # build
    for a, _ in m.blocks:
        a.layer_scale.assign(0.1)  # A unica variavel da v21.3
    opt = tf.keras.optimizers.Adam(1e-4, clipnorm=cfg["clip_norm"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    B, L, STEPS, V = 2, 32, 200, 512
    xb = tf.random.uniform((B, L), 0, V, dtype=tf.int32)
    n_clip, nan_hit = 0, False
    for s in range(STEPS):
        with tf.GradientTape() as tape:
            logits = m(xb, training=True)
            loss = loss_fn(xb[:, 1:], logits[:, :-1, :V]) + sum(m.losses)
        if bool(tf.math.is_nan(loss)):
            nan_hit = True; break
        grads = tape.gradient(loss, m.trainable_variables)
        gn = float(tf.linalg.global_norm([g for g in grads if g is not None]))
        if gn > cfg["clip_norm"]: n_clip += 1
        opt.apply_gradients(zip(grads, m.trainable_variables))
    ls = float(m.blocks[0][0].layer_scale.numpy())
    tel = m.blocks[0][0].telemetry()
    rows.append([sd, float(loss), n_clip/STEPS, gn, ls, tel["nhu"], nan_hit])
    print(f"seed {sd}: loss={float(loss):.3f} clip={n_clip/STEPS:.2f} gn={gn:.3f} ls={ls:.3f} nan={nan_hit}", flush=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "loss", "clip_rate", "gn", "layer_scale", "nhu", "nan"]); w.writerows(rows)
print(f"ok -> {out}")
