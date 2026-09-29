"""v21.1 3-seeds — exigência das regras p/ promover."""
import json, csv, pathlib
import numpy as np, tensorflow as tf
from model_v21 import ConsciousV21
ver = "v21.1-coupled"
cfg = json.loads(pathlib.Path(f"bundles/{ver}/config.json").read_text())
seeds = cfg["seeds"]
out = pathlib.Path(f"bundles/{ver}/metrics_3seeds.csv")
rows = []
for sd in seeds:
    tf.random.set_seed(sd); np.random.seed(sd)
    m = ConsciousV21(coupling=True)
    opt = tf.keras.optimizers.Adam(1e-4, clipnorm=cfg["clip_norm"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    B, L, STEPS, V = 2, 32, 50, 512
    xb = tf.random.uniform((B, L), 0, V, dtype=tf.int32)
    n_clip = 0
    for s in range(STEPS):
        with tf.GradientTape() as tape:
            logits = m(xb, training=True)
            loss = loss_fn(xb[:, 1:], logits[:, :-1, :V]) + sum(m.losses)
        grads = tape.gradient(loss, m.trainable_variables)
        gn = float(tf.linalg.global_norm([g for g in grads if g is not None]))
        if gn > cfg["clip_norm"]: n_clip += 1
        opt.apply_gradients(zip(grads, m.trainable_variables))
    tel = m.blocks[0][0].telemetry()
    rows.append([sd, float(loss), n_clip/STEPS, gn, tel["nhu"], tel["nuh"]])
    print(f"seed {sd}: loss={float(loss):.3f} clip={n_clip/STEPS:.2f} gn={gn:.3f} nhu={tel['nhu']:.3f}", flush=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "loss", "clip_rate", "gn", "nhu", "nuh"]); w.writerows(rows)
print(f"ok -> {out}")
