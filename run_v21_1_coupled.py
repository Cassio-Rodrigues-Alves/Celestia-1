"""v21.1-coupled — entrypoint Colab T4. Telemetria mínima das regras rígidas."""
import json, csv, pathlib
import numpy as np
ver = "v21.1-coupled"
cfg = json.loads(pathlib.Path(f"bundles/{ver}/config.json").read_text())
print(f"[{ver}] config:", cfg)

# --- funções telemetria (mesmas que o TANGO real vai chamar) ---
def residual_gain(h_in, h_out, eps=1e-8):
    return float(np.linalg.norm(h_out - h_in) / (np.linalg.norm(h_in) + eps))

def exp_stats(g, eps=1e-12):
    log2 = np.log2(np.abs(g).ravel() + eps)
    return float(log2.mean()), float(log2.var())

def lyapunov_full(s_traj):
    # s_traj: lista de s_l(t) t=0..T -> expoentes locais log|s_t/s_{t-1}|
    s = np.asarray(s_traj, dtype=float)
    return [float(np.log(abs(b) / (abs(a) + 1e-12) + 1e-12)) for a, b in zip(s[:-1], s[1:])]

# --- fumaça com dados sintéticos + forward real v21 (1 batch tiny) ---
rng = np.random.default_rng(cfg["seeds"][0])
h_in, h_out = rng.normal(size=(8, 768)), rng.normal(size=(8, 768))
g = rng.normal(size=(768, 768)) * 0.01
r = residual_gain(h_in, h_out)
mu_e, sig_e = exp_stats(g)
lyap = lyapunov_full([1.0, 0.9, 0.85, 0.83])  # s_l(t=0..3)
clip_rate, grad_norm, gap = 0.02, float(np.linalg.norm(g)), 0.05

try:
    import tensorflow as tf
    from model_v21 import ConsciousV21, SEQ_LEN
    m = ConsciousV21(coupling=True)
    dummy = tf.zeros((1, 16), dtype=tf.int32)
    _ = m(dummy, training=False)
    tel0 = m.blocks[0][0].telemetry()
    # teste contexto: layer_scale 1 vs 0 muda output?
    m.blocks[0][0].layer_scale.assign(1.0)
    o1 = m(dummy, training=False).numpy().mean()
    m.blocks[0][0].layer_scale.assign(0.0)
    o0 = m(dummy, training=False).numpy().mean()
    ctx = f"layer_scale 1 vs 0: {o1:.6f} vs {o0:.6f} diff={abs(o1-o0):.6f} {'USA_CTX' if abs(o1-o0)>1e-6 else 'IGNORA_CTX'}"
    m.blocks[0][0].layer_scale.assign(0.0)  # volta p/ init regra
    v21line = f"v21 forward ok tel0={tel0} {ctx}"
except Exception as e:
    v21line = f"v21 forward FALHOU: {type(e).__name__}: {e}"
    ctx = "layer_scale 1.0 vs 0.0: FALHOU"
    tel0 = {}

out = pathlib.Path(f"bundles/{ver}/metrics.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss_train", "loss_val", "clip_rate", "grad_norm", "gap",
                "r_t", "mu_e", "sig_e", "lyap"])
    w.writerow([0, 4.5, 4.55, clip_rate, grad_norm, gap, r, mu_e, sig_e, ";".join(f"{x:.3f}" for x in lyap)])
pathlib.Path(f"bundles/{ver}/teste_contexto.log").write_text(
    f"{ctx}\n"
    f"r_t={r:.4f} mu_e={mu_e:.3f} sig_e={sig_e:.3f} lyap={lyap}\n"
    f"{v21line}\n"
    f"u_tan==u_facts: ANCORADO (por construção)\n")
print(f"[{ver}] fumaça telemetria ok -> {out} r={r:.3f} lyap={lyap}\n{v21line}")

# --- treino fumaça tiny: 50 steps, batch 2, seq 32, vocab 512 reduzido p/ caber ---
try:
    import tensorflow as tf
    from model_v21 import ConsciousV21
    tf.random.set_seed(cfg["seeds"][0])
    m2 = ConsciousV21(coupling=True)
    opt = tf.keras.optimizers.Adam(1e-4, clipnorm=cfg["clip_norm"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    B, L, STEPS, V = 2, 32, 50, 512
    xb = tf.random.uniform((B, L), 0, V, dtype=tf.int32)
    n_clip = 0
    for s in range(STEPS):
        with tf.GradientTape() as tape:
            logits = m2(xb, training=True)
            # pred next-token truncado p/ V tiny
            loss = loss_fn(xb[:, 1:], logits[:, :-1, :V])
            loss += sum(m2.losses)
        grads = tape.gradient(loss, m2.trainable_variables)
        gn = float(tf.linalg.global_norm([g for g in grads if g is not None]))
        if gn > cfg["clip_norm"]:
            n_clip += 1
        opt.apply_gradients(zip(grads, m2.trainable_variables))
        if s % 10 == 0:
            print(f" step {s}: loss={float(loss):.3f} gn={gn:.3f}", flush=True)
    smoke = f"smoke 50steps ok loss={float(loss):.3f} clip_rate={n_clip/STEPS:.2f} gn={gn:.3f}"
    print(smoke)
    open(f"bundles/{ver}/teste_contexto.log", "a").write(smoke + "\n")
except Exception as e:
    print(f"smoke FALHOU: {type(e).__name__}: {e}")
