"""v21.0-baseline — entrypoint Colab T4. Telemetria mínima das regras rígidas."""
import json, csv, pathlib
import numpy as np
ver = "v21.0-baseline"
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

# --- fumaça com dados sintéticos (valida pipeline sem TF pesado) ---
rng = np.random.default_rng(cfg["seeds"][0])
h_in, h_out = rng.normal(size=(8, 768)), rng.normal(size=(8, 768))
g = rng.normal(size=(768, 768)) * 0.01
r = residual_gain(h_in, h_out)
mu_e, sig_e = exp_stats(g)
lyap = lyapunov_full([1.0, 0.9, 0.85, 0.83])  # s_l(t=0..3)
clip_rate, grad_norm, gap = 0.02, float(np.linalg.norm(g)), 0.05

out = pathlib.Path(f"bundles/{ver}/metrics.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss_train", "loss_val", "clip_rate", "grad_norm", "gap",
                "r_t", "mu_e", "sig_e", "lyap"])
    w.writerow([0, 4.5, 4.55, clip_rate, grad_norm, gap, r, mu_e, sig_e, ";".join(f"{x:.3f}" for x in lyap)])
pathlib.Path(f"bundles/{ver}/teste_contexto.log").write_text(
    f"layer_scale 1.0 vs 0.0: PENDENTE (plug ConsciousModel aqui)\n"
    f"r_t={r:.4f} mu_e={mu_e:.3f} sig_e={sig_e:.3f} lyap={lyap}\n"
    f"u_tan==u_facts: PENDENTE\n")
print(f"[{ver}] fumaça telemetria ok -> {out} r={r:.3f} lyap={lyap}")
