"""diag_v21 — telemetria rica p/ diagnóstico (regra: 5 casas + geometria + conectividade)."""
import json, csv, numpy as np, tensorflow as tf

def matrix_metrics(w, dim=None):
    w64 = np.asarray(w.numpy() if hasattr(w, "numpy") else w, dtype=np.float64)
    d = {"nan": bool(np.isnan(w64).any()), "inf": bool(np.isinf(w64).any()), "fro": float(np.linalg.norm(w64))}
    if w64.ndim == 2:
        s = np.linalg.svd(w64, compute_uv=False)
        d["sig_max"], d["sig_min"] = float(s.max()), float(s.min())
        if dim is not None and w64.shape == (dim, dim):
            d["ortho_fro"] = float(np.linalg.norm(w64.T @ w64 - np.eye(dim)))
        if w64.shape[0] == w64.shape[1]:
            d["rho"] = float(np.max(np.abs(np.linalg.eigvals(w64))))
    return d

def grad_connectivity(m, V=512):
    """Célula estilo CEL-LAB H3/H4: LM loss conecta w_uu/w_hu/w_uh? (camada 0)"""
    xb = tf.random.uniform((2, 32), 0, V, dtype=tf.int32)
    with tf.GradientTape(persistent=True) as tape:
        logits = m(xb, training=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=xb[:, 1:], logits=logits[:, :-1, :V])) + sum(m.losses)
    out = {}
    a0 = m.blocks[0][0]
    for nm, w in [("w_uu", a0.w_uu), ("w_hu", a0.w_hu), ("w_uh", a0.w_uh), ("stone", a0.w_stone)]:
        g = tape.gradient(loss, w)
        out[nm] = 0.0 if g is None else float(tf.linalg.global_norm([g]).numpy())
    del tape
    return out

def diag_model(m, ver, out_dir, extra=None):
    rows, g_nan, g_inf = [], False, False
    for i, (a, _) in enumerate(m.blocks):
        st = matrix_metrics(a.w_stone, dim=a.d_logic)
        uu = matrix_metrics(a.w_uu, dim=a.d_facts)
        hu, uh = matrix_metrics(a.w_hu), matrix_metrics(a.w_uh)
        g_nan |= st["nan"] or uu["nan"] or hu["nan"] or uh["nan"]
        g_inf |= st["inf"] or uu["inf"] or hu["inf"] or uh["inf"]
        rows.append({"layer": i,
                     "stone_fro": f"{st['fro']:.5f}", "stone_ortho": f"{st.get('ortho_fro', float('nan')):.5f}",
                     "stone_rho": f"{st.get('rho', float('nan')):.5f}",
                     "hu_fro": f"{hu['fro']:.5f}", "hu_sig_max": f"{hu.get('sig_max', float('nan')):.5f}",
                     "uh_fro": f"{uh['fro']:.5f}", "uh_sig_max": f"{uh.get('sig_max', float('nan')):.5f}",
                     "uu_ortho": f"{uu.get('ortho_fro', float('nan')):.5f}",
                     "layer_scale": f"{float(a.layer_scale.numpy()):.5f}",
                     "beta": f"{float(a.beta.numpy()):.5f}",
                     "gate_mean": f"{float(tf.reduce_mean(a.gate).numpy()):.5f}"})
    conn = grad_connectivity(m)
    out = {"ver": ver, "global_nan": g_nan, "global_inf": g_inf,
           "layer_scale_final": [float(a.layer_scale.numpy()) for a, _ in m.blocks],
           "grad_conn_L0": {k: f"{v:.5e}" for k, v in conn.items()}}
    if extra: out.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diag_geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out_dir / "diag.json").write_text(json.dumps(out, indent=2))
    print(f"[diag {ver}] nan={g_nan} inf={g_inf} layer_scale_L0={out['layer_scale_final'][0]:.5f} grad_conn={out['grad_conn_L0']}")
    return out
