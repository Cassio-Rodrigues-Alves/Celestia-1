"""kvquant_v21 — protótipo de quantização de KV-cache com FIDELIDADE (prioridade 2 do inst.ger.1).

Motivação (inst.ger.1, 01/09): "INT8 parece segura; INT4 pode degradar fidelidade" — e a
regra 3.4 exige que compressão seja medida com `L_faith = 1 - |R_sustentado|/|R|`,
não só com acurácia. Só accuracy é insuficiente: uma resposta pode continuar "certa"
enquanto o contexto que a sustentava foi destruído.

O que este script mede (numpy puro, sem TF):
  - `acc`: a busca por chave (argmax q·k) acerta a chave certa?
  - `L_faith`: 1 - |respostas sustentadas pelo contexto| / |respostas|
  - `KL`: divergência da distribuição de atenção (softmax) quantizada vs FP16
  - `recuperou_contexto`: o valor lido (após softmax) está dentro de tolerância do valor real?

Isto é um PROTÓTIPO da medição, não um run de modelo real: mede se a métrica funciona e
qual é o comportamento esperado de FP16/INT8/INT4. Números locais = debug (regra 1.3).

Contribuição: DeepSeek (noite 2026-09-30).
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent / "bundles" / "v21.4-harness" / "kvquant"
OUT.mkdir(parents=True, exist_ok=True)

D = 64          # dimensão por cabeça
N_KEYS = 64     # "fatos" no contexto
N_Q = 32        # consultas
TRIALS = 20
TOL = 0.25      # tolerância p/ "resposta sustentada"


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def quant_uniform(x, bits):
    """Quantização uniforme simétrica (per-tensor). bits=16 -> float16 real."""
    if bits == 16:
        return x.astype(np.float16).astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max() / qmax
    if s == 0:
        return x.copy()
    q = np.clip(np.round(x / s), -qmax, qmax)
    return (q * s).astype(np.float32)


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def run_trial(rng, bits):
    # contexto com "fatos": cada chave carrega um valor
    keys = rng.normal(0, 1 / np.sqrt(D), size=(N_KEYS, D)).astype(np.float32)
    vals = rng.normal(0, 1 / np.sqrt(D), size=(N_KEYS, D)).astype(np.float32)
    q = rng.normal(0, 1 / np.sqrt(D), size=(N_Q, D)).astype(np.float32)

    # FP16 como referência
    K_ref, V_ref = quant_uniform(keys, 16), quant_uniform(vals, 16)
    S_ref = q @ K_ref.T / np.sqrt(D)
    P_ref = softmax(S_ref)
    OUT_ref = P_ref @ V_ref

    # consulta: qual chave é a "certa" para cada query? (maior produto com q)
    alvo = np.argmax(q @ keys.T, axis=1)

    K_q, V_q = quant_uniform(keys, bits), quant_uniform(vals, bits)
    S_q = q @ K_q.T / np.sqrt(D)
    P_q = softmax(S_q)
    OUT_q = P_q @ V_q

    acc = float(np.mean(np.argmax(S_q, axis=1) == alvo))
    kl_mean = float(np.mean([kl(P_ref[i], P_q[i]) for i in range(N_Q)]))
    sustentado = []
    for i in range(N_Q):
        # resposta sustentada = atenção concentrada na chave certa + valor lido próximo
        massa_certa = float(P_q[i, alvo[i]])
        erro_valor = float(np.linalg.norm(OUT_q[i] - vals[alvo[i]]) /
                           (np.linalg.norm(vals[alvo[i]]) + 1e-8))
        sustentado.append(massa_certa > 0.5 and erro_valor < TOL)
    l_faith = 1.0 - float(np.mean(sustentado))
    recuperou = float(np.mean([np.linalg.norm(OUT_q[i] - OUT_ref[i]) /
                               (np.linalg.norm(OUT_ref[i]) + 1e-8) < TOL for i in range(N_Q)]))
    return {"acc": acc, "l_faith": l_faith, "kl": kl_mean, "recuperou_contexto": recuperou}


def main():
    rng = np.random.default_rng(20260930)
    rows, resumo = [], {}
    for bits in (16, 8, 4):
        accs, faiths, kls, recs = [], [], [], []
        for _ in range(TRIALS):
            r = run_trial(rng, bits)
            accs.append(r["acc"]); faiths.append(r["l_faith"])
            kls.append(r["kl"]); recs.append(r["recuperou_contexto"])
        resumo[bits] = {
            "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "l_faith": float(np.mean(faiths)), "l_faith_std": float(np.std(faiths)),
            "kl": float(np.mean(kls)), "recuperou_contexto": float(np.mean(recs)),
        }
        rows.append([bits, f"{resumo[bits]['acc']:.5f}", f"{resumo[bits]['l_faith']:.5f}",
                     f"{resumo[bits]['kl']:.5e}", f"{resumo[bits]['recuperou_contexto']:.5f}"])
        print(f"bits={bits:2d} acc={resumo[bits]['acc']:.5f} L_faith={resumo[bits]['l_faith']:.5f} "
              f"KL={resumo[bits]['kl']:.3e} recuperou={resumo[bits]['recuperou_contexto']:.5f}", flush=True)

    with open(OUT / "kvquant.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bits", "acc", "L_faith", "KL_medio", "recuperou_contexto"])
        w.writerows(rows)

    # veredito explícito (regra 3.4): INT4 só passa se preservar L_faith
    v = {}
    ref = resumo[16]
    for bits in (8, 4):
        v[bits] = {
            "delta_acc_vs_fp16": round(resumo[bits]["acc"] - ref["acc"], 5),
            "delta_l_faith_vs_fp16": round(resumo[bits]["l_faith"] - ref["l_faith"], 5),
            "preserva_fidelidade": bool(resumo[bits]["l_faith"] <= ref["l_faith"] + 0.05),
        }
    (OUT / "kvquant.json").write_text(json.dumps({"resumo": resumo, "veredito": v,
                                                  "config": {"D": D, "N_KEYS": N_KEYS, "N_Q": N_Q,
                                                             "TRIALS": TRIALS, "TOL": TOL}}, indent=2))
    print("\nveredito (regra 3.4):", json.dumps(v, indent=2))
    print(f"ok -> {OUT}/kvquant.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
