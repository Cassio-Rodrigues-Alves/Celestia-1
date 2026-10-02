"""kvquant_v2 — métrica de fidelidade CORRIGIDA (regra 3.4) + controles de falsificação.

## O defeito da v1 (NOITE_2026-09-30.md §4)
Media "sustentação" contra o índice da chave do **gabarito**: uma resposta errada-mas-
confiante continuava contando como sustentada, e `L_faith` ficou **1.0 até em INT4**.
A acurácia pegava a degradação; a fidelidade não pegava nada.

## A v2
Uma resposta é **sustentada/aterrada** quando:
  1. a evidência é inequívoca: a maior massa de atenção `P_max >= TAU`; E
  2. a resposta é rastreável à entrada que o modelo de fato mais atencionou:
     `||A - V[used_key]|| / ||V[used_key]|| < TOL`.
Atenção difusa → leitura é mistura → não aterrada. Atenção apontando para outra entrada
→ resposta não corresponde → não aterrada. É o que a regra 3.4 quer proteger.

## Correção de rota registrada (por que a montagem mudou)
A 1ª execução deu `L_faith = 1.0` em TUDO, inclusive FP16. Causa: com
`keys, q ~ N(0, 1/√D)` e divisão por `√D`, os scores ficam ~N(0, 1/64) → a atenção é
**quase uniforme por construção** → toda leitura é uma média de 64 valores e nunca
equivale a uma entrada. A métrica estava certa em acusar; o EXPERIMENTO é que não tinha
evidência decisiva. Montagem corrigida: chaves e valores unitários (quase-ortogonais),
consulta = chave-alvo + ruído pequeno, e scores com GANHO alto → atenção decisiva
(~99% da massa na chave certa), que é o regime de um cache realmente consultável.

## Controles de falsificação (pré-registrados; sem eles a métrica não vale)
  (a) `answers_alheias`: respostas lidas de um cache INDEPENDENTE → têm de ser acusadas.
  (b) `massa_difusa`: atenção uniforme → resposta é mistura → têm de ser acusadas.

Escala local = debug (regra 1.3). Autor: DeepSeek (método: skills writing-plans +
verification-before-completion da biblioteca compartilhada ~/.agents/skills).
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent / "bundles" / "v21.4-harness" / "kvquant"
OUT.mkdir(parents=True, exist_ok=True)

D, N_KEYS, N_Q, TRIALS = 64, 64, 32, 20
TOL = 0.25      # tolerância de rastreabilidade da resposta a uma entrada do cache
TAU = 0.50      # massa mínima da maior evidência para a resposta ser considerada aterrada
GANHO = 10.0    # torna a atenção decisiva (equivalente a q·k com norma alta)
MODOS = ("normal", "answers_alheias", "massa_difusa")


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def quant_uniform(x, bits):
    if bits == 16:
        return x.astype(np.float16).astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max() / qmax
    if s == 0:
        return x.copy()
    return (np.clip(np.round(x / s), -qmax, qmax) * s).astype(np.float32)


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def l_faith_v2(answers, V, used_keys, pmax, tol=TOL, tau=TAU):
    """1 - |aterradas|/N. Aterrada = evidência inequívoca E resposta rastreável a ela."""
    det = {"evidencia_inequivoca": 0, "resposta_rastreavel": 0, "n": len(answers)}
    aterradas = 0
    vnorm = np.linalg.norm(V, axis=1) + 1e-8
    for i, a in enumerate(answers):
        dists = np.linalg.norm(V - a, axis=1) / vnorm
        k_star = int(np.argmin(dists))
        rastreavel = bool(dists[k_star] < tol and k_star == int(used_keys[i]))
        inequivoca = bool(pmax[i] >= tau)
        det["resposta_rastreavel"] += int(rastreavel)
        det["evidencia_inequivoca"] += int(inequivoca)
        aterradas += int(rastreavel and inequivoca)
    return 1.0 - aterradas / max(len(answers), 1), det


def _unit(x, rng):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x.astype(np.float32)


def run_trial_v2(rng, bits, modo="normal"):
    # contexto "consultável": chaves unitárias quase-ortogonais, valores unitários
    keys = _unit(rng.normal(size=(N_KEYS, D)), rng)
    vals = _unit(rng.normal(size=(N_KEYS, D)), rng)
    alvo = rng.integers(0, N_KEYS, size=N_Q)            # chave correta de cada consulta
    q = _unit(keys[alvo] + 0.05 * rng.normal(size=(N_Q, D)), rng)   # consulta ~ chave-alvo

    K_q, V_q = quant_uniform(keys, bits), quant_uniform(vals, bits)
    S_q = GANHO * (q @ K_q.T)
    P_q = softmax(S_q)
    used = np.argmax(S_q, axis=1)
    pmax = P_q.max(axis=1)
    answers = P_q @ V_q

    acc = float(np.mean(used == alvo))
    kl_ref = None
    if modo == "normal":
        K_r = quant_uniform(keys, 16)
        P_r = softmax(GANHO * (q @ K_r.T))
        kl_ref = float(np.mean([kl(P_r[i], P_q[i]) for i in range(N_Q)]))
    elif modo == "answers_alheias":
        k2 = _unit(rng.normal(size=(N_KEYS, D)), rng)
        v2 = _unit(rng.normal(size=(N_KEYS, D)), rng)
        P2 = softmax(GANHO * (q @ quant_uniform(k2, bits).T))
        answers = P2 @ quant_uniform(v2, bits)
    elif modo == "massa_difusa":
        P_u = np.full((N_Q, N_KEYS), 1.0 / N_KEYS)
        answers = P_u @ V_q
        used = np.argmax(P_u, axis=1)
        pmax = P_u.max(axis=1)

    lf, det = l_faith_v2(answers, V_q, used, pmax)
    return {"acc": acc, "l_faith": lf, "kl": kl_ref, "pmax_medio": float(np.mean(pmax)), "detalhes": det}


def main():
    rng = np.random.default_rng(20261002)
    resumo = {}
    for modo in MODOS:
        for bits in (16, 8, 4):
            accs, lfs, kls, pms = [], [], [], []
            for _ in range(TRIALS):
                r = run_trial_v2(rng, bits, modo)
                accs.append(r["acc"]); lfs.append(r["l_faith"]); pms.append(r["pmax_medio"])
                if r["kl"] is not None:
                    kls.append(r["kl"])
            resumo[f"{modo}|{bits}"] = {
                "acc": float(np.mean(accs)), "l_faith": float(np.mean(lfs)),
                "l_faith_std": float(np.std(lfs)), "pmax_medio": float(np.mean(pms)),
                "kl": float(np.mean(kls)) if kls else None,
            }
            print(f"{modo:16s} bits={bits:2d} acc={np.mean(accs):.5f} "
                  f"L_faith_v2={np.mean(lfs):.5f} pmax={np.mean(pms):.4f}", flush=True)

    crit = {
        "A1_fp16_lfaith_baixa": resumo["normal|16"]["l_faith"] <= 0.05,
        "A2a_answers_alheias_acusadas": resumo["answers_alheias|16"]["l_faith"] >= 0.90,
        "A2b_massa_difusa_acusada": resumo["massa_difusa|16"]["l_faith"] >= 0.90,
        "A3_int8_ok": (resumo["normal|8"]["l_faith"] <= 0.05 and resumo["normal|8"]["acc"] >= 0.95),
    }
    ok = all(crit.values())
    veredito = "METRICA VALIDADA" if ok else "METRICA REPROVADA NOS CONTROLES"

    out = {"config": {"D": D, "N_KEYS": N_KEYS, "N_Q": N_Q, "TRIALS": TRIALS,
                      "TOL": TOL, "TAU": TAU, "GANHO": GANHO},
           "resumo": resumo, "criterios": crit, "veredito": veredito,
           "comparacao_com_v1": {
               "v1_int4_l_faith": 1.0,
               "v1_problema": "media sustentacao contra o indice do gabarito -> nao acusava nada",
               "v2_int4_l_faith": resumo["normal|4"]["l_faith"],
               "v2_int4_acc": resumo["normal|4"]["acc"]},
           "nota": "escala debug; nao promove versao (regra 1.3)"}
    (OUT / "v2_corrigida.json").write_text(json.dumps(out, indent=2))

    with open(OUT / "v2_corrigida.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modo", "bits", "acc", "L_faith_v2", "l_faith_std", "pmax_medio", "KL"])
        for k, v in resumo.items():
            modo, bits = k.split("|")
            w.writerow([modo, bits, f"{v['acc']:.5f}", f"{v['l_faith']:.5f}", f"{v['l_faith_std']:.5f}",
                        f"{v['pmax_medio']:.5f}", "" if v["kl"] is None else f"{v['kl']:.3e}"])

    print("\n=== CRITÉRIOS PRÉ-REGISTRADOS ===")
    for k, v in crit.items():
        print(f"  {'OK ' if v else 'FALHOU'} {k}")
    print(f"VEREDITO: {veredito}")
    if resumo["normal|4"]["l_faith"] > 0.05:
        print(f"  ACHADO: INT4 degrada FIDELIDADE (L_faith={resumo['normal|4']['l_faith']:.5f}) "
              f"com acc={resumo['normal|4']['acc']:.5f} — exatamente o que a regra 3.4 quer pegar.")
    print(f"ok -> {OUT}/v2_corrigida.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
