"""kvquant_scan_gain — varredura do ganho: existe regime "acerta mas sem fidelidade"?

Com a métrica v2 já validada (kvquant_v2), a pergunta que sobra é a do artigo citado no
inst.ger.1: "INT4 degrada a fidelidade MESMO em respostas que continuam corretas".

A curiosidade anterior mostrou que, com atenção muito decisiva (GANHO=10), a quantização
nem mexe: acc=1.0 e L_faith=0.0 em todos os bits. Aqui varremos o GANHO para encontrar o
regime em que a evidência fica AMBÍGUA sob compressão sem a resposta ficar errada —
que é precisamente o caso que a regra 3.4 precisa enxergar (e que a métrica da v1 nunca
veria, porque ela ficava travada em 1.0).

Critério de achado: existe (ganho, bits) com acc >= 0.95 E L_faith >= 0.50.
Escala local = debug (regra 1.3). Autor: DeepSeek.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

import kvquant_v2 as K

OUT = pathlib.Path(__file__).parent / "bundles" / "v21.4-harness" / "kvquant"
OUT.mkdir(parents=True, exist_ok=True)
GANHOS = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
BITS = (16, 8, 4)
TRIALS = 20


def main():
    rng = np.random.default_rng(31415)
    linhas, achados = [], []
    for g in GANHOS:
        K.GANHO = g
        for bits in BITS:
            accs, lfs, pms = [], [], []
            for _ in range(TRIALS):
                r = K.run_trial_v2(rng, bits, "normal")
                accs.append(r["acc"]); lfs.append(r["l_faith"]); pms.append(r["pmax_medio"])
            acc, lf, pm = float(np.mean(accs)), float(np.mean(lfs)), float(np.mean(pms))
            linhas.append({"ganho": g, "bits": bits, "acc": acc, "l_faith": lf, "pmax": pm})
            print(f"ganho={g:5.2f} bits={bits:2d} acc={acc:.5f} L_faith={lf:.5f} pmax={pm:.5f}", flush=True)
            if bits == 4 and acc >= 0.95:
                pass  # o critério relativo é avaliado abaixo, no comparativo 16 vs 4

    # o achado mais limpo: compara mesma condição entre 16 e 4 bits.
    # CRITÉRIO CORRIGIDO: tem de ser RELATIVO entre precisões. Usar L_faith absoluto
    # confunde "tarefa mal-condicionada" (pmax < TAU já em FP16) com "dano da compressão".
    comparativo = []
    for g in GANHOS:
        r16 = next(l for l in linhas if l["ganho"] == g and l["bits"] == 16)
        r4 = next(l for l in linhas if l["ganho"] == g and l["bits"] == 4)
        d_lf = r4["l_faith"] - r16["l_faith"]
        comparativo.append({
            "ganho": g,
            "acc_16": r16["acc"], "acc_4": r4["acc"],
            "d_acc": r4["acc"] - r16["acc"],
            "lf_16": r16["l_faith"], "lf_4": r4["l_faith"],
            "d_l_faith": d_lf,
            "acerta_sem_fidelidade": bool(r4["acc"] >= 0.95 and d_lf >= 0.50),
        })

    achados = [{"ganho": c["ganho"], "d_acc": c["d_acc"], "d_l_faith": c["d_l_faith"]}
               for c in comparativo if c["acerta_sem_fidelidade"]]

    out = {"nota": "escala debug (regra 1.3); nao promove versao",
           "ganhos": list(GANHOS), "bits": list(BITS), "trials": TRIALS,
           "linhas": linhas, "comparativo_16_vs_4": comparativo,
           "achados_acerta_sem_fidelidade": achados,
           "veredito": ("EXISTE regime acerta-sem-fidelidade" if achados
                        else "nenhum regime com acc>=0.95 e L_faith>=0.50 nesta varredura")}
    (OUT / "v2_scan_gain.json").write_text(json.dumps(out, indent=2))
    with open(OUT / "v2_scan_gain.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ganho", "bits", "acc", "l_faith", "pmax"])
        w.writeheader()
        w.writerows(linhas)

    print("\n=== 16 bits vs 4 bits (mesma condição) ===")
    for c in comparativo:
        print(f"ganho={c['ganho']:5.2f} acc: {c['acc_16']:.4f}->{c['acc_4']:.4f} ({c['d_acc']:+.4f}) | "
              f"L_faith: {c['lf_16']:.4f}->{c['lf_4']:.4f} ({c['d_l_faith']:+.4f})"
              f"{'   <-- ACERTA SEM FIDELIDADE' if c['acerta_sem_fidelidade'] else ''}")
    print(f"\nVEREDITO: {out['veredito']}")
    print(f"ok -> {OUT}/v2_scan_gain.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
