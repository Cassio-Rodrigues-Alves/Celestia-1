# veredito — v21.6 gate-fixo, piloto (T4, 04/10)

**Status: HIPÓTESE DO GATE MORTA. Forçar a atenção aberta não muda nada.**

fixo/induction/s0, 1000 steps, lr 3e-4, `layer_scale` congelado em 1.0
(`--freeze-ls`, sem peso no otimizador — flag validada no repo):
train 6.27968 → 6.24287 (curva PLANA o trajeto todo), **loss_val 6.24997**
vs uniforme 6.23832. Critério (< 5.93): longe.
`layer_scale_L0 = 1.00000` (congelado como desenhado), clip 1.00 sempre,
grad_conn stone 6.8e-03 (sinal útil quase nulo).

## Tabela de todas as tentativas induction full-scale (V=512)

| tentativa | loss_val | vs uniforme 6.238 |
|---|---|---|
| fixo LS 0 (v21.4) | 6.32277 (3 seeds) | ≈ |
| fixo LS 0.1 (v21.4) | 6.32277 | ≈ |
| qkv 300 steps (v21.5) | 6.34181 | ≈ |
| qkv 1000 steps lr 3e-4 (piloto) | 6.25580 | ≈ |
| **fixo LS congelado 1.0 (v21.6)** | **6.24997** | ≈ |

Cinco configurações, mesmo resultado: nada sai do uniforme. O gate nunca foi
o bloqueador (já estava suficientemente aberto em 0.1 — r_attn 0.75 provava).
O circuito de indução não se forma na escala real em orçamento prático, com ou
sem gate, com ou sem QKV. Opção (a) da fila: ENCERRADA com negativo.

*Proveniência: transcrição do console (artefatos no /tmp do Colab).*
