# veredito — probe_assoc, matriz completa (Colab CPU, 10/10)

**Status: C1–C4 CONFIRMADOS com 3 seeds. Atenção linear resolve o que o grafo
fixo não resolve, sem regressão, com gate vivo.**

Matriz 2 braços × 4 tarefas × 3 seeds, microscópio (2L/d64, V=16, L=32, B=8,
400 passos, lr 1e-3, clip 1.0, LS 0.1), uniforme ln16 = 2.77259, limiar C1 = 2.63396.
Selftest paralelo==scan 1.1e-05, vazamento causal 0.0.

## replace (memória no lugar do grafo)

| tarefa | seeds | média±std | C1 |
|---|---|---|---|
| induction | 1.31076/1.30358/1.31566 | 1.31000±0.00496 | ✓ |
| lag | 0.00272/0.00265/0.00264 | 0.00267±0.00003 | ✓ |
| copy | 0.00138/0.00134/0.00147 | 0.00140±0.00006 | ✓ |
| assoc (corrigida) | 1.30234/1.30337/1.31394 | 1.30655±0.00524 | ✓ |

induction e assoc no piso teórico 1.2939; lag e copy ≈ 0 (resolvidas).
Ref QKV: induction 1.30733, lag 0.00306 — **atenção linear empata com softmax**.

## hybrid (gate grafo+memória) — C3 e C4

| tarefa | média±std | alpha L0/L1 | C3 | C4 |
|---|---|---|---|---|
| induction | 1.31204±0.00453 | ~0.01 / ~0.45 | — | ✓ |
| lag | 0.00281±0.00001 | ~0.05 / ~0.23 | ✓ (≪ 1.51304+0.05) | ✓ |
| copy | 0.00139±0.00005 | ~0.51 / ~0.51 | — | ✓ |
| assoc | 1.30738±0.00368 | ~0.08 / ~0.50 | — | ✓ |

Leitura do gate: camada 0 colapsa para memória (alpha≈0.01), camada 1 mistura;
em `copy` (trivial) fica balanceado 0.51/0.51. **Especialização por profundidade
emergente** — o gate descobriu sozinho onde cada roteamento presta. C4 passa
pela média, mas o padrão por camada é o achado mais informativo.

## C5 (honestidade)

Efeitos (Δ≈1.47 em induction, ≈1.51 em lag vs fixo) ≫ stds entre seeds
(≤0.005). Nenhum NaN em 24 runs.

*Proveniência: transcrição do console Colab (resumo + 24 linhas de run);
`probe_assoc.json` bruto ficou no /tmp do runtime.*
