# veredito — probe_lag_reach: até onde o grafo enxerga (Colab CPU, 04/10)

**O TANGO posicional é uma delay line de k=6 — nem mais, nem menos.**

Matriz 8 atrasos × 3 seeds, fixo (v21.1: coupling, LS 0.1), microscópio, 400 steps.
Loss bruta NÃO-monotônica (pico em lag10) porque mistura alvos-fáceis
(sentinela 0, fração crescente com j) com alvos-eco. Descontada a sentinela
(verificação: 4 atrasos convergem para o MESMO 2.717–2.721 — a conta fecha):

| nominal | efetivo | echo (descontado) | leitura |
|---|---|---|---|
| lag2 | 0 | 0.001 | identidade, sem atenção |
| lag4 | 2 | 1.602 | aprende |
| lag6 | 4 | 1.780 | aprende |
| lag8 | 6 | 1.374 | aprende (borda da janela) |
| lag10 | 8 | 2.717 | uniforme: NÃO aprende |
| lag12 | 10 | 2.717 | uniforme |
| lag16 | 14 | 2.721 | uniforme |
| lag20 | 18 | 2.718 | uniforme |

**Penhasco exatamente entre atraso efetivo 6 e 8 = borda da janela direta k=6.**
Além dela, o resíduo (2.72 ≈ ln15 = 2.708) é só "nunca preveja 0" — zero eco.
Com 2 camadas o alcance 2-hop (12) existiria em princípio; **o modelo não
aprende composição multi-hop**: só usa a janela direta.

Spec do roteamento posicional, pela primeira vez exata: eco com atraso
efetivo ≤ 6 aprende; ≥ 8, nunca (neste orçamento). É o delay line fixo que o
`lag` usa — e tudo que ele usa.

*Proveniência: console Colab (24 runs, stds ≤ 0.04); JSON bruto no /tmp do runtime.*
