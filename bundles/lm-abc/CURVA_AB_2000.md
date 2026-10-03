# Curva A/B @2000 steps em PT real (T4, 02–03/10) — 1 seed

**Status: sem sinal de QKV em nenhum orçamento testado (500 e 2000).
Tendência a arquivar; veredito definitivo exige a oficial 3-seed.**

| braço | train | val | cópia | geral |
|---|---|---|---|---|
| A/s0 | 5.6227 | 6.44901 | 4.42808 | 7.19864 |
| B/s0 | 5.7033 | 6.51395 | 4.49333 | 7.26347 |

Δ(B−A) val = **+0.06494** (1 seed, sem piso). Por fatia: cópia +0.065, geral
+0.065 — uniforme, sem benefício nem onde roteamento deveria brilhar.
Curvas coladas o trajeto todo (≤0.05 por checkpoint); ambas ainda descendo
suave no fim (6.19→5.62 nos últimos 200 steps de A).

Leitura: com 500 steps Δ era +0.016; com 2000, +0.065. Direção consistente
(B nunca abaixo de A) e o gap alargou — compatível com "QKV é peso-morto que
o otimizador contorna" (21M params aleatórios sem gradiente útil, cf. gate
fechando no piloto da induction). Não é prova (1 seed, sem piso, sem plena
convergência), mas o padrão em 3 orçamentos (50/200 sintético, 500/2000 PT)
é o mesmo: **nenhum sinal positivo do QKV em lugar nenhum**.

*Proveniência: transcrição do console (runs em /tmp do Colab).*
