# veredito — v21.4-harness

**Status: NÃO SE APLICA — v21.4 não é candidato a promoção.**

A v21.4 não modifica a arquitetura; modifica o **instrumento de medição** e adiciona
**instrumentação**. Pelas regras, promoção compara *versões da arquitetura* sob o *mesmo
instrumento*. Não há arquitetura nova aqui para promover.

## Efeito real da v21.4
1. Invalida a comparabilidade com os números do harness antigo (chão de memorização 0.003).
   A partir de agora, todo Δloss precisa ser medido no harness novo — ou seja, é preciso
   **rebaselinar** a v21.1-coupled (a baseline atual) no harness novo antes de qualquer ranking.
2. Dá à Fase 1 o que ela não tinha: um teste com poder de discriminação, telemetria de
   estabilidade (r_t, mu_e, sigma_e^2, ||dtheta||/||theta||) e validação out-of-batch.

## O que a v21.4 NÃO faz
- Não promove nenhuma variante de arquitetura.
- Não decide autoregressivo vs difusão (proibido na Fase 1 — ver PLANO_GUARDADO).
- Não inicia treino real (Fase 2 não fechada).

*Autoria: DeepSeek (noite 2026-09-30).*
