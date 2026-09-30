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

## Validação T4 (2026-09-30, conta GPU do Cássio, `run_v21_4_harness.py` padrão)

12 runs, 300 steps, V=512 (uniforme ln512=6.238), 3 seeds, `set_random_seed` ativo.

- **induction:** ls0.0 6.35428/6.31636/6.31852 (média 6.32972, std 0.017) |
  ls0.1 6.30835/6.35494/6.30502 (média 6.32277, std 0.023) → **Δ=−0.00695, ruído**.
  Ambas ≈ uniforme ou pior. Nenhum regime aprende induction.
- **mem (controle):** train → 0.0058, val → 8.37–8.41 (≫ uniforme).
  Colapso de memorização reproduzido no T4 — o instrumento antigo media isto.
- **r_attn separa regimes (0.01 vs 0.75) mas a loss não se move.**
  Atenção viva ≠ atenção útil: o roteamento é posicional, não por conteúdo.
  `r_attn` sozinho é métrica-vaidade para capacidade — ver Achado #2.
- **grad_conn (diag):** stone 1.47e-1 vs w_uh 3.3e-2, w_hu 4.3e-3, w_uu 9.9e-4.
  Acoplamento conectado mas 1–2 ordens abaixo da pedra. Consistente com a P2.
- Zero NaN/Inf em 12 runs. `clip_rate` induction = 1.00 (sinal de tarefa não-aprendida,
  não de instabilidade — gn ~5 estável do step 0 ao 299).

**Conclusão:** harness validado no T4; baseline segue **v21.1-coupled**; próxima
candidata de Fase 1 com evidência: sonda Q/K/V (`model_qkv_probe.py`) testada
neste mesmo harness (tarefa induction) — se aprender, vira v21.5-candidata.
