# Celestia (Mycelium-LM)

Arquitetura própria Mycelium-LM + atenção TANGO. Fase 1 — prototipagem.
Regras de comparação em `regras_comparacao.md`. Ponte PC<->Colab via `celestia.py`.

## Status (regras rígidas)
- v21.0-baseline: VÁLIDA (fumaça + forward USA_CTX)
- v21.1-coupled: PROMOVIDA a nova baseline (3 seeds loss 0.081-0.084, clip 0.38, ctx +584% vs v21.0, sem regressão) — 2026-09-29 T4
- v21.1-coupled @200 steps T4: 0.010/0.010/0.011, clip 0.10 — REFERÊNCIA LONGA
- v21.2-langevin: REJEITADA promoção (Δloss=0 vs v21.1 @200 steps; layer_scale=0 mascara atenção). Mantida como branch-pesquisa: estável, 3 seeds 0.010, clip 0.10, sem NaN. Retomar quando layer_scale sair do zero (500+ steps ou init ≠ 0).
- v21.1 @500 steps T4: 0.003/0.003/0.003, clip 0.04 — REFERÊNCIA LONGA
- v21.3-layerscale01: REJEITADA promoção (Δ=0 vs v21.1 @500: tarefa sintética saturou no chão 0.003; indistinguível neste orçamento). Mantida como branch-pesquisa: estável, sem NaN. Diagnóstico fino (5 casas + geometria) via diag_v21.py a partir de agora.
- v21.4-harness: VALIDADO no T4 (30/09, 12 runs). mem reproduz colapso (train 0.006/val 8.37); induction derrota os dois regimes (≈ uniforme 6.24, Δ=−0.007 ruído); r_attn separa (0.01 vs 0.75) sem mover loss — atenção viva ≠ útil. Baseline segue v21.1-coupled. Próxima candidata: sonda Q/K/V na induction.
- v21.5-qkv: NÃO PROMOVE (T4, 01/10, 12 runs). induction: qkv 6.34181 vs fixo 6.32277 (ganho −0.019, FALSIFICADA nesta escala/prazo — suspeita de subtreino: 21M params novos, lr 1e-4, 300 steps, clip 1.00 o tempo todo). lag: qkv 5.68322 vs fixo 5.86600 (+0.18, CONFIRMADA com ressalva de variância). Reprodutibilidade entre sessões bit-exata (fix de seed validado no T4). Próximo: induction 1000 steps lr 3e-4 quando a quota voltar.
- probe_assoc (CEL-2.001, Claude): matriz 2×4×3 no Colab CPU (10/10) — C1–C4 confirmados: replace aprende as 4 tarefas (induction 1.31000 e assoc 1.30655 no piso 1.2939; lag/copy ≈ 0), hybrid sem regressão (C3) e gate vivo com especialização por camada (C4). Empata com QKV.
- Protocolo A/B/C em PT real: PRONTO (`lm_abc.py` + `run_lm_abc.py`, fiação validada em smoke). Braço C calibrado por contagem (<1%), fatiamento cópia-vs-geral, critério pré-registrado. Falta: corpus + T4 (quota esgotada).
- v21.2-langevin: REJEITADA p/ promoção (Δ 200 steps = 0 vs v21.1, layer_scale=0 mascara) — branch-pesquisa estável arquivado — 2026-09-29 T4
- v21.3: ABERTA (1 variável: layer_scale init 0→0.1, deixa atenção aparecer)
- v21.4-harness: INSTRUMENTO NOVO (não é candidato a promoção). O harness antigo media MEMORIZAÇÃO (lote fixo de 64 tokens): no controle `mem` deu train 0.020 / val 7.017 (pior que o uniforme 2.773). O harness novo usa validação out-of-batch: `lag` deu train 1.533 / val 1.507 (45% abaixo do uniforme) → generalização real. Instrumentação: r_t^(l), mu_e, sigma_e^2, ||dtheta||/||theta||. Sonda de capacidade: a v21 (mistura por grafo FIXO, sem Q/K/V) não aprende `induction` (2.780 ≈ uniforme 2.773); com Q/K/V em probe (model_qkv_probe.py) cai para 1.307 → a lacuna é roteamento por conteúdo. Detalhes: NOITE_2026-09-30.md, bundles/v21.4-harness/.
- `run_v21_0_baseline.py` — entrypoint executado no Colab T4 (1 célula)
- `push_results.py` — sobe só leves (csv/log/hash) de volta
- `bundles/v21.0-baseline/config.json` — config congelada
- Pesos `.h5` NUNCA sobem aqui (ficam no Drive).
