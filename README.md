# Celestia (Mycelium-LM)

Arquitetura própria Mycelium-LM + atenção TANGO. Fase 1 — prototipagem.
Regras de comparação em `regras_comparacao.md`. Ponte PC<->Colab via `celestia.py`.

## Status (regras rígidas)
- v21.0-baseline: VÁLIDA (fumaça + forward USA_CTX)
- v21.1-coupled: PROMOVIDA a nova baseline (3 seeds loss 0.081-0.084, clip 0.38, ctx +584% vs v21.0, sem regressão) — 2026-09-29 T4
- v21.1-coupled @200 steps T4: 0.010/0.010/0.011, clip 0.10 — REFERÊNCIA LONGA
- v21.2-langevin: REJEITADA promoção (Δloss=0 vs v21.1 @200 steps; layer_scale=0 mascara atenção). Mantida como branch-pesquisa: estável, 3 seeds 0.010, clip 0.10, sem NaN. Retomar quando layer_scale sair do zero (500+ steps ou init ≠ 0).
- v21.2-langevin: REJEITADA p/ promoção (Δ 200 steps = 0 vs v21.1, layer_scale=0 mascara) — branch-pesquisa estável arquivado — 2026-09-29 T4
- v21.3: ABERTA (1 variável: layer_scale init 0→0.1, deixa atenção aparecer)
- `run_v21_0_baseline.py` — entrypoint executado no Colab T4 (1 célula)
- `push_results.py` — sobe só leves (csv/log/hash) de volta
- `bundles/v21.0-baseline/config.json` — config congelada
- Pesos `.h5` NUNCA sobem aqui (ficam no Drive).
