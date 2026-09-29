# Celestia (Mycelium-LM)

Arquitetura própria Mycelium-LM + atenção TANGO. Fase 1 — prototipagem.
Regras de comparação em `regras_comparacao.md`. Ponte PC<->Colab via `celestia.py`.

## Status (regras rígidas)
- v21.0-baseline: VÁLIDA (fumaça + forward USA_CTX)
- v21.1-coupled: PROMOVIDA a nova baseline (3 seeds loss 0.081-0.084, clip 0.38, ctx +584% vs v21.0, sem regressão) — 2026-09-29 T4
- `run_v21_0_baseline.py` — entrypoint executado no Colab T4 (1 célula)
- `push_results.py` — sobe só leves (csv/log/hash) de volta
- `bundles/v21.0-baseline/config.json` — config congelada
- Pesos `.h5` NUNCA sobem aqui (ficam no Drive).
