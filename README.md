# Celestia (Mycelium-LM)

Arquitetura própria Mycelium-LM + atenção TANGO. Fase 1 — prototipagem.
Regras de comparação em `regras_comparacao.md`. Ponte PC<->Colab via `celestia.py`.

## Layout
- `run_v21_0_baseline.py` — entrypoint executado no Colab T4 (1 célula)
- `push_results.py` — sobe só leves (csv/log/hash) de volta
- `bundles/v21.0-baseline/config.json` — config congelada
- Pesos `.h5` NUNCA sobem aqui (ficam no Drive).
