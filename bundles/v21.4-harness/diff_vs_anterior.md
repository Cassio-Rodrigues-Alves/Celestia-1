# diff vs anterior — v21.4-harness

## O que NÃO mudou (importante)
- `model_v21.py` — **zero alterações**. `w_stone`, `w_hu`, `w_uh`, `w_uu`, `gate`,
  `layer_scale`, `beta`, a adjacência `_A_np`, o FFN, o head: tudo idêntico à v21.3.
- Nenhum hiperparâmetro de treino (`beta_start`, `clip_norm`, LR, warmup) mudou.
  Logo a regra 1.4 não é violada: continua a MESMA baseline, medida melhor.

## O que mudou
| Item | Antes (v21.0–v21.3) | Agora (v21.4) |
|---|---|---|
| Tarefa de treino | `xb` aleatório FIXO (B=2,L=32,V=512) por 500 passos | `tasks_v21.make_env`: dados NOVOS a cada passo + validação congelada (8 lotes) |
| O que a loss mede | memorização de 64 tokens (chão 0.003) | generalização out-of-batch (sem chão de memorização) |
| Métrica de atenção | `nhu`/`nuh` (normas do acoplamento) | `+ r_t^(l)` ganho residual por camada (attn vs ffn) |
| Gradientes | só `grad_norm`/clipping | `+ mu_e`, `sigma_e^2` (expoentes log2|g|) por camada + n_zero_grad |
| Deriva de pesos | ausente | `||dtheta||_2/||theta||_2` por passo |
| Validação | nenhuma (loss era o próprio lote) | `loss_val`, `perplexidade_val`, `overfitting_gap` |
| Teste de contexto (regra 3.2) | manual | `runner_v21.context_test` automatizado |
| Diagnóstico de geometria | `diag_v21.py` | o mesmo, agora acoplado ao run |

## Arquivos novos
- `tasks_v21.py` — geradores de tarefa (mem / lag / copy / induction / assoc).
- `stab_v21.py` — instrumentação (r_t, mu_e, sigma_e^2, ||dtheta||/||theta||, clip/NaN).
- `runner_v21.py` — loop único com todas as métricas obrigatórias da regra 2.
- `run_v21_4_harness.py` — entrada oficial do Colab (esta versão).
- `local_debug_v21_4.py` — validação local em escala reduzida (SÓ debug, regra 1.3).

## Consequência metodológica
Um Δloss medido no harness antigo **não é comparável** com um Δloss medido no harness novo.
A troca de instrumento obriga a uma **nova linha de base**: até ela existir, não há ranking.
Isso é explicitado no veredito desta versão.

*Autoria: DeepSeek (noite 2026-09-30).*
