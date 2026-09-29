# Regras Rígidas — Comparação entre Versões Mycelium-LM

Status: vinculante para Fase 1. Nenhuma versão promove sem passar TODAS.

## 0. Princípio
Sem número comparável, sem claim. Opinião não promove versão.

## 1. Congelamento de condições (obrigatório)
1.1. Mesmo corpus, mesmo split, mesmo tokenizer, mesmo `SEQ_LEN`, `D_MODEL`, `VOCAB_SIZE` salvo exceção documentada.
1.2. Mesma seed base + 2 seeds extras para o teste final (mínimo 3 runs para promoção).
1.3. Mesmo hardware de referência (T4 Colab) para números finais. Número de WSL/local vale só como debug.
1.4. Qualquer mudança de `BETA_START`, `CLIP_NORM`, LR, warmup invalida comparação direta — vira nova baseline, não "mesmo teste".
1.5. Checkpoint: `mycelium_milestone_ep{N:04d}.weights.h5` imune à poda. `skip_mismatch=True` proibido em teste final (mascara queda de `w_stone`/`gate` como no bug ep53).

## 2. Métricas obrigatórias (toda versão loga todas)
- `loss_train`, `loss_val`, `perplexidade_val`
- `overfitting_gap = loss_val - loss_train`
- `clip_rate`, `grad_norm`, `loss_scale`, `overflow_fp16_rate`, `NaN_count`
- `drift_max`, `||W^T W - I||_F` (ortogonalidade)
- Telemetria TANGO: `s_l(t)` e RMS em t=0,1,2, trajetória Lyapunov completa (não só t=2), maior autovalor real de `w_stone`, `||w_hu|| / ||w_uh||`, `r_t^(l)` residual, `mu_e`, `sigma_e^2`
- Custo: `tokens/s`, `bytes/token`, `T_read + T_decode + T_matmul`, VRAM pico

Faltou uma métrica = teste inválido, refazer.

## 3. Testes funcionais mínimos (não só loss)
3.1. Coerência PT em 2.6% do treino (teste de fumaça histórico) — regrediu, reprova.
3.2. Uso de contexto: `layer_scale=1.0 vs 0.0` tem que mudar output. Se igual, modelo ignora contexto = reprova.
3.3. `facts` ancorado: `u_tan == u_facts` bit-exato. Se explorar, reprova.
3.4. Fidelidade (se usar memória/RAG/KV-cache): `L_faith = 1 - |R_sustentado|/|R|`. Só accuracy não basta. INT4 só com `L_faith` preservado.

## 4. Critério de promoção (todos têm que bater)
- `Δloss_val < 0` com p<0.05 nas 3 seeds OU `Δloss_val == 0` + ganho custo/fidelidade ≥10% sem regressão funcional.
- `clip_rate` não pode subir >2pp vs baseline (lição `BETA_START=1.0`).
- `overfitting_gap` não pode abrir >5% relativo.
- Zero NaN não-explicado. 1 NaN = investigar, 2 = reprova automática.
- Compatibilidade: checkpoint novo carrega no loader antigo ou documenta break + script migração. Quebra silenciosa reprova.

## 5. Proibições
- Não trocar 2 coisas de uma vez (ex: TANGO + difusão juntos). 1 variável por versão.
- Não comparar FP16 vs INT8 vs 2-bit pelo mesmo `ΔL` sem `T_decode` junto (regra Leech-lattice: representação só vale com caminho eficiente).
- Não descartar item que estourou iteração sem logar como exemplo negativo ou lixo — decidir e registrar.
- Não tratar decisão Fase 1 como definitiva. Nenhum treino oficial antes Fase 2 fechada.

## 6. Artefato por versão (pasta `vXX.Y/`)
`config.json`, `diff_vs_anterior.md`, `csvs_por_epoca/`, `banner_sinais_criticos.log`, `teste_contexto.log`, `custo_serving.md`, `veredito.md` (promove / rejeita / branch-pesquisa).

Quebrou regra 1-5 = `veredito: inválido`, nem entra em ranking.
