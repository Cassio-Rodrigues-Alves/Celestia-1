# Próximos passos — fila da noite (executar UM por vez, com evidência)

> Fila vinculada ao trabalho da noite de 2026-09-30 (coordenador: Hermes/DeepSeek).
> Regra: cada item é BOUNDED (≤ ~20 min), produz evidência em arquivo e é commitado local.
> Nada fora de `/mnt/dados/Projetos/Celestia`. Sem `git push`. Sem login. Sem `rm -rf`.

- [x] **P1 — Matriz local ampliada.** FEITO (noite de 30/09): `local_debug_v21_4.py`,
      V=16 L=32 B=8, 400 passos, tarefas `mem`/`lag`/`induction`, LS 0.0/0.1.
      `mem`: train 0.020 / val 7.017 (memorização, pior que uniforme 2.773).
      `lag`: train 1.533 / val 1.507 (45% abaixo do uniforme → generaliza).
      `induction`: 2.786 ≈ uniforme (não aprendível por esta arquitetura).
- [x] **P2 — Acoplamento isolado no harness novo.** DUAS medições (30/09 e 02/10):
      (a) sessão pré-fix: `lag`, LS 0.1, 400 steps, 1 seed → sem 1.50972 / com
      1.50177, Δ=−0.00796 (arquivado em `p2_coupling_pre_fix.md`);
      (b) re-medição DeepSeek c/ seed determinística: 300 steps, 2 seeds,
      Δ=−0.00324. O veredito automático ("acima do piso 0.0") foi CORRIGIDO
      (`CORRECAO_P2.md`): piso 0.0 é de re-run, não de comparação entre braços
      com inits diferentes; Δ≈0.003–0.008 com n≤2 = ruído. **Veredito conjunto:
      INDISTINGUÍVEL.** Evidência: `bundles/v21.4-harness/p2_coupling/`.
- [x] **P3 — Sonda de capacidade completa.** FEITO (30/09): `local_p3_probes.py`,
      `copy`+`assoc` × morta/viva + `induction` da P1. Tabela final: todas as 6
      células ≈ uniforme (2.77259) — copy 2.77802/2.77804, assoc 2.77446/2.77494,
      induction 2.78592/2.78707. **Nenhuma capacidade por conteúdo existe hoje,
      em nenhum regime.** Evidência: `bundles/v21.4-harness/p3_probes/probe_table.json`.
- [x] **P4 — Proposta Q/K/V (arquivo NOVO, `model_qkv_probe.py`).** FEITO (30/09):
      subclasse que troca só a mistura (grafo fixo → Q/K/V causal + softmax).
      Resultado: `induction` 2.78047 → **1.30733**; `lag` 1.51304 → **0.00306**.
      Hipótese confirmada: a lacuna é roteamento por conteúdo.
      Evidência: `bundles/v21.4-harness/probe_qkv/probe_qkv.json`.
- [x] **P5 — Protótipo KV-cache (prioridade 2 do inst.ger.1).** FEITO (30/09):
      `kvquant_v21.py` (numpy puro). FP16 acc 1.000 / INT8 acc 0.984 / INT4 acc **0.802**
      (−19,8 pp); KL cresce com a compressão. Ressalva: a `L_faith` implementada **não
      discriminou** (ficou 1.0 até em INT4) → a definição da métrica precisa ser corrigida
      antes da regra 3.4 valer. Evidência: `bundles/v21.4-harness/kvquant/`.
- [x] **P6 — Verificação final.** FEITO (30/09): `verify_v21_4.py` (artefatos OK,
      consistência OK, reprodutibilidade **DIVERGIU**) + `noise_floor_v21_4.py`
      (spread 0,0269) → caçada a causa raiz: **`tf.random.set_seed` não controla a
      inicialização do Keras 3** (max|diff| 0,695 entre duas inits com a mesma seed).
      Corrigido com `tf.keras.utils.set_random_seed` → **spread 0,00000000 (bit-exato)**.
      Evidência: `bundles/v21.4-harness/{verify,noise}/`. Detalhes: NOITE_2026-09-30.md §11.

## Restrições permanentes (não violar de madrugada)
- Não alterar `model_v21.py` nem `regras_comparacao.md` (documentos comparativos).
  Propostas vão em arquivo separado, marcadas como proposta.
  (Exceções registradas com motivo: gancho `attn_cls` neutro + `ffn_mult` neutro,
  ambos default-idênticos — ver log.)
- Não rodar treino pesado: RAM livre ~0,9 GB (Firefox do Cássio está aberto — não mexer).
- Não usar GPU/Colab (exige login do Cássio → proibido sem autorização).
- Todo número local é **debug** (regra 1.3). Nenhum veredito de promoção sai daqui.

## Pivot 03/10 (decisão do Cássio — Porta 1): v21.5-qkv ARQUIVADA

Motivo em `bundles/v21.5-qkv/veredito.md` (seção ARQUIVAMENTO FORMAL). A fila
P1–P6 está cumprida; a fila nova nasce da pergunta pivot: **o que o TANGO faz
de único SEM conteúdo?** (roteamento posicional funciona — `lag` prova).
Candidatos em aberto, nenhum iniciado: (a) QKV com layer_scale fixo (sonda) —
  EXECUTADA 04/10 (v21.6-gatefix): val 6.24997 ≈ uniforme, hipótese do gate
  MORTA, veredito em `bundles/v21.6-gatefix/VEREDITO.md`;
(b) caracterizar capacidade posicional (eco com atraso variável, janela de
alcance do grafo k=6); (c) A/B/C oficial em PT se GPU longa aparecer
(protocolo pronto em `run_lm_abc.py`, piloto calibra N).
