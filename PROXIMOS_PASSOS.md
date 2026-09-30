# Próximos passos — fila da noite (executar UM por vez, com evidência)

> Fila vinculada ao trabalho da noite de 2026-09-30 (coordenador: Hermes/DeepSeek).
> Regra: cada item é BOUNDED (≤ ~20 min), produz evidência em arquivo e é commitado local.
> Nada fora de `/mnt/dados/Projetos/Celestia`. Sem `git push`. Sem login. Sem `rm -rf`.

- [x] **P1 — Matriz local ampliada.** FEITO (noite de 30/09): `local_debug_v21_4.py`,
      V=16 L=32 B=8, 400 passos, tarefas `mem`/`lag`/`induction`, LS 0.0/0.1.
      `mem`: train 0.020 / val 7.017 (memorização, pior que uniforme 2.773).
      `lag`: train 1.533 / val 1.507 (45% abaixo do uniforme → generaliza).
      `induction`: 2.786 ≈ uniforme (não aprendível por esta arquitetura).
- [ ] **P2 — Acoplamento isolado no harness novo.** `coupling=False` vs `True` (LS 0.1 fixo)
      na tarefa `lag` → o acoplamento `w_hu/w_uh/w_uu` muda o resultado além do `layer_scale`?
- [ ] **P3 — Sonda de capacidade completa.** Rodar `copy` e `assoc` (probes) nas mesmas
      condições e registrar a tabela final de sondas (qual capacidade existe hoje).
- [x] **P4 — Proposta Q/K/V (arquivo NOVO, `model_qkv_probe.py`).** FEITO (30/09):
      subclasse que troca só a mistura (grafo fixo → Q/K/V causal + softmax).
      Resultado: `induction` 2.78047 → **1.30733**; `lag` 1.51304 → **0.00306**.
      Hipótese confirmada: a lacuna é roteamento por conteúdo.
      Evidência: `bundles/v21.4-harness/probe_qkv/probe_qkv.json`.
- [ ] **P5 — Protótipo KV-cache (prioridade 2 do inst.ger.1).** `kvquant_v21.py` em numpy
      puro: FP16 vs INT8 vs INT4 com `L_faith` (regra 3.4) — só accuracy não basta.
- [ ] **P6 — Verificação final.** Re-rodar tudo do zero (mesmas seeds), conferir CSVs e
      `diag.json`, atualizar `NOITE_2026-09-30.md`, commit local, deixar pronto p/ push manual.

## Restrições permanentes (não violar de madrugada)
- Não alterar `model_v21.py` nem `regras_comparacao.md` (documentos comparativos).
  Propostas vão em arquivo separado, marcadas como proposta.
- Não rodar treino pesado: RAM livre ~0,9 GB (Firefox do Cássio está aberto — não mexer).
- Não usar GPU/Colab (exige login do Cássio → proibido sem autorização).
- Todo número local é **debug** (regra 1.3). Nenhum veredito de promoção sai daqui.
