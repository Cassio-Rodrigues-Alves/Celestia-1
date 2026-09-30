# Próximos passos — fila da noite (executar UM por vez, com evidência)

> Fila vinculada ao trabalho da noite de 2026-09-30 (coordenador: Hermes/DeepSeek).
> Regra: cada item é BOUNDED (≤ ~20 min), produz evidência em arquivo e é commitado local.
> Nada fora de `/mnt/dados/Projetos/Celestia`. Sem `git push`. Sem login. Sem `rm -rf`.

- [ ] **P1 — Matriz local ampliada.** `local_debug_v21_4.py` com V=32, B=8, 600 steps,
      tarefas `mem`/`lag`/`induction`, LS 0.0/0.1 → confirmar que `lag` discrimina e que
      `induction` fica no uniforme (prova de ausência de atenção por conteúdo).
- [ ] **P2 — Acoplamento isolado no harness novo.** `coupling=False` vs `True` (LS 0.1 fixo)
      na tarefa `lag` → o acoplamento `w_hu/w_uh/w_uu` muda o resultado além do `layer_scale`?
- [ ] **P3 — Sonda de capacidade completa.** Rodar `copy` e `assoc` (probes) nas mesmas
      condições e registrar a tabela final de sondas (qual capacidade existe hoje).
- [ ] **P4 — Proposta Q/K/V (arquivo NOVO, `model_qkv_probe.py`).** Implementar atenção
      por conteúdo (Q,K,V + softmax causal) como PROBE separada, sem tocar `model_v21.py`,
      e medir se a sonda `induction` deixa de ficar no uniforme. Isto é EXPERIMENTO de Fase 1,
      não promoção e não mudança de baseline.
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
