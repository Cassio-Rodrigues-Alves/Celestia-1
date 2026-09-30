# Instruções do relatório da manhã (cron 08:00) — projeto Celestia

Produza o **relatório da noite** do projeto Celestia para o Cássio, em português
brasileiro, direto, honesto e sem enfeite.

## Ler antes
- `/mnt/dados/Projetos/Celestia/NOITE_2026-09-30.md`
- `/mnt/dados/Projetos/Celestia/PROXIMOS_PASSOS.md`
- `bundles/v21.4-harness/`: `config.json`, `diff_vs_anterior.md`, `veredito.md`,
  `local_debug/local_debug.json`, `probe_qkv/probe_qkv.json`
- `git -C /mnt/dados/Projetos/Celestia log --oneline -20` e `... status --short`
- Verifique se os checkpoints horários acrescentaram seções novas ao `NOITE_2026-09-30.md`.

## Formato (Markdown do Telegram, bullets, SEM tabelas)
1. Uma linha de status geral.
2. Achado 1: o instrumento antigo media memorização — com os números reais.
3. Achado 2: estrutural, a atenção não tem Q/K/V — evidência + resultado da sonda QKV.
4. O que foi implementado (arquivos novos) e o que foi comitado (local, NÃO empurrado).
5. O que NÃO foi feito e por quê (GPU/Colab exigem o login dele; nada foi para o GitHub;
   nada pago).
6. O que ele precisa decidir/rodar agora: comando de push pronto e a célula do Colab,
   citando `run_v21_4_harness.py`.
7. Pendências e dúvidas em aberto.

## Regras
- Use SOMENTE números que estejam nos arquivos; não invente.
- Deixe claro o que é debug local (regra 1.3) e o que valeria no T4.
- No fim, liste os itens que precisam de autorização dele (login, GPU, storage,
  download de dados).
