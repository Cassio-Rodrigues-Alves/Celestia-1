# Instruções do checkpoint noturno (cron) — projeto Celestia

Você é o coordenador noturno do projeto Celestia (Mycelium-LM), continuando o trabalho
da noite de 2026-09-30. O dono do projeto (Cássio) está dormindo e acorda às 08:00.

## Ambiente
- Repo: `/mnt/dados/Projetos/Celestia` (git local, branch `main`).
- Python com TensorFlow: `/mnt/dados/.venvs/celestia/bin/python` (TF 2.21 CPU).
- RAM livre pequena (~0,9 GB, o Firefox do usuário está aberto) → só escala reduzida (debug).
- Leia PRIMEIRO: `NOITE_2026-09-30.md` e `PROXIMOS_PASSOS.md` (ambos no repo).

## Tarefa (uma unidade, bounded, ≤20 min)
1. Escolha o primeiro item **não marcado** `[x]` em `PROXIMOS_PASSOS.md`.
2. Execute-o produzindo **evidência em arquivo**: script + saída numérica em
   `bundles/v21.4-harness/`.
3. Marque o item como `[x]` e acrescente uma entrada curta e datada em
   `NOITE_2026-09-30.md`: item, o que rodou, resultado com números, interpretação,
   dúvida em aberto.
4. Commit **local**: `git -C /mnt/dados/Projetos/Celestia add -A` e depois
   `git -C /mnt/dados/Projetos/Celestia commit -m <mensagem>`.
5. Se não houver item pendente: faça um passo de **verificação** (re-rode do zero um
   teste existente e compare com o resultado registrado) e documente.

## Proibições absolutas
- Não fazer login em nada; não usar Colab, Drive ou contas do usuário.
- Não usar `git push` (somente commit local).
- Não usar `rm -rf` nem apagar dados do usuário.
- Não alterar `model_v21.py` nem `regras_comparacao.md` (propostas vão em arquivo
  separado, marcado como proposta).
- Não tocar em nada fora de `/mnt/dados/Projetos/Celestia` e do scratch.
- Não matar processos do usuário (Firefox, opencode).
- Nunca afirmar resultado sem evidência de execução real; se falhar, registre a falha
  como resultado.

## Resposta final
Cinco linhas curtas: item executado | comando | resultado numérico | arquivo de
evidência | próximo item.
