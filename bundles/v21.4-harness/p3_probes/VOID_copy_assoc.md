# VOID — números pré-fix de `copy` e `assoc` (30/09)

Análise de alinhamento (Claude, linha CEL-2.001): `train_run` usa
`loss(yb[:,1:], logits[:,:-1])` sobre `(seq[:-1], seq[1:])` → alvo(p) = seq[p+2].

- `copy` (versão antiga): alvo = seq[p+2] de sequência i.i.d. = **impossível para
  qualquer modelo causal**, inclusive Q/K/V. Os valores ≈ uniforme da P3 para
  `copy` são vacuidade do desenho, não medição de capacidade. VOID.
- `assoc` (versão antiga): a sequência terminava nas consultas — os valores como
  alvo apareciam antes das consultas (imprevisíveis sempre) e as chaves eram
  copiáveis por offset fixo variável. Media cópia de offset, não associação. VOID.

Seguem VÁLIDOS (tarefas intactas): `mem`, `lag`, `induction` — incluindo
`probe_qkv.json` (induction+lag) e as linhas `induction::` da P1.
`tasks_v21.py` foi corrigido (`copy` = identidade sanity; `assoc` com respostas
após as consultas, piso 14/30·lnV para modelo com conteúdo).
