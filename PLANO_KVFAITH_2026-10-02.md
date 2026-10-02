# Correção da métrica L_faith (regra 3.4) — Plano de Implementação

> **Método:** skill `writing-plans` (biblioteca compartilhada `~/.agents/skills`,
> instalada via Antigravity — o que o reel de @andredozero demonstra).
> Execução: `executing-plans` + verificação por `verification-before-completion`.

**Objetivo:** fazer a métrica de fidelidade (`L_faith`, regra 3.4 do projeto Celestia)
realmente discriminar compressão de KV-cache, quando a versão anterior não discriminava.

**Arquitetura:** a v1 media "sustentação" contra o índice da chave do **gabarito** — por
isso uma resposta errada-mas-confiante ainda contava como sustentada, e `L_faith` ficou
1.0 até em INT4. A v2 mede a fidelidade **do que o modelo de fato usou**: a resposta
precisa ser rastreável a uma entrada do cache E corresponder à entrada que recebeu a
maior massa de atenção. Inclui um **controle de falsificação** (cache deliberadamente
corrompido) onde a métrica TEM de acusar fidelidade zero.

**Tech Stack:** Python + NumPy (sem TensorFlow — a métrica é sobre o cache, não sobre treino).

**Spec:** `NOITE_2026-09-30.md` §4 (achado da `L_faith` que não discriminou) e
`regras_comparacao.md` regra 3.4.

## Restrições Globais
- Escala local = **debug** (regra 1.3). Nada aqui promove versão.
- Não alterar `regras_comparacao.md` nem `model_v21.py`.
- Saídas em `bundles/v21.4-harness/kvquant/`.

## Critérios de aceitação (pré-registrados, verificados ao final)
- **A1** FP16 → `L_faith_v2` ≤ 0.05 (referência sem compressão não pode acusar infidelidade).
- **A2** Controles de falsificação (dois) → `L_faith_v2` ≥ 0.90 em ambos:
  - (a) respostas lidas de um cache **independente** (não sustentadas por este contexto);
  - (b) atenção **difusa** (uniforme) → a resposta é uma mistura, não corresponde a
    nenhuma entrada do cache.
  *Este é o critério que a v1 falhava: uma métrica que não acusa resposta não-sustentada
  não mede nada.*
  **Correção de rota registrada:** a primeira versão deste plano usava "valores
  embaralhados" como controle — errado. Embaralhar valores deixa toda resposta
  sustentada (pelo cache errado): isso é falha de **acurácia**, não de **fidelidade**.
  O controle tem de produzir uma resposta que o contexto não sustenta.
- **A3** INT8 → `L_faith_v2` ≤ 0.05 e `acc` ≥ 0.95.
- **A4** INT4 → medir e reportar (sem limiar fixo, é o dado que interessa).
- **A5** Execução sem erro, JSON salvo, números citáveis no relatório.

## Foco de Revisão (entradas que podem quebrar a métrica)
1. Query sem evidência clara (massa difusa) — deve contar como NÃO sustentada.
2. Resposta que não corresponde a nenhuma entrada do cache ("alucinação") — NÃO sustentada.
3. Empate de massa entre duas chaves — decidir por índice estável, não por ordem de `argsort`.
4. Cache corrompido — tem de acusar (é o controle A2).
5. `acc` alto com fidelidade baixa — o caso que a regra 3.4 existe para pegar.

---

### Task 1: redefinir a fidelidade e escrever o teste que falha

**Arquivos:**
- Criar: `kvquant_v2.py`
- Evidência: `bundles/v21.4-harness/kvquant/v2_corrigida.json`

**Interfaces:**
- Produz: `l_faith_v2(answers, vals, used_keys, tol) -> (l_faith: float, detalhes: dict)`
- Produz: `run_trial_v2(rng, bits, corrupt=False) -> dict`

- [x] **Step 1:** definir `supported(i)` = a resposta está a menos de `TOL` de ALGUMA
      entrada do cache **e** essa entrada é a de maior massa (`used_key`).
- [x] **Step 2:** caso de teste que a v1 errava — controles de falsificação têm de ser
      acusados (`L_faith ≥ 0.90`); a v1 dava 0.0. Ver `RESULTADO_KVFAITH_2026-10-02.md`.
- [x] **Step 3:** implementar `run_trial_v2` espelhando a v1 na distribuição (mesma
      construção de chaves/valores/queries).
- [x] **Step 4:** rodar `bits ∈ {16, 8, 4}` × modos; 20 trials. FEITO — mais uma varredura
      de ganho (`kvquant_scan_gain.py`).
- [x] **Step 5:** checar A1–A5. **TODOS PASSARAM** → métrica validada. E o achado principal
      foi NEGATIVO: em cache bem-condicionado, 4 bits não degrada nada (Δacc=0, ΔL_faith=0).
- [x] **Step 6:** commit local (sem push).
