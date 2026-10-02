# Resultado — correção da métrica de fidelidade (L_faith) e re-teste INT8/INT4

**Data:** 02/10/2026 · **Autor:** Hermes (modelo deepseek-v4.1-flash)
**Método:** skills `writing-plans` + `verification-before-completion` da biblioteca
compartilhada `~/.agents/skills` (a instalação global que o reel de @andredozero demonstra)
**Plano:** `PLANO_KVFAITH_2026-10-02.md` · **Artefatos:** `bundles/v21.4-harness/kvquant/`

---

## 1. O que estava quebrado

A `L_faith` da v1 (noite de 30/09) media "sustentação" contra o índice da chave do
**gabarito**. Resultado: uma resposta errada-mas-confiante continuava contando como
sustentada, e `L_faith` ficou **1.00000 em todas as precisões, inclusive INT4** — ou seja,
a métrica não media fidelidade nenhuma. Só a acurácia acusava algo.

## 2. A correção

`kvquant_v2.py` — resposta **aterrada** exige duas coisas:
1. **evidência inequívoca**: maior massa de atenção `P_max ≥ 0.50`;
2. **rastreabilidade**: `‖A − V[used_key]‖ / ‖V[used_key]‖ < 0.25`, com `used_key` = a
   entrada que recebeu a maior massa.

Mais dois **controles de falsificação** (sem eles a métrica não vale):
- `answers_alheias` — respostas lidas de um cache independente;
- `massa_difusa` — atenção uniforme (resposta é mistura, não corresponde a entrada alguma).

## 3. Verificação (critérios pré-registrados — todos passaram)

| Critério | Resultado | |
|---|---|---|
| A1 — FP16 ⇒ `L_faith` ≤ 0.05 | **0.00000** | OK |
| A2a — respostas alheias acusadas (≥ 0.90) | **1.00000** | OK |
| A2b — massa difusa acusada (≥ 0.90) | **1.00000** | OK |
| A3 — INT8: `L_faith` ≤ 0.05 e `acc` ≥ 0.95 | **0.00000 / 1.00000** | OK |
| A5 — execução limpa + JSON salvo | exit 0 | OK |

**Veredito: MÉTRICA VALIDADA.**

## 4. Achado principal — e ele derruba o número que eu mesmo reportei

Varredura de 6 ganhos de atenção × 3 precisões (20 trials cada), em cache **bem-condicionado**
(chaves e valores unitários quase-ortogonais, consulta alinhada à chave-alvo):

| ganho | acc 16b → 4b | Δacc | L_faith 16b → 4b | ΔL_faith |
|---|---|---|---|---|
| 0.25 – 4.0 | 1.0000 → 1.0000 | 0.0000 | 1.0000 → 1.0000 | 0.0000 |
| 10.0 | 1.0000 → 1.0000 | 0.0000 | 0.0000 → 0.0000 | 0.0000 |

**Δacc = 0.0000 e ΔL_faith = 0.0000 em TODOS os regimes.** Nenhum regime onde a acurácia
sobreviva e a fidelidade caia (`v2_scan_gain.json`, veredito automático).

Interpretação (evidência ≠ interpretação, como pede o doc §31):
- *Evidência:* quantização uniforme per-tensor de 4 bits não degradou acurácia **nem**
  fidelidade neste cache bem-condicionado, em nenhum ganho testado.
- *Interpretação:* o número de ontem ("INT4 −19,8 pp de acurácia", `kvquant_v21.py`) era
  **artefato da montagem**: na v1 os scores ficavam ~N(0, 1/64) — atenção quase uniforme
  por construção — e aí sim o ruído de quantização virava o `argmax`. Com o cache
  bem-condicionado esse efeito desaparece.
- *Consequência:* a hipótese do inst.ger.1 ("INT8 seguro, INT4 degrada fidelidade") **não
  se confirma neste protótipo** e precisa ser testada no modelo real (cache real, atenção
  real), não no brinquedo. O que sobrevive como regra metodológica é a **obrigação do
  controle**: sem os dois controles de falsificação, as duas métricas (v1 e v2) teriam
  passado por boas.

## 5. Correções de rota registradas (o processo funcionou)

1. **Controle A2 mal posto** (plano original): "embaralhar valores" é falha de **acurácia**,
   não de fidelidade — toda resposta segue sustentada pelo cache errado. Trocado por
   respostas-alheias + massa-difusa.
2. **Montagem do teste mal-condicionada** (1ª execução da v2): `L_faith = 1.0` até em FP16,
   porque a atenção era uniforme por construção. A métrica estava certa em acusar; o
   experimento é que não tinha evidência decisiva. Corrigido com chaves unitárias + ganho.
3. **Critério de "acerta sem fidelidade" mal definido**: `L_faith` absoluto confunde
   "tarefa mal-condicionada" com "dano de compressão". Corrigido para o **Δ entre
   precisões**.

## 6. O que isso muda no projeto

- A regra 3.4 continua válida, mas a **medição** de `L_faith` passa a exigir: (a) controle
  de falsificação, (b) comparação relativa entre precisões, (c) cache com condicionamento
  declarado. Proposta de redação da regra fica pendente de aprovação do Cássio (não alterei
  `regras_comparacao.md`).
- O protótipo de KV-cache **não** é evidência para decidir nada sobre a Celestia: precisa
  rodar no cache real do modelo (Fase 2 / contexto longo).
- Nada disso promove ou rebaixa versão (regra 1.3: número local é debug).
