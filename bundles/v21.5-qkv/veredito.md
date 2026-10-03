# veredito — v21.5-qkv (T4, 2 sessões bit-idênticas)

**Status: NÃO PROMOVE. Induction: sem efeito detectável (não "falsificada" —
ver correção). `lag`: sugestiva, não significativa (p≈0,22). Leitura conjunta:
INCONCLUSIVO nesta escala/prazo, com hipótese principal de SUBTREINO.**

## Números (12 runs, 300 steps, V=512, uniforme ln512=6.23832, 3 seeds)

- **induction:** fixo 6.30835/6.35494/6.30502 (média 6.32277, std populacional
  0.023) | qkv 6.37272/6.35047/6.30224 (média 6.34181, std 0.030).
  Ganho por seed (fixo−qkv): −0.06437/+0.00447/+0.00278 → média −0.01904,
  t pareado ≈ −0.84, p ≈ 0.49. A média negativa vem de UMA seed.
  Critério (qkv < 5.92641): não atingido por nenhum braço. **Sem efeito
  detectável** (ausência de evidência, não evidência de ausência).
- **lag:** fixo 5.88178/5.84866/5.86757 (média 5.86600) |
  qkv 5.49524/5.75237/5.80206 (média 5.68322) → ganho +0.18278.
  **Correção:** o próprio fixo (5.866) já passa o limiar 5.926 — o limiar não
  discrimina. Ganho por seed 0.387/0.096/0.066, t = 1.79, 2 g.l., p ≈ 0.22:
  compatível com ruído. **Rebaixado de "CONFIRMADA com ressalva" para
  inconclusivo.**
- Zero NaN/Inf. `clip_rate` = 1.00 do step 0 ao 299 **nos dois braços**
  (indicador de transiente, não evidência específica contra o QKV).
  `diag`: layer_scale_L0 = 0.13809.
- Warnings de `attn_*/layer_scale` sem gradiente: peso-morto da troca de blocos
  (código pré-gancho `attn_cls`); não afeta o forward. Já corrigido no repo.

## Reprodutibilidade entre sessões: BIT-EXATA

As duas sessões Colab (30/09 e 01/10, runtimes diferentes) produziram números
idênticos até a 5ª casa (ex.: fixo/induction/s0 6.30835 nas duas). O fix de seed
(`tf.keras.utils.set_random_seed`, Achado #3) está validado no T4. Piso de ruído
same-seed ≈ 0 para este harness.

## Por que o microscópio resolveu e o T4 não (hipótese de subtreino)

| | microscópio (resolveu) | T4 (não resolveu) |
|---|---|---|
| camadas/dim | 2L/64 | 12L/768 |
| vocabulário | 16 | 512 (32×) |
| batch | 8 | 2 |
| alvos vistos | 96.000 | 18.000 (5,3× menos) |
| LR | 1e-3 | 1e-4 |
| steps | 400 | 300 |
| params novos aleatórios (QKV) | ~25K | ~21M |

Vocabulário 32× maior, batch 4× menor e 5× menos alvos pesam tanto ou mais
que o LR na explicação. O circuito provavelmente nem saiu do transiente;
o mecanismo segue provado no microscópio (piso 1.2939). Falta orçamento
(ou desenho que forme o circuito com menos alvos), não evidência contra.

## Gate fechando: hipótese, não achado

`layer_scale` L0 0.1→0.087 no piloto (1 seed) contra 0.138 no run de 300 steps:
sinais opostos entre runs. Sem ablação (ex.: LS fixo) e sem repetição, "o gate
sufoca o circuito" é hipótese de trabalho — é exatamente o que a sonda barata
da opção (a) testaria. Removida a extrapolação "0,00017/passo" (comparava LRs
diferentes).

## Próximo teste (quando a quota GPU voltar)

`run_v21_5_qkv.py --models fixo qkv --tasks induction --steps 1000` com lr 3e-4
(6 runs ≈ 25 min): se qkv descolar do uniforme e fixo não, a hipótese original
está confirmada na escala real. Sem isso, v21.5 segue sonda, não candidata.
(Executado 02/10 — ver piloto abaixo. Não confirmou.)

*Proveniência: console Colab colado no chat (artefatos CSV ficaram no /tmp do
runtime; runtime posterior morreu com a quota). Números acima são transcrição
literal do bloco === VEREDITO === + linhas de run.*

## Piloto qkv-only, 1000 steps, lr 3e-4 (T4, 02/10) — PORTÃO FALHOU, matriz cancelada

- qkv/induction/s0: train 6.28028 → **loss_val 6.25580**, clip 1.00 (todo o
  trajeto), gn 5.9→3.2, layer_scale_L0 0.08739 (caiu de 0.1), grad_conn stone
  1e-3. Critério (val < 5.92641): **NÃO ATINGIDO** (6.25580 > 5.92641, e acima
  do próprio uniforme 6.23832).
- Progressão com orçamento: 300 steps → 6.37272; 1000 steps @3e-4 → 6.25580.
  Move na direção certa a ~0.00017/step — extrapolação linear exigiria ~2000+
  steps só para encostar no uniforme, sem garantia.
- **Sinal arquitetural novo e importante:** `layer_scale` 0.1 → 0.087. O
  otimizador está FECHANDO o portão da atenção, não abrindo. Dinâmica
  auto-reforçada plausível: caminho sem gradiente útil → gate fecha → menos
  gradiente → gate continua fechado. No microscópio (2 camadas) o circuito se
  forma antes do gate fechar; em 12 camadas, não.
- Veredito do piloto: **não justifica a matriz 3-seed completa**. v21.5-qkv
  segue sonda: mecanismo provado no microscópio (piso 1.2939), **não transfere
  em orçamento prático na escala real**. Arquivar como informação, não derrota.

## ARQUIVAMENTO FORMAL (decisão do Cássio, 03/10 — Porta 1)

**v21.5-qkv ARQUIVADA. Motivo escrito (revisado pós-revisão: INCONCLUSIVA nos
orçamentos testados, não refutada):**

1. Microscópio (2L/d64): QKV resolve induction (1.30733) e lag (0.00306) —
   mecanismo provado, no piso teórico.
2. Escala real, induction: 300 steps lr 1e-4 → qkv 6.34181 vs fixo 6.32277
   (ganho −0.019, p≈0.49 — sem efeito detectável); piloto 1000 steps lr 3e-4
   → 6.25580 (portão 5.93 não atingido).
3. Escala real, PT (A/B/C): 500 steps Δ=+0.016; 2000 steps Δ=**+0.065**,
   uniforme nas fatias cópia e geral (1 seed, sem piso — sem leitura de efeito).
4. Dinâmica em aberto (hipótese, não achado): `layer_scale` 0.1→0.087 no piloto
   contra 0.138 no run de 300 steps; gradientes do acoplamento 10–100× abaixo
   da pedra (P2).

Em 5 orçamentos, **nenhum sinal positivo do QKV — mas todos abaixo do regime
em que roteamento faria diferença** (modelos ainda entre unigrama e bigrama em
PT; induction T4 em transiente com clip 1.00). Leitura honesta: **inconclusivo,
não evidência contra**. Reabrir exige: (i) orçamento com convergência
verificada + portão de prontidão (regra 2.1: abaixo do piso de bigrama), ou
(ii) sonda barata do gate (layer_scale fixo — opção (a)).

**Pivot do projeto:** o que funciona é roteamento posicional (lag resolvido
pelo grafo fixo). A pergunta passa a ser o que o TANGO faz de único SEM
conteúdo — não mais como importar conteúdo para dentro dele. A sonda de
memória associativa (CEL-2.001) herda o mesmo risco de transferência e entra
na mesma fila de prova, sem privilégio.
