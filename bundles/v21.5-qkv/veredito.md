# veredito — v21.5-qkv (T4, 2 sessões bit-idênticas)

**Status: NÃO PROMOVE. Critério pré-registrado da induction: FALSIFICADO nesta
escala/prazo. `lag` deu sinal positivo (+0.18) mas ruidoso. Hipótese principal
para a divergência micro-vs-T4: SUBTREINO, não refutação do mecanismo.**

## Números (12 runs, 300 steps, V=512, uniforme ln512=6.23832, 3 seeds)

- **induction:** fixo 6.30835/6.35494/6.30502 (média 6.32277, std 0.017) |
  qkv 6.37272/6.35047/6.30224 (média 6.34181, std 0.030) → ganho −0.01904.
  Critério (qkv < 0.95×uniforme = 5.92641): **FALSIFICADA**. Ambas ≈ uniforme.
- **lag:** fixo 5.88178/5.84866/5.86757 (média 5.86600, std 0.014) |
  qkv 5.49524/5.75237/5.80206 (média 5.68322, std 0.134) → ganho +0.18278.
  Critério: **CONFIRMADA** pelo limiar, com ressalva de variância (std do qkv
  0.134 ≈ ordem do ganho; s0 destoa para melhor). Efeito sugestivo, não sólido.
- Zero NaN/Inf. `clip_rate` induction = 1.00 do step 0 ao 299, gn ~5 estável
  (regime transiente — nada convergiu). `diag`: layer_scale_L0 = 0.13809.
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
| LR | 1e-3 | 1e-4 |
| steps | 400 | 300 |
| params novos aleatórios (QKV) | ~25K | ~21M |

Projeções QKV 768×768×3/camada nascidas do zero, lr 10× menor, 300 passos,
batch 2: o circuito de indução provavelmente nem saiu do transiente
(clip 1.00 o tempo todo = evidência). O mecanismo está provado no microscópio
(no piso 1.2939); falta orçamento de treino na escala real, não evidência contra.

## Próximo teste (quando a quota GPU voltar)

`run_v21_5_qkv.py --models fixo qkv --tasks induction --steps 1000` com lr 3e-4
(6 runs ≈ 25 min): se qkv descolar do uniforme e fixo não, a hipótese original
está confirmada na escala real. Sem isso, v21.5 segue sonda, não candidata.

*Proveniência: console Colab colado no chat (artefatos CSV ficaram no /tmp do
runtime; runtime posterior morreu com a quota). Números acima são transcrição
literal do bloco === VEREDITO === + linhas de run.*
