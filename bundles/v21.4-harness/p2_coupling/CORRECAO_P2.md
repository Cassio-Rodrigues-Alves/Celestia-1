# CORREÇÃO ao veredito da re-medição P2 (piso 0.0)

O `p2_coupling.json` atual diz Δ=−0.00324 com piso 0.0 → "DIFERENÇA ACIMA DO PISO".
Essa leitura está errada, e o erro é instrutivo:

1. **Piso 0.0 é o piso de re-run da MESMA seed** (determinismo bit-exato — ótimo,
   valida o fix). Mas a comparação é entre braços com **inicializações diferentes**:
   `coupling=True` cria pesos extras, que deslocam o stream do RNG — os pesos
   compartilhados (stone, embeddings) nascem diferentes entre os braços mesmo na
   "mesma seed". Somado a n=2 seeds, a variância cross-init desta config é
   desconhecida e certamente > 0.
2. **Evidência externa do próprio repo:** a medição pré-fix (mesma tarefa, 400
   steps) deu Δ=−0.00796 — mesma ordem de grandeza e mesmo sinal, com seeds e
   inits igualmente não-emparelhados. Duas medições independentes em ±0.003–0.008
   com n≤2 e sem variância medida = ruído, não efeito.
3. **Conclusão corrigida: INDISTINGUÍVEL neste orçamento** (acordo com a medição
   anterior). O acoplamento continua sem evidência de efeito no `lag`.

Regra geral (proposta para regra 2.1): piso honesto = spread **cross-seed E
cross-init da mesma config**, nunca 0.0 por determinismo. Determinismo elimina
o ruído de re-run; não elimina a loteria de inicialização entre braços.

*Correção por Muse Spark, 02/10 — sem alterar o arquivo do DeepSeek.*
