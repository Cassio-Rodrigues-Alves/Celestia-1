# CORREÇÃO ao veredito do kvfaith v2 (revisão independente do CSV)

O commit que a validou diz "métrica validada com controles de falsificação".
Os controles passam (answers_alheias e massa_difusa acusam 1.0 corretamente),
**mas o critério A1 falha**: `v2_scan_gain.csv` mostra `l_faith = 1.0` já em
FP16 em todos os ganhos 0.25–4.0 (pmax ≤ 0.37, abaixo do corte 0.5). Ou seja,
fora do regime de atenção concentrada (ganho ≈ 10, pmax 0.99), a métrica acusa
infidelidade até da referência sem compressão — ela confunde "atenção difusa"
com "infidelidade por quantização".

Leitura corrigida:
- Regime válido da v2: só atenção concentrada (ganho ≈ 10). Nele, INT4 não
  degrada — mas a margem é tão grande que o teste nem precisava existir.
- A faixa onde a degradação moraria (ganhos 4–10) **não foi varrida**.
- "Δ = 0 em todos os regimes" é mais fraco do que parece; o que se sustenta é
  a refutação do número da v1 (−19,8 pp sob atenção quase uniforme por
  construção — artefato do setup, não efeito de quantização).
- Falta: varredura 4–10 + definição que separe nitidez da atenção de fidelidade
  ao cache (ex.: normalizar pela nitidez da referência FP16 no mesmo ganho).

*Correção por Muse Spark, 03/10, a partir de `v2_scan_gain.csv` — sem alterar
os arquivos do autor.*
