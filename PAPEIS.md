# Papéis e portão de promoção (decisão do Cássio, 02/10)

- **Principal (implementa):** Muse Spark (papel Claude: arquitetura, código,
  debugging, organização, vereditos).
- **Revisor (audita):** DeepSeek/Hermes. **Uso gratuito minúsculo** → revisão
  SÓ em portão de versão, em pacote compacto (diff + números + perguntas
  fechadas), nunca aberta/contínua. Sem pacote, sem revisão.
- **Outros (Claude/CEL-2.001 etc.):** propostas em arquivo separado, entram na
  fila como qualquer experimento.

## Portão de promoção (toda próxima versão)

1. Principal termina a versão + roda a comparação contra a **última aprovada
   (hoje: v21.1-coupled)** no MESMO harness, MESMAS seeds, MESMO orçamento.
2. Veredito preliminar pelo critério pré-registrado (regra 2.1 + piso honesto
   cross-seed/cross-init — nunca 0.0).
3. Pacote de revisão p/ o DeepSeek: diff da versão, tabela de números,
   veredito preliminar, 1–3 perguntas fechadas.
4. Promove SÓ com: critério atendido + revisão sem objeção bloqueadora.
   Objeção bloqueadora = volta pra fila como item novo, não como discussão.
