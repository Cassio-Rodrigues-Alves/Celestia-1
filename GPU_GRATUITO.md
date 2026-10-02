# GPU gratuita na nuvem — opções reais em outubro/2026 (para o projeto Celestia)

Levantamento feito em 02/10/2026 (fontes na web; números mudam com o tempo — confira no
próprio painel do serviço antes de contar com eles).

## Comparativo

- **Kaggle Notebooks** — ~30 h/semana de GPU (T4 ×2, 30 GB VRAM somados, ou P100), sessão
  de até 12 h, **sem cartão de crédito**, 73 GB de storage persistente. Exige verificação
  por telefone para liberar acelerador. Pegadinha: com GPU ligada, a internet do notebook
  vem desligada por padrão (instale as dependências antes ou ligue o acesso nas Settings).
- **Google Colab (grátis)** — T4 16 GB, sessão de até 12 h, cota semanal **não publicada**
  (varia com demanda), sem cartão. É o que o projeto já usa. Pegadinha: pode cair para CPU
  sem aviso e desconecta se ficar ocioso.
- **Lightning AI Studios** — **80 h de GPU/mês grátis** (T4/L4/A10G), ambiente persistente
  estilo VS Code, com **CLI e SSH**. Exige verificação por telefone, sem cartão.
  É a melhor opção para rodadas longas sem recomeçar ambiente a cada sessão.
- **Paperspace Gradient (free)** — M4000 8 GB, sessão de 6–12 h, 5 GB de storage. Vários
  relatos de fila longa; uma das fontes diz exigir cartão. Menos atrativo.
- **Hugging Face ZeroGPU** — 5 min/dia para conta grátis (40 min no PRO). Serve para
  *hospedar demo*, não para treinar.
- **Amazon SageMaker Studio Lab** — 4 h GPU/dia, mas **fechado para novas contas desde
  30/07/2026**. Só quem já tem.
- **Modal / NSF ACCESS** — para pesquisa acadêmica (grant); não é self-service.

## Recomendação para a Celestia (concreta)

1. **Adicionar o Kaggle ao lado do Colab.** Ganho imediato: +30 h/semana garantidas, sem
   cartão, com dois T4. O `run_v21_4_harness.py` roda igual (o requisito é TensorFlow, que
   o Kaggle já traz instalado).
2. **O ponto que muda o jogo para o seu caso: o Kaggle tem API/CLI.**
   `kaggle notebooks push` + `kaggle kernels output` permitem **empurrar um notebook e
   executá-lo em segundo plano sem interação**, com "Save & Run All". Ou seja: com um
   token de API configurado **uma vez**, uma rodada de GPU deixa de exigir você logado e
   clicando — o que é exatamente o bloqueio que travou a noite de 30/09.
   Isso precisa da sua autorização explícita (é a sua conta), mas depois disso os
   experimentos passam a ser executáveis de forma autônoma.
3. **Lightning AI como segunda opção** para rodadas longas (80 h/mês, ambiente persistente,
   SSH/CLI). A verificação por telefone uma vez resolve.

## O que continua proibido sem sua autorização
- Criar conta, aceitar termos, fazer verificação por telefone, gerar token de API ou
  logar em qualquer serviço em seu nome. Nenhum desses passos foi feito.
