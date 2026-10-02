# Piloto A/B em PT real (T4, 02/10) — 500 steps, 1 seed

**Status: pipeline validado de ponta a ponta; N=500 INSUFICIENTE (sem convergência).
N oficial ainda por calibrar.**

Corpus split=859d5e7e8ee1684a. Auditoria: N_A=88402700, N_B=109636364,
N_C=109650188 (ffn_mult_C=5.5; |N_C−N_B|<0.02%).

| braço | train | val | cópia | geral |
|---|---|---|---|---|
| A/s0 | 6.5938 | 7.01747 | 4.79110 | 7.84330 |
| B/s0 | 6.6343 | 7.03339 | 4.82330 | 7.85318 |

Δ(B−A) = +0.01592 (1 seed, sem piso — sem leitura de efeito).
Curvas quase idênticas (step 0/200/400/499 diferem ≤0.04) e ainda descendo
(6.95→6.59 nos últimos 100 steps): **500 steps não converge** → veredito
qualquer aqui seria em transiente (a guarda marcaria).

*Proveniência: transcrição do console (o CSV do A se perdeu no ^C; o do B
ficou em bundles/lm-abc-pilot-B no /tmp do runtime).*
