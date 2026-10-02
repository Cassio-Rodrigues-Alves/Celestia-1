# P2 — medição anterior (sessão pré-fix de seed, 400 steps, 1 seed) — ARQUIVO

Preservado porque `p2_coupling.json` foi sobrescrito pela re-medição com seed
determinística. Nenhum dos dois é "o certo"; são orçamentos diferentes.

- sem acoplamento: loss_val = 1.50972 (train 1.54212, clip 0.89)
- com acoplamento: loss_val = 1.50177 (train 1.53417, clip 0.90)
- Δ(com−sem) = −0.00796
- Piso usado então: 0.027 (spread run-to-run pré-fix) → veredito: indistinguível.

Limitação conhecida então: 1 seed só + inits não controlados entre braços.
Ver CORRECAO_P2.md para a leitura conjunta das duas medições.
