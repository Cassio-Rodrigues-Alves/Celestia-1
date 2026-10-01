"""tasks_v21 — harness de avaliação DISCRIMINANTE (v21.4).

## Diagnóstico que motiva este módulo

O harness antigo (`run_v21_*.py`) faz:
    xb = tf.random.uniform((2, 32), 0, 512)   # UM lote fixo
    for s in range(500): ... treina no MESMO xb ...
Isso mede **capacidade de memorização de 64 tokens fixos**. O chão de loss (0.003)
é o chão da memorização. Nesse chão, atenção viva vs morta empatam -> Δ=0 -> nenhuma
arquitetura se distingue (foi o que matou a promoção da v21.3).

## O que este harness faz

Gera dados NOVOS a cada passo (treino) + um conjunto de validação CONGELADO.
Loss de validação não pode ser levada a zero por memorização -> mede generalização.
Assim o teste volta a ter poder de discriminação sobre o mecanismo de atenção.

Tarefas:
  - `mem`       : controle — reproduz o harness antigo (lote fixo). Deve saturar.
  - `induction` : cópia com atraso ("induction heads"). Sequência = [t, t]; prever a
                  segunda cópia exige UM CABEÇOTE DE INDUÇÃO (atenção). Sem atenção: impossível.
  - `copy`      : t+1 = t (identidade). Exige atenção mínima; bom gradiente de dificuldade.
  - `assoc`     : pares chave->valor no contexto; ao final, prever o valor. Exige
                  recuperação por conteúdo (atenção por chave).

Regra 1.1 preservada: `SEQ_LEN`, `VOCAB`, split e seeds vêm de fora; o gerador é
determinístico por seed -> mesma tarefa em toda comparação.

Contribuição: DeepSeek (v21.4 — noite 2026-09-30).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

TASKS = ("mem", "copy", "induction", "assoc", "lag")
# NOTA DE ALINHAMENTO (armadilha documentada 30/09, análise Claude CEL-2.001):
# `train_run` usa loss(yb[:,1:], logits[:,:-1]) e make_task devolve (seq[:-1], seq[1:]).
# Efeito líquido: alvo(p) = seq[p+2]. Toda tarefa nova TEM que ser desenhada contra
# esse deslocamento — `copy` ingênuo vira "adivinhe 2 à frente" (impossível) e `assoc`
# sem respostas após as consultas nunca testa recuperação (BUG pré-fix, ver VOID).
# Tarefas que a Mycelium-LM v21 CONSEGUE aprender (mistura posicional fixa basta):
HARNESS_TASKS = ("mem", "lag")
# Sondas de CAPACIDADE: exigem atenção por CONTEÚDO (Q/K/V). A v21 não tem Q/K/V
# (a adjacência é um grafo fixo) -> devem falhar. Falhar aqui é o resultado esperado,
# e mede exatamente a lacuna estrutural (ver NOITE_2026-09-30.md).
PROBE_TASKS = ("induction", "assoc", "copy")


def _rng(seed):
    return np.random.default_rng(seed)


def make_task(name, V, L, B, seed):
    """Um lote (x, y) da tarefa. Para 'mem' o lote é FIXO por seed (controle do antigo)."""
    rng = _rng(seed)
    if name == "mem":
        x = rng.integers(0, V, size=(B, L))
        return tf.constant(x, tf.int32), tf.constant(x, tf.int32)

    if name == "copy":
        # CORRIGIDA 30/09: identidade sob o deslocamento do harness.
        # yb = [sentinela, x[0..]] -> alvo(p) = yb[p+1] = x[p]. Sanity probe:
        # qualquer modelo funcional copia; se falhar, o modelo está quebrado.
        # (Versão antiga pedia seq[p+2] de seq i.i.d. = impossível até p/ QKV.)
        x = rng.integers(0, V, size=(B, L - 1))
        yb = np.zeros((B, L - 1), dtype=np.int64)
        yb[:, 1:] = x[:, :-1]
        return tf.constant(x, tf.int32), tf.constant(yb, tf.int32)

    if name == "induction":
        K = max(2, L // 2)
        t = rng.integers(0, V, size=(B, K))
        seq = np.concatenate([t, t], axis=1)[:, :L]          # [t, t] truncado a L
        return tf.constant(seq[:, :-1], tf.int32), tf.constant(seq[:, 1:], tf.int32)

    if name == "lag":
        # y_t = s_{t-j} (eco com atraso fixo j). Exige ROTEAMENTO POSICIONAL: o mix de
        # grafo entrega s_{t-j} na posição t. Sem mistura (layer_scale=0) é impossível,
        # porque o FFN é position-wise e o residual mantém s_t. Não é memorizável
        # (lote novo a cada passo). Posições t<j recebem alvo-sentinela 0 (posição conhecida
        # via position embedding) -> previsível, não infla o sinal.
        j = max(1, L // 8)
        s = rng.integers(1, V, size=(B, L))        # 1..V-1 (0 reservado p/ sentinela)
        y = np.zeros_like(s)
        y[:, j:] = s[:, :-j]
        return tf.constant(s[:, :-1], tf.int32), tf.constant(y[:, 1:], tf.int32)

    if name == "assoc":
        # CORRIGIDA 30/09: seq = [k1 v1 ... kK vK | q1..qK | a1..aK], a_j = v_j.
        # A resposta vem DEPOIS da consulta, então alvo(p)=seq[p+2] para as respostas
        # exige ter visto q_j — só roteamento por CONTEÚDO resolve. Piso p/ modelo
        # com conteúdo: 14/30*lnV (chaves/valores de primeira ocorrência).
        # (Versão antiga terminava nas consultas: valores imprevisíveis sempre e
        # chaves copiáveis por offset fixo variável — media cópia, não associação.)
        K = max(2, L // 4)
        keys = rng.integers(0, V, size=(B, K))
        vals = rng.integers(0, V, size=(B, K))
        kv = np.stack([keys, vals], axis=-1).reshape(B, 2 * K)   # k1 v1 k2 v2 ...
        seq = np.concatenate([kv, keys, vals], axis=1)           # ... q1..qK a1..aK
        if seq.shape[1] > L:
            seq = seq[:, :L]
        elif seq.shape[1] < L:
            seq = np.concatenate([seq, np.zeros((B, L - seq.shape[1]), dtype=seq.dtype)], axis=1)
        return tf.constant(seq[:, :-1], tf.int32), tf.constant(seq[:, 1:], tf.int32)

    raise ValueError(f"tarefa desconhecida: {name!r} (use uma de {TASKS})")


def make_env(name, V, L, B, seed, n_val=8):
    """Ambiente da tarefa: gerador de treino (novo a cada passo) + validação congelada.

    Retorna (batch_fn, val_x, val_y, meta). ``batch_fn(step)`` devolve um lote novo.
    Para 'mem' o lote NÃO varia (é o controle) — comparabilidade com o harness antigo.
    """
    if name not in TASKS:
        raise ValueError(f"tarefa desconhecida: {name!r} (use uma de {TASKS})")

    if name == "mem":
        x, y = make_task(name, V, L, B, seed)

        def batch_fn(step, _x=x, _y=y):
            return _x, _y
    else:
        def batch_fn(step, _name=name, _V=V, _L=L, _B=B, _seed=seed):
            return make_task(_name, _V, _L, _B, _seed * 100003 + step)

    # validação congelada (seeds bem separadas do treino)
    vx = [make_task(name, V, L, B, seed * 7919 + i + 500000) for i in range(n_val)]
    val_x = tf.concat([t[0] for t in vx], axis=0)
    val_y = tf.concat([t[1] for t in vx], axis=0)
    meta = {"task": name, "V": V, "L": L, "B": B, "seed": seed, "n_val_batches": n_val}
    return batch_fn, val_x, val_y, meta


def eval_loss(model, val_x, val_y, V, loss_fn):
    """Loss de validação (o slice [:V] espelha o harness antigo: logits truncados ao vocab da tarefa)."""
    logits = model(val_x, training=False)
    return float(loss_fn(val_y[:, 1:], logits[:, :-1, :V]))


def saturated(loss_val, tol=1e-2):
    """Heurística de saturação, **válida apenas para a tarefa-controle `mem`**.

    Em `mem` os dados são o MESMO lote fixo em todos os passos: loss_val baixa ali é
    memorização, não aprendizado. Já em tarefas com dados novos a cada passo
    (`lag`, `copy`, `induction`, `assoc`) a validação é um conjunto CONGELADO e
    out-of-batch: loss_val baixa é **tarefa resolvida**, não saturação. Usar este
    atalho fora de `mem` produz conclusão invertida.
    """
    return float(loss_val) < tol
