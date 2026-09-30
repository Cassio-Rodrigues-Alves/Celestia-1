"""model_qkv_probe — SONDA CONTROLADA: a Mycelium-LM v21 consegue fazer atenção por conteúdo?

## Por que este arquivo existe

A v21 (`model_v21.TangoV21._sparse_mix`) mistura posições por uma matriz de adjacência
**FIXA** (grafo causal small-world, não-treinável). Não há Q/K/V nem softmax em nenhum
arquivo do projeto (verificado por busca — 0 ocorrências). Isso significa que a
"atenção" da Mycelium-LM **não pode fazer roteamento por CONTEÚDO** (ex.: "copie o token
que seguiu a última ocorrência deste") — só mistura posicional fixa.

Resultado empírico correspondente: a tarefa-probe `induction` fica parada no uniforme
(loss_val ≈ ln(V)) tanto com layer_scale=0.0 quanto 0.1 (ver local_debug).

## O que esta sonda faz

Subclasse de `TangoV21` que troca **UMA** coisa: o mecanismo de mistura.
    fixo (grafo A_np)  ->  aprendido por conteúdo (Q,K,V + softmax causal)
Todo o resto (w_stone, w_hu/w_uh/w_uu, gate, layer_scale, beta, Langevin, FFN, head)
é idêntico. Se `induction` deixar de ficar no uniforme com a sonda, a causa da falha
está confirmada: falta roteamento por conteúdo.

## O que esta sonda NÃO é

- NÃO é uma versão promovida, NÃO é candidato a baseline, NÃO altera `model_v21.py`.
- É experimento de Fase 1 (prototipagem) — "1 variável por versão" respeitado.
- Cabeça única, por simplicidade de sonda.

Contribuição: DeepSeek (noite 2026-09-30).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from model_v21 import ConsciousV21, TangoV21


class TangoQKVProbe(TangoV21):
    """Mistura por conteúdo (Q/K/V causal, cabeça única) no lugar do grafo fixo."""

    def build(self, input_shape):
        super().build(input_shape)          # cria w_stone, w_hu, w_uh, w_uu, gate, layer_scale
        d = int(input_shape[-1])
        self._d_head = d
        self.wq = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="wq")
        self.wk = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="wk")
        self.wv = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="wv")

    def _sparse_mix(self, x_norm):
        q = tf.matmul(x_norm, self.wq)
        k = tf.matmul(x_norm, self.wk)
        v = tf.matmul(x_norm, self.wv)
        scale = tf.cast(tf.math.sqrt(tf.cast(self._d_head, tf.float32)), x_norm.dtype)
        scores = tf.matmul(q, k, transpose_b=True) / scale
        L = tf.shape(x_norm)[1]
        keep = tf.linalg.band_part(tf.ones((L, L), x_norm.dtype), -1, 0)
        scores = scores + (1.0 - keep) * (-1e9)
        att = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(att, v)


class ConsciousV21QKVProbe(ConsciousV21):
    """`ConsciousV21` com os blocos TANGO trocados pela sonda QKV (1 variável)."""

    def __init__(self, coupling=False, langevin=False, layer_scale_init=0.0):
        super().__init__(coupling=coupling, langevin=langevin, layer_scale_init=layer_scale_init)
        import model_v21 as M
        self.blocks = [
            (TangoQKVProbe(M.D_MODEL, coupling=coupling, langevin=langevin,
                           layer_scale_init=layer_scale_init, name=f"qkv_{i}"),
             f)
            for i, (_, f) in enumerate(self.blocks)
        ]
