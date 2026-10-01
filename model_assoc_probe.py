"""model_assoc_probe — SONDA CONTROLADA: memória associativa causal (custo linear) consegue
fazer o roteamento por CONTEÚDO que o grafo fixo da v21 não faz?

STATUS: PROPOSTA (arquivo novo). Não altera `model_v21.py`, não é candidato a baseline.

## Contexto

`model_qkv_probe.py` mostrou que trocar a mistura fixa (grafo A_np) por Q/K/V + softmax
causal tira `induction` do uniforme (2.780 -> 1.307) e resolve `lag` (1.513 -> 0.003).
Esta sonda faz a MESMA troca (só `_sparse_mix`), mas com o mecanismo da linha CEL-2.001
(Liquid Gated Attention / atenção linear com retenção): um estado associativo recorrente

    M_t = lambda_t * M_{t-1} + phi(k_t) v_t^T          z_t = lambda_t * z_{t-1} + 1
    y_t = phi(q_t)^T M_t / z_t                         phi(x) = elu(x) + 1

com `lambda_t = sigmoid(w . x_t + b)` aprendido por conteúdo. Normalização por `z_t` e
LayerNorm na saída são obrigatórias: sem elas, no protótipo CEL-2.001 a norma do ramo de
memória ficou ~680x (e depois ~90x) acima do ramo estrutural em init aleatório.

## Forma paralela (CPU-friendly, sem tf.scan)

A recorrência acima é equivalente a (c_t = cumsum(log lambda)):

    y_t = sum_{i<=t} exp(c_t - c_i) * <phi(q_t), phi(k_i)> * v_i  /  sum_{i<=t} exp(c_t - c_i)

Custo O(L^2) em memória por [B,L,L] (e não [B,L,d,d]), idêntico ao da atenção softmax
nesta escala. O ganho de custo LINEAR só existe na forma recorrente (inferência);
esta sonda mede CAPACIDADE, não custo. `assoc_mix_scan` é a forma recorrente de
referência; `selftest()` compara as duas numericamente.

## Dois modos (1 variável cada, em relação à v21)

- `replace`: a memória SUBSTITUI o grafo fixo (comparável 1:1 com a sonda Q/K/V).
- `hybrid` : alpha_t * grafo_fixo + (1 - alpha_t) * memória, alpha_t = sigmoid(w_a . x_t + b_a)
             por posição (init 0.5). Preserva o roteamento posicional (que resolve `lag`)
             e acrescenta o roteamento por conteúdo — é o desenho da CEL-2.001.

## O que esta sonda NÃO é

- Não é versão promovida nem altera `model_v21.py` ("1 variável por versão" respeitado).
- Cabeça única, 1 estado por camada. Resultado local de CPU é DEBUG (regra 1.3).

Contribuição: Claude (proposta CEL-2.001, a partir da linha de pesquisa Liquid Gated
Attention + híbrido Tango/memória dos relatórios diários).
Fix: delattr dos blocos `attn_*` substituídos (pesos-mortos geravam warning de
gradiente inexistente) — aplicação local.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from model_v21 import ConsciousV21, TangoV21

EPS = 1e-6


def _phi(x):
    """Feature map positivo (atenção linear): phi(x) = elu(x) + 1 > 0."""
    return tf.nn.elu(x) + 1.0


class TangoAssocProbe(TangoV21):
    """Mistura por memória associativa causal no lugar do grafo fixo (modo `replace`)
    ou combinada a ele por gate por posição (modo `hybrid`)."""

    def __init__(self, d_model, mode="replace", d_mem=None, lambda_bias=2.0, **kw):
        super().__init__(d_model, **kw)
        if mode not in ("replace", "hybrid"):
            raise ValueError(f"mode deve ser 'replace' ou 'hybrid', veio {mode!r}")
        self.mode = mode
        self._d_mem = d_mem
        self._lambda_bias = float(lambda_bias)
        # criada no __init__ (Keras 3 não aceita novo estado após o build)
        self.mem_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.last_alpha = None

    def build(self, input_shape):
        super().build(input_shape)  # w_stone, w_hu/w_uh/w_uu, gate, layer_scale, _A_np
        d = int(input_shape[-1])
        dm = int(self._d_mem or d)
        self._dm = dm
        self.wq = self.add_weight(shape=(d, dm), initializer="glorot_uniform", name="wq")
        self.wk = self.add_weight(shape=(d, dm), initializer="glorot_uniform", name="wk")
        self.wv = self.add_weight(shape=(d, dm), initializer="glorot_uniform", name="wv")
        # retenção lambda_t = sigmoid(x w + b); b=2 -> ~0.88 ("começa lembrando")
        self.w_lam = self.add_weight(shape=(d, 1), initializer="zeros", name="w_lam")
        self.b_lam = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(self._lambda_bias),
                                     name="b_lam")
        if dm != d:
            self.w_out = self.add_weight(shape=(dm, d), initializer="glorot_uniform", name="w_out")
        else:
            self.w_out = None
        if self.mode == "hybrid":
            self.w_alpha = self.add_weight(shape=(d, 1), initializer="zeros", name="w_alpha")
            self.b_alpha = self.add_weight(shape=(1,), initializer="zeros", name="b_alpha")

    # ---- projeções comuns às duas formas -------------------------------------------------
    def _proj(self, x_norm):
        q = _phi(tf.matmul(x_norm, self.wq))
        k = _phi(tf.matmul(x_norm, self.wk))
        v = tf.matmul(x_norm, self.wv)
        lam = tf.nn.sigmoid(tf.matmul(x_norm, self.w_lam) + self.b_lam)  # [B,L,1]
        return q, k, v, lam

    def _finish(self, y):
        if self.w_out is not None:
            y = tf.matmul(y, self.w_out)
        return self.mem_norm(y)

    # ---- forma paralela (usada no treino) -------------------------------------------------
    def assoc_mix(self, x_norm):
        q, k, v, lam = self._proj(x_norm)
        logl = tf.math.log(lam + EPS)[..., 0]            # [B,L]   (<= 0)
        c = tf.cumsum(logl, axis=1)                       # c_t = sum_{j<=t} log lambda_j
        diff = c[:, :, None] - c[:, None, :]              # [B,t,i] = sum_{j=i+1..t} log lambda_j
        L = tf.shape(x_norm)[1]
        keep = tf.cast(tf.linalg.band_part(tf.ones((L, L)), -1, 0), x_norm.dtype)  # i <= t
        decay = tf.exp(tf.minimum(diff, 0.0)) * keep      # min() evita overflow nas posições mascaradas
        sim = tf.matmul(q, k, transpose_b=True)           # <phi(q_t), phi(k_i)>
        num = tf.matmul(decay * sim, v)                   # [B,t,dm]
        z = tf.reduce_sum(decay, axis=-1, keepdims=True)  # z_t
        return self._finish(num / (z + EPS))

    # ---- forma recorrente de referência (inferência / teste de equivalência) ---------------
    def assoc_mix_scan(self, x_norm):
        q, k, v, lam = self._proj(x_norm)
        B = tf.shape(x_norm)[0]
        L = x_norm.shape[1]
        M = tf.zeros((B, self._dm, self._dm), x_norm.dtype)
        z = tf.zeros((B, 1), x_norm.dtype)
        ys = []
        for t in range(L):
            lam_t = lam[:, t, :]                                      # [B,1]
            kv = tf.einsum('bd,be->bde', k[:, t, :], v[:, t, :])
            M = lam_t[:, :, None] * M + kv
            z = lam_t * z + 1.0
            y_t = tf.einsum('bd,bde->be', q[:, t, :], M) / (z + EPS)
            ys.append(y_t)
        return self._finish(tf.stack(ys, axis=1))

    def _sparse_mix(self, x_norm):  # mesmo "encaixe" do TangoQKVProbe
        mem = self.assoc_mix(x_norm)
        if self.mode == "replace":
            return mem
        struct = super()._sparse_mix(x_norm)  # grafo fixo original
        alpha = tf.nn.sigmoid(tf.matmul(x_norm, self.w_alpha) + self.b_alpha)  # [B,L,1]
        if tf.executing_eagerly():
            self.last_alpha = float(tf.reduce_mean(alpha).numpy())
        return alpha * struct + (1.0 - alpha) * mem


class ConsciousV21AssocProbe(ConsciousV21):
    """`ConsciousV21` com os blocos TANGO trocados pela sonda de memória associativa."""

    def __init__(self, coupling=False, langevin=False, layer_scale_init=0.0,
                 mode="replace", d_mem=None):
        super().__init__(coupling=coupling, langevin=langevin, layer_scale_init=layer_scale_init)
        import model_v21 as M
        n = len(self.blocks)
        self.blocks = [
            (TangoAssocProbe(M.D_MODEL, mode=mode, d_mem=d_mem, coupling=coupling,
                             langevin=langevin, layer_scale_init=layer_scale_init,
                             name=f"assoc_{i}"),
             f)
            for i, (_, f) in enumerate(self.blocks)
        ]
        for i in range(n):
            old = getattr(self, f"attn_{i}", None)
            if old is not None:
                old.trainable = False  # peso-morto (só layer_scale existe): fora do otimizador
            try:
                delattr(self, f"attn_{i}")  # blocos substituídos: sem acesso fantasma
            except AttributeError:
                pass


def selftest(d_model=64, L=32, B=3, seed=0, tol=1e-4):
    """Forma paralela == recorrência (scan). Levanta AssertionError se divergir."""
    tf.keras.utils.set_random_seed(seed)
    layer = TangoAssocProbe(d_model, mode="replace", name="selftest")
    x = tf.random.normal((B, L, d_model))
    layer(x)  # build
    # estressa lambda: pesos aleatórios em w_lam tornam lambda variável por posição
    layer.w_lam.assign(tf.random.normal(layer.w_lam.shape, stddev=0.5))
    xn = layer.norm(x)
    a = layer.assoc_mix(xn).numpy()
    b = layer.assoc_mix_scan(xn).numpy()
    err = float(np.max(np.abs(a - b)))
    assert err < tol, f"paralelo != scan: max|diff|={err:.3e} (tol {tol})"
    # causalidade: alterar o futuro não pode mudar o passado
    x2 = x.numpy().copy()
    x2[:, L // 2:, :] += 5.0
    a2 = layer.assoc_mix(layer.norm(tf.constant(x2))).numpy()
    leak = float(np.max(np.abs(a[:, :L // 2] - a2[:, :L // 2])))
    assert leak < tol, f"vazamento causal: max|diff passado|={leak:.3e}"
    return {"max_abs_diff_parallel_vs_scan": err, "max_abs_diff_causal_leak": leak}


if __name__ == "__main__":
    import model_v21
    model_v21.D_MODEL = 64
    print("selftest:", selftest())
