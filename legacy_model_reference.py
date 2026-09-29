import numpy as np
import tensorflow as tf

VOCAB_SIZE = 32000
D_MODEL = 768
LAYERS = 12
SEQ_LEN = 128
ORTHO_REG_WEIGHT = 1e-4


class MyceliumTangoAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, eta=0.005, beta_start=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eta = eta
        self.beta = tf.Variable(beta_start, trainable=False, dtype=tf.float32, name="beta_temp")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _build_causal_small_world(self, n_nodes, k=6, p=0.1):
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for i in range(n_nodes):
            for j in range(1, k + 1):
                if i - j >= 0:
                    adj[i, i - j] = 1.0
        num_shortcuts = int(p * n_nodes * k)
        for _ in range(num_shortcuts):
            src = np.random.randint(1, n_nodes)
            dst = np.random.randint(0, src)
            adj[src, dst] = 1.0
        adj = adj + np.eye(n_nodes, dtype=np.float32)
        degree = np.sum(adj, axis=-1, keepdims=True)
        return adj / (degree + 1e-8)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.d_logic = int(dim * 0.95)
        self.d_facts = dim - self.d_logic

        self.w_stone = self.add_weight(
            shape=(self.d_logic, self.d_logic),
            initializer="orthogonal",
            trainable=True,
            name="stone"
        )

        self.A_norm = self.add_weight(
            shape=(SEQ_LEN, SEQ_LEN),
            initializer=tf.keras.initializers.Constant(
                self._build_causal_small_world(n_nodes=SEQ_LEN, k=6, p=0.1)
            ),
            trainable=False,
            name="A_norm"
        )

        self.gate = self.add_weight(shape=(dim,), initializer="ones", trainable=True, name="gate")
        super().build(input_shape)

    def project_orthogonal(self, x):
        logic_part = x[..., :self.d_logic]
        zeros = tf.zeros_like(x[..., self.d_logic:])
        return tf.concat([logic_part, zeros], axis=-1)

    def call(self, x, introspection_active=False, training=False):
        x_norm = self.norm(x)
        x_mixed = tf.einsum('ij,bjd->bid', self.A_norm, x_norm)

        if training:
            self.add_loss(ORTHO_REG_WEIGHT * self._orthogonality_penalty())

        if not introspection_active:
            h_logic = self.project_orthogonal(x_mixed)
            h_logic_active = h_logic[..., :self.d_logic]
            res = tf.matmul(h_logic_active, self.w_stone)
            h_out = tf.concat([res, x_mixed[..., self.d_logic:]], axis=-1)
            return h_out * tf.nn.sigmoid(self.gate)

        h = x_mixed
        u_facts = x_mixed[..., self.d_logic:]
        beta_t = self.beta
        eta_t = tf.cast(self.eta, dtype=h.dtype)

        for t in range(3):
            h_logic = self.project_orthogonal(h)
            h_logic_active = h_logic[..., :self.d_logic]
            f_h = tf.matmul(h_logic_active, self.w_stone)
            f_h_full = tf.concat([f_h, tf.zeros_like(h[..., self.d_logic:])], axis=-1)

            g_logic = h - f_h_full
            g_logic_facts = h[..., self.d_logic:] - u_facts

            dot_product = tf.reduce_sum(tf.math.multiply(u_facts, g_logic_facts), axis=-1, keepdims=True)
            g_norm_sq = tf.reduce_sum(tf.math.multiply(g_logic_facts, g_logic_facts), axis=-1, keepdims=True)

            g_norm_sq_32 = tf.cast(g_norm_sq, tf.float32)
            dot_product_32 = tf.cast(dot_product, tf.float32)
            g_norm_sq_safe = tf.maximum(g_norm_sq_32, 1e-4)
            ratio = tf.math.divide(dot_product_32, g_norm_sq_safe)
            ratio = tf.clip_by_value(ratio, -10.0, 10.0)
            ratio = tf.cast(ratio, h.dtype)
            u_tan = u_facts - tf.math.multiply(ratio, g_logic_facts)

            raw_noise = tf.random.normal(tf.shape(h), dtype=h.dtype)
            noise_scale = tf.cast(tf.math.sqrt(2.0 * self.eta / beta_t), dtype=h.dtype)

            h_logic_next = h[..., :self.d_logic] - tf.math.multiply(eta_t, g_logic[..., :self.d_logic]) \
                           + tf.math.multiply(raw_noise[..., :self.d_logic], noise_scale)
            h_facts_next = tf.math.multiply((1.0 - eta_t), h[..., self.d_logic:]) \
                           + tf.math.multiply(eta_t, u_tan)

            h = tf.concat([h_logic_next, h_facts_next], axis=-1)
            beta_t = beta_t * 1.05

        return h * tf.nn.sigmoid(self.gate)

    def _orthogonality_penalty(self):
        w_stone_32 = tf.cast(self.w_stone, tf.float32)
        wt_w = tf.matmul(w_stone_32, w_stone_32, transpose_a=True)
        identity = tf.eye(self.d_logic, dtype=tf.float32)
        return tf.reduce_mean(tf.square(wt_w - identity))


class PreLNFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.w1 = tf.keras.layers.Dense(d_ff, activation="swish", name="w1")
        self.w2 = tf.keras.layers.Dense(d_model, name="w2")

    def call(self, x):
        x_norm = self.norm(x)
        return self.w2(self.w1(x_norm))


class ConsciousModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, layers_count):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_embed = tf.keras.layers.Embedding(SEQ_LEN, d_model, name="pos_embed")
        self.blocks = []
        for i in range(layers_count):
            attn = MyceliumTangoAttention(d_model, name=f"attn_{i}")
            ffn = PreLNFFN(d_model, d_model * 4, name=f"ffn_{i}")
            setattr(self, f"attn_{i}", attn)
            setattr(self, f"ffn_{i}", ffn)
            self.blocks.append([attn, ffn])
        self.ln_final = tf.keras.layers.LayerNormalization(dtype='float32')
        self.head = tf.keras.layers.Dense(vocab_size, dtype='float32')

    def call(self, inputs, training=False, introspection_active=False):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)
        x = self.embed(inputs) + self.pos_embed(positions)
        for attn, ffn in self.blocks:
            x = x + attn(x, introspection_active=introspection_active, training=training)
            x = x + ffn(x)
        x = self.ln_final(x)
        return self.head(x)


class StatefulConsciousModel(ConsciousModel):
    def __init__(self, vocab_size, d_model, layers_count):
        super().__init__(vocab_size, d_model, layers_count)
        self._layer_outputs = []

    def call(self, inputs, training=False, introspection_active=False):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)
        x = self.embed(inputs) + self.pos_embed(positions)
        self._layer_outputs = []
        for attn, ffn in self.blocks:
            x = x + attn(x, introspection_active=introspection_active, training=training)
            x = x + ffn(x)
            self._layer_outputs.append(x)
        x = self.ln_final(x)
        return self.head(x)

    def get_states(self):
        return [tf.identity(s) for s in self._layer_outputs]

    def set_states(self, states):
        self._layer_outputs = [tf.identity(s) for s in states]
