"""model_v21 — Mycelium-LM v21.0-baseline limpa. Parte da legacy ref, aplica fixes obrigatórios."""
import numpy as np, tensorflow as tf
from legacy_model_reference import PreLNFFN  # noqa
VOCAB_SIZE, D_MODEL, LAYERS, SEQ_LEN = 32000, 768, 12, 128

class TangoV21(tf.keras.layers.Layer):
    def __init__(self, d_model, beta_start=0.15, eta=0.005, **kw):
        super().__init__(**kw)
        self.d_model, self.eta = d_model, eta
        self.beta = tf.Variable(beta_start, trainable=False, dtype=tf.float32)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # ponytail: layer_scale=0 init, 1 escalar por camada
        self.layer_scale = self.add_weight(shape=(), initializer="zeros", trainable=True, name="layer_scale")

    def _causal_sw_np(self, n=SEQ_LEN, k=6, p=0.1):
        adj = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(1, k+1):
                if i-j >= 0: adj[i, i-j] = 1.0
        for _ in range(int(p*n*k)):
            s, d = np.random.randint(1, n), np.random.randint(0, 1)
            adj[s, np.random.randint(0, s)] = 1.0
        adj += np.eye(n, dtype=np.float32)
        return (adj / (adj.sum(-1, keepdims=True) + 1e-8)).astype(np.float32)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.d_logic = int(dim*0.95); self.d_facts = dim - self.d_logic
        self.w_stone = self.add_weight(shape=(self.d_logic, self.d_logic), initializer="orthogonal", name="stone")
        # acoplamento assimétrico (não transposto) — desligado na baseline (frozen)
        self.w_hu = self.add_weight(shape=(self.d_logic, self.d_facts), initializer="zeros", trainable=False, name="w_hu")
        self.w_uh = self.add_weight(shape=(self.d_facts, self.d_logic), initializer="zeros", trainable=False, name="w_uh")
        self.w_uu = self.add_weight(shape=(self.d_facts, self.d_facts), initializer="zeros", trainable=False, name="w_uu")
        # fix scratch-graph: guarda numpy denso, fatia p/ L real no call
        self._A_np = self._causal_sw_np()
        self.gate = self.add_weight(shape=(dim,), initializer="ones", name="gate")
        super().build(input_shape)

    def _sparse_mix(self, x_norm):
        # x: [B,L,D], A: [128,128] -> fatia [:L,:L]
        L = tf.shape(x_norm)[1]
        A = tf.cast(tf.constant(self._A_np)[:L, :L], x_norm.dtype)
        return tf.einsum('ij,bjd->bid', A, x_norm)

    def call(self, x, introspection_active=False, training=False):
        x_norm = self.norm(x)
        x_mixed = self._sparse_mix(x_norm)
        if training:
            w32 = tf.cast(self.w_stone, tf.float32)
            self.add_loss(1e-4 * tf.reduce_mean(tf.square(tf.matmul(w32, w32, transpose_a=True) - tf.eye(self.d_logic, dtype=tf.float32))))
        # facts ancorado: u_tan == u_facts (idêntico, sem retreino)
        h_logic = tf.matmul(x_mixed[..., :self.d_logic], self.w_stone)
        h_out = tf.concat([h_logic, x_mixed[..., self.d_logic:]], -1)
        return self.layer_scale * h_out * tf.nn.sigmoid(self.gate)

    def telemetry(self):
        w = self.w_stone.numpy() if hasattr(self.w_stone, "numpy") else np.zeros((2, 2))
        try: eig = float(np.max(np.real(np.linalg.eigvals(w)))) if w.shape[0] < 800 else -1.0
        except Exception: eig = -1.0
        return {"nhu": float(np.linalg.norm(self.w_hu.numpy())), "nuh": float(np.linalg.norm(self.w_uh.numpy())), "eig_stone": eig}

class ConsciousV21(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = tf.keras.layers.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = [(TangoV21(D_MODEL, name=f"attn_{i}"), PreLNFFN(D_MODEL, D_MODEL*4, name=f"ffn_{i}")) for i in range(LAYERS)]
        self.ln = tf.keras.layers.LayerNormalization(dtype="float32")
        self.head = tf.keras.layers.Dense(VOCAB_SIZE, dtype="float32")
    def call(self, inp, training=False, introspection_active=False):
        x = self.embed(inp) + self.pos(tf.range(tf.shape(inp)[1]))
        for a, f in self.blocks:
            x = x + a(x, training=training) + f(x)
        return self.head(self.ln(x))
