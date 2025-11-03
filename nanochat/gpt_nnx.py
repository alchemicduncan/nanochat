"""
GPT model (NNX version)
"""

import jax
import jax.numpy as jnp
import optax
from flax.experimental import nnx
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

class KVCache(nnx.Module):
    def __init__(self, config: GPTConfig):
        self.pos = nnx.Variable(0)
        self.kv: List[Tuple[nnx.Variable, nnx.Variable]] = [
            (nnx.Variable(jnp.zeros((1, config.n_kv_head, 0, config.n_embd // config.n_head), dtype=jnp.bfloat16)),
             nnx.Variable(jnp.zeros((1, config.n_kv_head, 0, config.n_embd // config.n_head), dtype=jnp.bfloat16)))
            for _ in range(config.n_layer)
        ]

    def get_pos(self):
        return self.pos.value

    def update_pos(self, T: int):
        self.pos.value += T

    def insert_kv(self, layer_idx: int, k: jax.Array, v: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pk, pv = self.kv[layer_idx]
        k = jnp.concatenate([pk.value, k], axis=2)
        v = jnp.concatenate([pv.value, v], axis=2)
        pk.value = k
        pv.value = v
        return k, v

def norm(x):
    return nnx.rms_norm(x, epsilon=1e-5)

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = jnp.concatenate([y1, y2], axis=3)
    return out.astype(x.dtype)

class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, bias=False, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, bias=False, rngs=rngs)

    def __call__(self, x, cos_sin, kv_cache: KVCache = None):
        B, T, C = x.shape

        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        y = jax.nn.scaled_dot_product_attention(q, k, v, is_causal=kv_cache is None)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.relu(x) ** 2
        x = self.c_proj(x)
        return x

class Block(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x, cos_sin, kv_cache: KVCache = None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.h = [Block(config, i, rngs=rngs) for i in range(config.n_layer)]
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, bias=False, rngs=rngs)
        
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = nnx.Param(cos)
        self.sin = nnx.Param(sin)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        cos, sin = jnp.cos(freqs), jnp.sin(freqs)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)

    def __call__(self, idx, targets=None, kv_cache: KVCache = None):
        B, T = idx.shape
        
        T0 = 0
        if kv_cache is not None:
            T0 = kv_cache.get_pos()
            kv_cache.update_pos(T)

        cos_sin = self.cos.value[:, T0:T0+T], self.sin.value[:, T0:T0+T]

        x = self.wte(idx)
        x = norm(x)
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        logits = self.lm_head(x)
        softcap = 15
        logits = softcap * jnp.tanh(logits / softcap)
        
        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return loss.mean()
        else:
            return logits
