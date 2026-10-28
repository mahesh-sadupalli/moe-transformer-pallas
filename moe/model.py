"""Decoder-only transformer in plain JAX: params are a pytree, the model is a pure function.

Pre-norm (RMSNorm), learned positional embeddings, tied input/output embedding.
The FFN is either dense (`ffn`) or Mixture-of-Experts (`moe_ffn`), set by `cfg.n_experts`.
"""

import math

import jax
import jax.numpy as jnp
import optax

from moe.config import ModelConfig


def init_params(cfg: ModelConfig, key):
    D, F, E = cfg.d_model, cfg.d_ff, cfg.n_experts
    keys = iter(jax.random.split(key, 2 + 5 * cfg.n_layers))

    def dense(fan_in, shape):
        return jax.random.normal(next(keys), shape) * fan_in**-0.5

    def ffn_params():
        if not E:
            return {"w_in": dense(D, (D, F)), "w_out": dense(F, (F, D))}
        return {"router": dense(D, (D, E)), "w_in": dense(D, (E, D, F)), "w_out": dense(F, (E, F, D))}

    return {
        "embed": jax.random.normal(next(keys), (cfg.vocab_size, D)) * 0.02,
        "pos": jax.random.normal(next(keys), (cfg.seq_len, D)) * 0.02,
        "layers": [
            {
                "attn_norm": jnp.ones(D),
                "wqkv": dense(D, (D, 3 * D)),
                "wo": dense(D, (D, D)),
                "ffn_norm": jnp.ones(D),
                "ffn": ffn_params(),
            }
            for _ in range(cfg.n_layers)
        ],
        "final_norm": jnp.ones(D),
    }


def param_count(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def active_param_count(params, cfg: ModelConfig) -> int:
    """Params touched per token: each token uses only top_k of n_experts experts."""
    total = param_count(params)
    if not cfg.n_experts:
        return total
    expert = sum(l["ffn"]["w_in"].size + l["ffn"]["w_out"].size for l in params["layers"])
    return total - expert * (cfg.n_experts - cfg.top_k) // cfg.n_experts


def rms_norm(x, w, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) * w


def attention(p, x, n_heads):
    B, T, D = x.shape
    q, k, v = jnp.split(x @ p["wqkv"], 3, axis=-1)
    q, k, v = (t.reshape(B, T, n_heads, D // n_heads) for t in (q, k, v))
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(D // n_heads)
    scores = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), scores, -jnp.inf)
    out = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), v)
    return out.reshape(B, T, D) @ p["wo"]


def ffn(p, x):
    return jax.nn.gelu(x @ p["w_in"]) @ p["w_out"]


def moe_ffn(p, x, cfg: ModelConfig):
    """Top-k MoE with einsum dispatch/combine (GShard / Switch style).

    x: (G, S, D). Each sequence is a routing group with its own expert capacity
    C = ceil(top_k * S * capacity_factor / E); assignments past capacity are dropped
    (the token then passes through on the residual only). First choices claim expert
    slots before any second choice, then earlier positions before later ones. So with
    top_k > 1 and dropping, a later token can evict an earlier token's 2nd choice: routing
    is only strictly causal when nothing is dropped.

    Returns (y, metrics) with load-balancing loss, router z-loss and dropped fraction.
    """
    G, S, D = x.shape
    E, k = cfg.n_experts, cfg.top_k
    C = math.ceil(k * S * cfg.capacity_factor / E)

    logits = x @ p["router"]  # (G, S, E)
    probs = jax.nn.softmax(logits, axis=-1)
    gate, idx = jax.lax.top_k(probs, k)  # (G, S, k); raw probs, so the router gets gradient even at k=1
    choice = jax.nn.one_hot(idx, E)  # (G, S, k, E)

    # Slot of each assignment in its expert's buffer, counted in (rank, position) order.
    ranked = choice.transpose(0, 2, 1, 3).reshape(G, k * S, E)
    slot = (jnp.cumsum(ranked, axis=1) - 1).reshape(G, k, S, E).transpose(0, 2, 1, 3)
    slot = jnp.sum(slot * choice, axis=-1).astype(jnp.int32)  # (G, S, k)
    keep = slot < C
    routed = choice * keep[..., None]
    slot_onehot = jax.nn.one_hot(slot, C)  # (G, S, k, C)

    dispatch = jnp.einsum("gske,gskc->gsec", routed, slot_onehot)
    combine = jnp.einsum("gsk,gske,gskc->gsec", gate, routed, slot_onehot)

    expert_in = jnp.einsum("gsec,gsd->egcd", dispatch, x)
    h = jax.nn.gelu(jnp.einsum("egcd,edf->egcf", expert_in, p["w_in"]))
    expert_out = jnp.einsum("egcf,efd->egcd", h, p["w_out"])
    y = jnp.einsum("gsec,egcd->gsd", combine, expert_out)

    load = choice.sum(axis=2).mean(axis=(0, 1)) / k  # fraction of assignments per expert, pre-drop
    metrics = {
        "aux": E * jnp.sum(load * probs.mean(axis=(0, 1))),  # 1.0 when balanced
        "z": jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2),
        "drop": 1.0 - keep.mean(),
        "load": load,  # (E,)
    }
    return y, metrics


def forward(params, tokens, cfg: ModelConfig):
    """tokens: (B, T) int32 -> (logits (B, T, vocab_size), MoE metrics).

    Scalar MoE metrics are averaged over layers; "load" is kept per layer, shape (L, E).
    """
    x = params["embed"][tokens] + params["pos"][: tokens.shape[1]]
    moe_metrics = []
    for layer in params["layers"]:
        x = x + attention(layer, rms_norm(x, layer["attn_norm"]), cfg.n_heads)
        h = rms_norm(x, layer["ffn_norm"])
        if cfg.n_experts:
            h, m = moe_ffn(layer["ffn"], h, cfg)
            moe_metrics.append(m)
        else:
            h = ffn(layer["ffn"], h)
        x = x + h
    metrics = {}
    if moe_metrics:
        stacked = jax.tree.map(lambda *v: jnp.stack(v), *moe_metrics)
        metrics = {k: v if k == "load" else v.mean() for k, v in stacked.items()}
    return rms_norm(x, params["final_norm"]) @ params["embed"].T, metrics


def loss_fn(params, x, y, cfg: ModelConfig):
    """Returns (total loss, metrics); metrics["ce"] is the pure language-modeling loss."""
    logits, moe = forward(params, x, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss = ce
    if moe:
        loss = ce + cfg.aux_loss_coef * moe["aux"] + cfg.z_loss_coef * moe["z"]
    return loss, {"ce": ce, **moe}
