import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from moe.config import PRESETS
from moe.model import active_param_count, ffn, forward, init_params, loss_fn, moe_ffn, param_count

TINY_MOE = PRESETS["tiny_moe"]


def _x(cfg, batch=4, key=1):
    return jax.random.normal(jax.random.key(key), (batch, cfg.seq_len, cfg.d_model))


def test_single_expert_matches_dense_ffn():
    cfg = dataclasses.replace(TINY_MOE, n_experts=1, top_k=1, capacity_factor=1.0)
    k1, k2 = jax.random.split(jax.random.key(0))
    dense = {"w_in": jax.random.normal(k1, (cfg.d_model, cfg.d_ff)),
             "w_out": jax.random.normal(k2, (cfg.d_ff, cfg.d_model))}
    moe = {"router": jnp.zeros((cfg.d_model, 1)), "w_in": dense["w_in"][None], "w_out": dense["w_out"][None]}
    x = _x(cfg)
    y, m = moe_ffn(moe, x, cfg)
    np.testing.assert_allclose(y, ffn(dense, x), rtol=1e-5, atol=1e-5)
    assert m["drop"] == 0.0


def test_router_gets_gradient_from_lm_loss():
    """Even with the aux/z losses switched off, the gates carry gradient to the router."""
    for top_k in (1, 2):
        cfg = dataclasses.replace(TINY_MOE, top_k=top_k, aux_loss_coef=0.0, z_loss_coef=0.0)
        params = init_params(cfg, jax.random.key(0))
        x = jax.random.randint(jax.random.key(1), (4, cfg.seq_len + 1), 0, cfg.vocab_size)
        grads = jax.grad(lambda p: loss_fn(p, x[:, :-1], x[:, 1:], cfg)[0])(params)
        for layer in grads["layers"]:
            assert jnp.abs(layer["ffn"]["router"]).sum() > 0


def test_aux_loss_is_one_when_balanced_and_e_when_collapsed():
    cfg = dataclasses.replace(TINY_MOE, top_k=1)
    E, x = cfg.n_experts, _x(cfg)
    params = init_params(cfg, jax.random.key(0))["layers"][0]["ffn"]

    uniform = {**params, "router": jnp.zeros_like(params["router"])}
    assert jnp.isclose(moe_ffn(uniform, x, cfg)[1]["aux"], 1.0)

    # every token sends all its probability mass to expert 0
    collapsed = {**params, "router": jnp.zeros_like(params["router"]).at[:, 0].set(100.0)}
    assert jnp.isclose(moe_ffn(collapsed, jnp.abs(x), cfg)[1]["aux"], E, rtol=1e-3)


def test_capacity_drops_tokens():
    cfg = dataclasses.replace(TINY_MOE, top_k=1, capacity_factor=1.0)
    params = init_params(cfg, jax.random.key(0))["layers"][0]["ffn"]
    x = jnp.abs(_x(cfg))
    collapsed = {**params, "router": jnp.zeros_like(params["router"]).at[:, 0].set(100.0)}
    # all S tokens pick expert 0, which holds only C = S/E of them
    np.testing.assert_allclose(moe_ffn(collapsed, x, cfg)[1]["drop"], 1 - 1 / cfg.n_experts, rtol=1e-6)
    # with enough capacity nothing is dropped
    big = dataclasses.replace(cfg, capacity_factor=float(cfg.n_experts))
    assert moe_ffn(collapsed, x, big)[1]["drop"] == 0.0


def test_causal_when_nothing_dropped():
    cfg = dataclasses.replace(TINY_MOE, capacity_factor=float(TINY_MOE.n_experts))
    params = init_params(cfg, jax.random.key(0))
    x = jax.random.randint(jax.random.key(1), (1, cfg.seq_len), 0, cfg.vocab_size)
    t = cfg.seq_len // 2
    x2 = x.at[0, t].set((x[0, t] + 1) % cfg.vocab_size)
    a, b = forward(params, x, cfg)[0], forward(params, x2, cfg)[0]
    np.testing.assert_allclose(a[:, :t], b[:, :t], atol=1e-5)


def test_moe_preset_is_about_10m_total_with_fewer_active():
    cfg = PRESETS["moe"]
    params = jax.eval_shape(lambda k: init_params(cfg, k), jax.random.key(0))
    total, active = param_count(params), active_param_count(params, cfg)
    assert 9e6 < total < 11e6
    assert active < 0.6 * total
