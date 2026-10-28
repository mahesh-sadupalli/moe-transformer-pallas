import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from moe.config import PRESETS, TrainConfig
from moe.model import forward, init_params, param_count
from moe.train import make_optimizer, make_train_step

TINY = PRESETS["tiny"]


def test_base_is_about_10m_params():
    params = jax.eval_shape(lambda k: init_params(PRESETS["base"], k), jax.random.key(0))
    assert 9e6 < param_count(params) < 11e6


def test_causal():
    """Changing token t must not change logits at positions < t."""
    params = init_params(TINY, jax.random.key(0))
    x = jax.random.randint(jax.random.key(1), (1, TINY.seq_len), 0, TINY.vocab_size)
    t = TINY.seq_len // 2
    x2 = x.at[0, t].set((x[0, t] + 1) % TINY.vocab_size)
    a, b = forward(params, x, TINY)[0], forward(params, x2, TINY)[0]
    np.testing.assert_allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not jnp.allclose(a[:, t], b[:, t])


def test_overfits_single_batch():
    tc = dataclasses.replace(TrainConfig(), steps=200, lr=3e-3, warmup_steps=10)
    params = init_params(TINY, jax.random.key(0))
    opt = make_optimizer(tc)
    opt_state = opt.init(params)
    step = make_train_step(TINY, opt)
    x = jax.random.randint(jax.random.key(1), (4, TINY.seq_len + 1), 0, TINY.vocab_size)
    x, y = x[:, :-1], x[:, 1:]
    for _ in range(tc.steps):
        params, opt_state, metrics = step(params, opt_state, x, y)
    assert metrics["ce"] < 0.1, f"loss {metrics['ce']}"
