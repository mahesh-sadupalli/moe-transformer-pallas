"""Sanity-check the toolchain every milestone depends on.

    uv run python scripts/check_env.py
    XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python scripts/check_env.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def main():
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")

    x = jnp.arange(8.0)
    assert jnp.allclose(jax.jit(jax.grad(lambda x: (x**2).sum()))(x), 2 * x)
    print("  [PASS] jit + grad")

    # Pallas only compiles for TPU/GPU; on CPU it must run in interpret mode.
    on_cpu = jax.default_backend() == "cpu"
    out = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=on_cpu,
    )(x, x)
    assert jnp.allclose(out, 2 * x)
    print(f"  [PASS] pallas_call (interpret={on_cpu})")

    print(f"  [INFO] {jax.device_count()} device(s) visible")


if __name__ == "__main__":
    main()
