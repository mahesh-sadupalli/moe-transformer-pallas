"""Time the FFN layer in isolation (forward, and forward+backward) for each preset.

    uv run python scripts/bench_moe.py
    uv run python scripts/bench_moe.py --profile /tmp/trace   # view in TensorBoard / Perfetto

"moe experts only" runs just the two expert matmuls on an already-dispatched buffer, so
`moe einsum - moe experts only` is the measured cost of routing + dispatch + combine.
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from moe.config import PRESETS, TrainConfig
from moe.model import ffn, init_params, moe_ffn


def bench(fn, *args, iters):
    jax.block_until_ready(fn(*args))  # compile
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def variants(B):
    key = jax.random.key(0)
    for name in ("base", "dense_matched", "moe"):
        cfg = PRESETS[name]
        p = init_params(cfg, key)["layers"][0]["ffn"]
        x = jax.random.normal(key, (B, cfg.seq_len, cfg.d_model))
        if cfg.n_experts:
            yield "moe einsum", (lambda p, x, cfg=cfg: moe_ffn(p, x, cfg)[0]), p, x
            C = math.ceil(cfg.top_k * cfg.seq_len * cfg.capacity_factor / cfg.n_experts)
            buf = jax.random.normal(key, (cfg.n_experts, B, C, cfg.d_model))
            experts = lambda p, b: jnp.einsum(
                "egcf,efd->egcd", jax.nn.gelu(jnp.einsum("egcd,edf->egcf", b, p["w_in"])), p["w_out"])
            yield "moe experts only", experts, {"w_in": p["w_in"], "w_out": p["w_out"]}, buf
        else:
            yield f"dense ffn ({name})", ffn, p, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile", help="write a jax.profiler trace to this directory")
    args = ap.parse_args()

    print(f"backend {jax.default_backend()}, batch {args.batch_size}\n")
    print("| layer | fwd ms | fwd+bwd ms |")
    print("|---|---:|---:|")
    if args.profile:
        jax.profiler.start_trace(args.profile)
    for name, fn, p, x in variants(args.batch_size):
        fwd = jax.jit(fn)
        fwd_bwd = jax.jit(jax.grad(lambda p, x, fn=fn: fn(p, x).sum(), argnums=(0, 1)))
        print(f"| {name} | {bench(fwd, p, x, iters=args.iters) * 1e3:.2f} | "
              f"{bench(fwd_bwd, p, x, iters=args.iters) * 1e3:.2f} |")
    if args.profile:
        jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
