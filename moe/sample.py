"""Generate text from a saved run.

    uv run python -m moe.sample runs/tiny --prompt "Once upon a time"
"""

import argparse
import dataclasses
from pathlib import Path

import jax
import numpy as np

from moe import checkpoint
from moe.data import decode, encode
from moe.model import forward


def generate(params, cfg, prompt: str, n_new: int, temperature: float, key):
    # Pad into a fixed (1, seq_len) buffer so one compiled function serves every step. The
    # model is causal, so padding after the context can't change its logits, provided MoE
    # routing drops nothing (see moe_ffn). capacity_factor = n_experts guarantees that.
    cfg = dataclasses.replace(cfg, capacity_factor=float(max(cfg.n_experts, 1)))
    T = cfg.seq_len

    @jax.jit
    def next_token(params, buf, i, key):
        logits = forward(params, buf, cfg)[0][0, i]
        return jax.random.categorical(key, logits / temperature)

    tokens = list(encode(prompt))
    for _ in range(n_new):
        ctx = tokens[-T:]
        buf = np.zeros((1, T), np.int32)
        buf[0, : len(ctx)] = ctx
        key, sub = jax.random.split(key)
        tokens.append(int(next_token(params, buf, len(ctx) - 1, sub)))
    return decode(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=300, help="bytes to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params, cfg = checkpoint.load(args.run_dir)
    print(generate(params, cfg, args.prompt, args.n, args.temperature, jax.random.key(args.seed)))


if __name__ == "__main__":
    main()
