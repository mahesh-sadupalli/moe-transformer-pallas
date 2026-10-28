"""Training loop.

    uv run python -m moe.train --preset tiny --train-split valid --steps 300
    uv run python -m moe.train --preset base

Each run writes runs/<name>/{config.json, params.pkl, metrics.jsonl}.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np
import optax

from moe import checkpoint
from moe.config import PRESETS, ModelConfig, TrainConfig
from moe.data import get_batch, load_split
from moe.model import active_param_count, init_params, loss_fn, param_count


def make_optimizer(tc: TrainConfig):
    warmup = min(tc.warmup_steps, tc.steps // 10)  # short smoke runs still get a decay phase
    schedule = optax.warmup_cosine_decay_schedule(0.0, tc.lr, warmup, tc.steps)
    decay_mask = lambda p: jax.tree.map(lambda x: x.ndim >= 2, p)  # no decay on norm gains
    return optax.chain(
        optax.clip_by_global_norm(tc.grad_clip),
        optax.adamw(schedule, weight_decay=tc.weight_decay, mask=decay_mask),
    )


def make_train_step(cfg: ModelConfig, opt):
    @jax.jit
    def train_step(params, opt_state, x, y):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    return train_step


def train(cfg: ModelConfig, tc: TrainConfig, train_tokens, val_tokens, out_dir: Path | None = None):
    params = init_params(cfg, jax.random.key(tc.seed))
    opt = make_optimizer(tc)
    opt_state = opt.init(params)
    train_step = make_train_step(cfg, opt)
    eval_ce = jax.jit(lambda p, x, y: loss_fn(p, x, y, cfg)[1]["ce"])
    rng = np.random.default_rng(tc.seed)
    # Fixed val batches, so val loss is comparable across evals and across runs.
    val_rng = np.random.default_rng(tc.seed + 1)
    val_batches = [get_batch(val_tokens, tc.batch_size, cfg.seq_len, val_rng) for _ in range(tc.eval_batches)]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "metrics.jsonl", "w")
    print(f"params: {param_count(params):,} total, {active_param_count(params, cfg):,} active"
          f"  backend: {jax.default_backend()}")

    t0, last_step = time.perf_counter(), 0
    for step in range(1, tc.steps + 1):
        x, y = get_batch(train_tokens, tc.batch_size, cfg.seq_len, rng)
        params, opt_state, metrics = train_step(params, opt_state, x, y)
        if step % tc.eval_every == 0 or step == tc.steps:
            metrics = jax.device_get(metrics)  # blocks until the step is done
            tok_per_s = (step - last_step) * tc.batch_size * cfg.seq_len / (time.perf_counter() - t0)
            val = float(np.mean([eval_ce(params, x, y) for x, y in val_batches]))
            record = {"step": step, "train_ce": float(metrics["ce"]), "val_ce": val, "tok_per_s": tok_per_s}
            moe = ""
            if cfg.n_experts:
                record |= {"aux": float(metrics["aux"]), "drop": float(metrics["drop"]),
                           "load": metrics["load"].tolist()}
                moe = f"  aux {record['aux']:.3f}  drop {record['drop']:.1%}"
            print(f"step {step:6d}  train {record['train_ce']:.4f}  val {val:.4f}{moe}  {tok_per_s:,.0f} tok/s")
            if out_dir:
                log.write(json.dumps(record) + "\n")
                log.flush()
                checkpoint.save(out_dir, params, cfg)
            t0, last_step = time.perf_counter(), step
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="base")
    ap.add_argument("--train-split", default="train", help="'valid' for a quick smoke run")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--name", help="run directory name under runs/ (default: preset)")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("steps", "batch_size", "eval_every") and v is not None}
    tc = dataclasses.replace(TrainConfig(), **overrides)
    out_dir = Path("runs") / (args.name or args.preset)
    train(PRESETS[args.preset], tc, load_split(args.train_split), load_split("valid"), out_dir)


if __name__ == "__main__":
    main()
