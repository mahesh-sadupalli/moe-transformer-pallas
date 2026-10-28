# MoE Transformer + Pallas

I'm building a small Mixture-of-Experts language model from scratch in JAX, training it on
TinyStories, and then writing a custom Pallas kernel to make its slowest layer fast on a TPU.

The model is about 10.8M parameters. Every byte is a token, the feed-forward block is a
Mixture of Experts (8 experts, each token goes to 2), and there's no Flax or Haiku anywhere.
The parameters are a plain dict of arrays and the model is one pure function.

## Why I'm doing this

I want to show I can work across the whole stack, from the model and training loop down to
the hardware. Kernels are where that shows up most clearly: you find what the chip is
actually waiting on, then write code that removes the wait. The idea came from Vlad
Feinberg's essay on getting into a frontier lab, which argues that work like this says more
than a list of courses: https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html

## Where it stands

- **Done:** the foundation (data, dense transformer, training loop, tests) and the MoE layer
  (router, expert capacity, load-balancing and z-loss).
- **Ready to run:** the TPU experiments. Everything that works on a laptop is in place:
  napkin-math predictions, a layer benchmark, logging, checkpoints, plots.
- **Next:** a Pallas grouped-matmul kernel, then fusing the whole MoE layer into one kernel.

The full plan, with a concrete finish line for each step, is in [MILESTONES.md](MILESTONES.md).

## What I've found so far

On my laptop (M2 CPU, 150 steps), the MoE model reaches a validation loss of 2.37 and the
share of tokens dropped by full experts falls from 40% to about 6% as the router learns to
spread the load.

The more interesting result is in the napkin math ([docs/napkin_math.md](docs/napkin_math.md)).
At this model size, every operation on a TPU v5e is limited by memory bandwidth, not compute.
The standard way of routing tokens to experts (one-hot einsums, as in GShard and Switch)
costs more time than the experts themselves. So the MoE model is predicted to be *slower*
than a dense model of the same size, even though it does half the work per token.

That sets up the kernel work:

| MoE layer, forward, one layer | Predicted time | Speedup |
|---|---:|---:|
| Today: einsum dispatch and combine | 295 µs | 1.0x |
| Sort tokens by expert + grouped matmul | 206 µs | 1.4x |
| One fused kernel (hidden layer stays on chip) | 52 µs | 5.7x |

These are predictions. Checking them on real hardware is the point of the next milestones.

## Running it

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                     # 12 tests, a few seconds
uv run python scripts/check_env.py
```

Train something small to see it work (downloads the 19 MB validation file on first run):

```bash
uv run python -m moe.train --preset tiny_moe --train-split valid --steps 300
uv run python -m moe.sample runs/tiny_moe
```

The real runs need a TPU. [docs/tpu.md](docs/tpu.md) has the exact steps for Colab, Kaggle or a
TPU VM.

```bash
uv run python -m moe.train --preset base            # dense, 10.8M params
uv run python -m moe.train --preset dense_matched   # dense, same compute per token as moe
uv run python -m moe.train --preset moe             # 10.8M total, 5.5M used per token
```

## What's in here

```
moe/
  config.py      model and training presets
  data.py        TinyStories as raw bytes, random training windows
  model.py       transformer, attention, dense FFN and the MoE layer
  train.py       AdamW, warmup + cosine schedule, eval, logging
  checkpoint.py  save and load runs
  sample.py      generate text from a checkpoint
tests/           one test per claim (causality, MoE matches dense with 1 expert, ...)
scripts/
  napkin_math.py FLOPs, bytes and roofline predictions
  bench_moe.py   times the FFN / MoE layer on its own
  plot.py        loss curves and expert-load heatmaps
  check_env.py   checks JAX and Pallas work
docs/            the predictions and the TPU runbook
```

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017)
- Lepikhin et al., *GShard* (2020)
- Fedus, Zoph and Shazeer, *Switch Transformers* (2021)
- Zoph et al., *ST-MoE* (2022)
- Gale et al., *MegaBlocks* (2022)
- Eldan and Li, *TinyStories* (2023)
- Google DeepMind, *How to Scale Your Model*
