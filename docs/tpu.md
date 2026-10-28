# Running on a TPU

The laptop (Apple M2) can check correctness only. Full training runs, the baseline profile and
all kernel timings happen here. Works on Colab (Runtime → TPU), Kaggle TPU, or a TPU Research
Cloud VM.

## Setup
```bash
pip install uv
git clone https://github.com/mahesh-sadupalli/moe-transformer-pallas.git
cd moe-transformer-pallas
uv sync --extra tpu
uv run python scripts/check_env.py      # expect backend=tpu and pallas_call (interpret=False)
```
In Colab, put `%cd moe-transformer-pallas` in its own cell first, since each `!` command runs in a fresh
shell. If the chip isn't a v5e, update `PEAK_FLOPS` / `HBM_BW` in `scripts/napkin_math.py` and
regenerate `docs/napkin_math.md` before comparing.

## M2 runs
```bash
# 1. Baseline profile of the layer (the number M3/M4 must beat)
uv run python scripts/bench_moe.py --profile runs/trace

# 2. Training runs: same data order, same val batches, same step count
uv run python -m moe.train --preset base
uv run python -m moe.train --preset dense_matched
uv run python -m moe.train --preset moe

# 3. Plots and a sample
uv run python scripts/plot.py runs/base runs/dense_matched runs/moe --out docs/figures
uv run python -m moe.sample runs/moe
```
Default is 5,000 steps × 8,192 tokens ≈ 41M tokens (~2% of TinyStories). Raise `--steps` if the
curves are still falling steeply at the end; use the same value for all three runs.

## Bring back
- `bench_moe.py` table → `docs/napkin_math.md` "What to check"
- `runs/*/metrics.jsonl` (small) and `docs/figures/*.png` → commit
- The profiler trace: note the top ops by time in the MoE layer
