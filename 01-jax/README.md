# Step 1 — Master JAX

Goal (from the roadmap): genuine comfort with `grad`, `jit`, `vmap`, `pmap`/sharding,
and — most importantly — the **functional style** that makes those transforms work.

## Why this matters for a frontier lab

Every transform in JAX (`jit`, `grad`, `vmap`, `pmap`, `shard_map`) is a function that
takes a *pure function* and returns a new function. If you internalize the functional
mental model, the rest is composition. Labs care that you can reason about *why* a
transform traces the way it does, not just that you can call it.

## How the tutor loop works

Each `lesson_NN_*.py` file has three parts:

1. **Notes** — a docstring explaining the concept precisely.
2. **Exercises** — functions with `# TODO` and `raise NotImplementedError`. You implement them.
3. **Checker** — run `python lesson_NN_*.py` to test your implementations and get pass/fail.

Fill in the TODOs, run the file, iterate until all checks pass. Then ping me to review
your solution and we move to the next lesson. Don't peek at solutions — the grit is the point.

## Curriculum

| # | Lesson | Core idea |
|---|--------|-----------|
| 00 | Functional model | immutable arrays, pure functions, PRNG keys, pytrees |
| 01 | `grad` | autodiff, `value_and_grad`, higher-order grads, a real loss |
| 02 | `jit` | tracing, why purity matters, `static_argnums`, `lax` control flow |
| 03 | `vmap` | auto-batching, `in_axes`/`out_axes`, per-sample gradients |
| 04 | `pmap` / sharding | multi-device parallelism, `NamedSharding`, `shard_map` |

## Running

```bash
python 01-jax/lesson_00_functional_model.py
```

For the multi-device lesson (04), we fake 8 CPU devices since the M2 has one:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python 01-jax/lesson_04_pmap_sharding.py
```

## Output for the portfolio

When all lessons pass, write `NOTES.md` here: what each transform does, the one pitfall
that bit you per lesson, and a 3-sentence explanation of the functional model in your own
words. That writeup is the artifact, not the passing checks.
