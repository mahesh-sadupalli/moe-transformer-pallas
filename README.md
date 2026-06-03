# Frontier AI Lab Prep

Roadmap based on Vlad Feinberg's advice: https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html

## Core thesis
Demonstrated skills beat credentials. Three traits predict success: **intent**, **mathematical maturity**, **grit**.

## Strategic entry points
Don't compete head-on for LLM training roles. Target the adjacent areas labs actively hire for:

- **Below the stack — Kernels.** Performance optimization via low-level code. Find bottlenecks in communication, computation, or memory. Examples: Flash Attention, quantization, custom kernels.
- **Above the stack — Agents.** Rigorous experiments using LLMs as components in larger systems (e.g., language models inside algorithm search loops).

## The 4-step roadmap

### 1. Master JAX → `01-jax/`
Work through the official JAX tutorials and exercises. Goal: comfort with `jit`, `vmap`, `pmap`, `grad`, sharding, and the functional style.

### 2. Scaling book → `02-scaling-book/`
Work through "How to Scale Your Model" (Google DeepMind's scaling book) with hands-on problems. Goal: internalize the math of compute/memory/communication tradeoffs at scale.

### 3. 10M parameter transformer → `03-transformer-10m/`
Build a ~10M parameter transformer **from scratch** in JAX. Train it end-to-end. Goal: prove you understand the full stack — tokenization, attention, optimizer, training loop, eval.

### 4. Custom kernel → `04-custom-kernel/`
Write a custom kernel (Pallas, Triton, or CUDA) demonstrating **measurable speedups** over a baseline. Goal: prove you can move below the framework and reason about hardware.

## Output
Each subdirectory should contain working code + a short writeup of what was learned and what was measured. The point is a documented portfolio, not just exercises completed.
