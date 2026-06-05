# 10M-Parameter MoE Transformer: Training, Scaling, and Kernel Fusion

A from-scratch Mixture-of-Experts transformer in JAX (~10M params), trained end-to-end, with a
custom Pallas kernel that fuses the MoE hot path for a measurable speedup.

Built as a four-step portfolio, following the strategy in Vlad Feinberg's "How to land a job at a
frontier lab": https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html

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

### 3. 10M-parameter MoE transformer → `03-transformer-10m/`
Build a ~10M parameter **Mixture-of-Experts** transformer **from scratch** in JAX and train it
end-to-end. Beyond a dense transformer this adds the pieces real frontier models use: a routing
network (top-k gating), expert dispatch/combine, and an auxiliary load-balancing loss. Goal: prove
you understand the full stack — tokenization, attention, MoE routing, optimizer, training loop, eval.

### 4. Custom Pallas kernel — MoE fusion → `04-custom-kernel/`
Write a custom **Pallas** kernel that fuses the MoE hot path (gating → dispatch → expert matmul →
combine) and demonstrate a **measurable speedup** over the step-3 baseline. Goal: prove you can move
below the framework, reason about memory traffic and the roofline, and actually make it faster.

## Output
Each subdirectory should contain working code + a short writeup of what was learned and what was measured. The point is a documented portfolio, not just exercises completed.
