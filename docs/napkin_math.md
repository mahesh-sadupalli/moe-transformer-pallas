# Napkin math

Predictions made **before** running on a TPU, to be confirmed or refuted by `scripts/bench_moe.py`
and the training runs. Everything below the line is the exact output of
`uv run python scripts/napkin_math.py`; rerun it after changing a config.

## Assumptions
- One TPU v5e chip: 197 TFLOP/s (bf16 MXU; f32 matmuls at default precision run as bf16 passes),
  819 GB/s HBM, so the critical intensity is ~241 FLOPs/byte.
- f32 params and activations, as the code currently runs.
- No fusion: every matmul reads its operands from HBM and writes its result back. This
  *over*-estimates time where XLA fuses, and elementwise ops (softmax, GELU, norms, one-hot)
  aren't counted at all, which *under*-estimates. Treat the step times as order-of-magnitude.

## Takeaways
1. **Everything is memory-bound.** With d_model = 384, no matmul gets near 241 FLOPs/byte
   (best is ~148). At this size HBM traffic sets the speed, not FLOPs, so fewer HBM round trips
   (fusion) is the lever, not fewer FLOPs.
2. **The einsum MoE is predicted to be slower than the dense model with the same total params**
   (10.4 vs 8.0 ms/step) despite half the active params, and **~1.7x slower than
   `dense_matched`**, which does the same useful FLOPs per token.
3. **Routing costs more than the experts.** Dispatch + combine are 1.33x the expert-matmul FLOPs,
   and they move 130 MB per layer through two (B, S, E, C) one-hot tensors (42 MB). Capacity
   padding wastes another 20% of expert FLOPs.
4. **Targets for the kernel work** (MoE layer forward, per layer): sorting tokens + a grouped
   matmul (M3) predicts only **~1.4x**, because gather/scatter and the intermediate `h` still go
   through HBM. Fusing gather → GMM → GELU → GMM → combine so `h` never leaves VMEM (M4) predicts
   **~5.7x**. Fusion is the real prize, which is what the roofline says for a memory-bound layer.

## What to check on the TPU
- [ ] `bench_moe.py`: is routing (`moe einsum` − `moe experts only`) ~60% of the MoE layer, as
      predicted? (On the M2 CPU it is: 23 of 39 ms forward, 59%.)
- [ ] Training tok/s ordering: `dense_matched` > `base` > `moe`?
- [ ] Measured step times vs the roofline column: how far off is each, and why?

---

```
# Napkin math. Target TPU v5e: 197 TFLOP/s, 819 GB/s, critical intensity 241 FLOPs/byte

## base: 10.82M total, 10.82M active params; batch 32 x 256 = 8,192 tokens/step
optimizer state (param + grad + Adam m,v; f32): 173 MB
train FLOPs/step: 584.7 G  (71.4 MFLOP/token; 6*active = 64.9)
roofline step time on TPU v5e (no fusion): 7.98 ms -> 1,027k tok/s (compute-only bound 2.97 ms)

| op (per layer) | GFLOP | MB moved | FLOPs/byte | bound | roofline us |
|---|---:|---:|---:|---|---:|
| attn qkv | 7.248 | 52.1 | 139 | memory | 63.6 |
| attn scores | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn @ v | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn out | 2.416 | 25.8 | 94 | memory | 31.4 |
| ffn in | 9.664 | 65.3 | 148 | memory | 79.7 |
| ffn out | 9.664 | 65.3 | 148 | memory | 79.7 |

## dense_matched: 5.51M total, 5.51M active params; batch 32 x 256 = 8,192 tokens/step
optimizer state (param + grad + Adam m,v; f32): 88 MB
train FLOPs/step: 323.7 G  (39.5 MFLOP/token; 6*active = 33.1)
roofline step time on TPU v5e (no fusion): 6.24 ms -> 1,313k tok/s (compute-only bound 1.64 ms)

| op (per layer) | GFLOP | MB moved | FLOPs/byte | bound | roofline us |
|---|---:|---:|---:|---|---:|
| attn qkv | 7.248 | 52.1 | 139 | memory | 63.6 |
| attn scores | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn @ v | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn out | 2.416 | 25.8 | 94 | memory | 31.4 |
| ffn in | 2.416 | 25.8 | 94 | memory | 31.4 |
| ffn out | 2.416 | 25.8 | 94 | memory | 31.4 |

## moe: 10.84M total, 5.53M active params; batch 32 x 256 = 8,192 tokens/step
optimizer state (param + grad + Adam m,v; f32): 173 MB
train FLOPs/step: 491.3 G  (60.0 MFLOP/token; 6*active = 33.2)
roofline step time on TPU v5e (no fusion): 10.43 ms -> 786k tok/s (compute-only bound 2.49 ms)

| op (per layer) | GFLOP | MB moved | FLOPs/byte | bound | roofline us |
|---|---:|---:|---:|---|---:|
| attn qkv | 7.248 | 52.1 | 139 | memory | 63.6 |
| attn scores | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn @ v | 1.611 | 75.5 | 21 | memory | 92.2 |
| attn out | 2.416 | 25.8 | 94 | memory | 31.4 |
| router | 0.050 | 12.9 | 4 | memory | 15.7 |
| moe dispatch | 4.027 | 65.0 | 62 | memory | 79.4 |
| expert in | 3.020 | 49.5 | 61 | memory | 60.5 |
| expert out | 3.020 | 49.5 | 61 | memory | 60.5 |
| moe combine | 4.027 | 65.0 | 62 | memory | 79.4 |

MoE layer: dispatch+combine = 1.33x the expert matmul FLOPs; capacity padding wastes 20% of expert FLOPs
MoE layer roofline time: routing 174.5 us vs experts 121.0 us
dispatch/combine tensors (B,S,E,C): 42 MB per layer

MoE layer forward, roofline per layer:
| variant | us | vs current |
|---|---:|---:|
| current einsum dispatch | 295.4 | 1.0x |
| sort + grouped matmul (M3) | 205.8 | 1.4x |
| fully fused gather-GMM-gelu-GMM-combine (M4) | 52.2 | 5.7x |
```
