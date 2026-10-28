"""Napkin math: params, FLOPs, memory traffic and a per-op roofline for the presets.

    uv run python scripts/napkin_math.py

Counts follow the code as written: f32 everywhere, full (unmasked) T x T attention, einsum
MoE dispatch/combine over capacity-padded expert buffers. No fusion is assumed: every op
reads its operands from HBM and writes its result back. Training = 3x forward (backward ~2x).
"""

import math

import jax

from moe.config import PRESETS, TrainConfig
from moe.model import active_param_count, init_params, param_count

# Target: one TPU v5e chip. Swap these for other hardware.
HW = "TPU v5e"
PEAK_FLOPS = 197e12  # bf16 MXU; f32 matmuls at default precision run as bf16 passes
HBM_BW = 819e9  # bytes/s
BYTES = 4  # f32
CRITICAL = PEAK_FLOPS / HBM_BW  # FLOPs/byte where an op stops being memory-bound


def matmul(name, m, k, n, count=1):
    """count independent (m,k) @ (k,n) matmuls -> (name, FLOPs, HBM bytes)."""
    return name, 2 * m * k * n * count, BYTES * (m * k + k * n + m * n) * count


def layer_ops(cfg, B):
    T, D, F, H = cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.n_heads
    N = B * T
    ops = [
        matmul("attn qkv", N, D, 3 * D),
        matmul("attn scores", T, D // H, T, count=B * H),
        matmul("attn @ v", T, T, D // H, count=B * H),
        matmul("attn out", N, D, D),
    ]
    if not cfg.n_experts:
        return ops + [matmul("ffn in", N, D, F), matmul("ffn out", N, F, D)]
    E = cfg.n_experts
    C = math.ceil(cfg.top_k * T * cfg.capacity_factor / E)  # groups = sequences
    return ops + [
        matmul("router", N, D, E),
        matmul("moe dispatch", E * C, T, D, count=B),  # (EC, S) @ (S, D) per group
        matmul("expert in", B * C, D, F, count=E),
        matmul("expert out", B * C, F, D, count=E),
        matmul("moe combine", T, E * C, D, count=B),  # (S, EC) @ (EC, D) per group
    ]


def roofline(ops):
    return sum(max(f / PEAK_FLOPS, b / HBM_BW) for _, f, b in ops)


def report(name, cfg, B):
    params = jax.eval_shape(lambda k: init_params(cfg, k), jax.random.key(0))
    total, active = param_count(params), active_param_count(params, cfg)
    N = B * cfg.seq_len
    layer = layer_ops(cfg, B)
    unembed = [matmul("unembed", N, cfg.d_model, cfg.vocab_size)]
    fwd_ops = layer * cfg.n_layers + unembed
    fwd_flops = sum(f for _, f, _ in fwd_ops)
    step_s = 3 * roofline(fwd_ops)

    print(f"\n## {name}: {total / 1e6:.2f}M total, {active / 1e6:.2f}M active params; "
          f"batch {B} x {cfg.seq_len} = {N:,} tokens/step")
    print(f"optimizer state (param + grad + Adam m,v; f32): {16 * total / 1e6:.0f} MB")
    print(f"train FLOPs/step: {3 * fwd_flops / 1e9:.1f} G  ({3 * fwd_flops / N / 1e6:.1f} MFLOP/token; "
          f"6*active = {6 * active / 1e6:.1f})")
    print(f"roofline step time on {HW} (no fusion): {step_s * 1e3:.2f} ms -> {N / step_s / 1e3:,.0f}k tok/s "
          f"(compute-only bound {3 * fwd_flops / PEAK_FLOPS * 1e3:.2f} ms)")
    print(f"\n| op (per layer) | GFLOP | MB moved | FLOPs/byte | bound | roofline us |")
    print(f"|---|---:|---:|---:|---|---:|")
    for op, f, b in layer:
        bound = "compute" if f / b > CRITICAL else "memory"
        print(f"| {op} | {f / 1e9:.3f} | {b / 1e6:.1f} | {f / b:.0f} | {bound} | "
              f"{max(f / PEAK_FLOPS, b / HBM_BW) * 1e6:.1f} |")
    if cfg.n_experts:
        E = cfg.n_experts
        C = math.ceil(cfg.top_k * cfg.seq_len * cfg.capacity_factor / E)
        get = lambda *names: sum(f for op, f, _ in layer if op in names)
        expert = get("expert in", "expert out")
        routing = get("moe dispatch", "moe combine")
        useful = expert * cfg.top_k * cfg.seq_len / (E * C)  # slots actually holding tokens (if nothing dropped)
        t = lambda *names: sum(max(f / PEAK_FLOPS, b / HBM_BW) for op, f, b in layer if op in names)
        print(f"\nMoE layer: dispatch+combine = {routing / expert:.2f}x the expert matmul FLOPs; "
              f"capacity padding wastes {1 - useful / expert:.0%} of expert FLOPs")
        print(f"MoE layer roofline time: routing {t('router', 'moe dispatch', 'moe combine') * 1e6:.1f} us vs "
              f"experts {t('expert in', 'expert out') * 1e6:.1f} us")
        print(f"dispatch/combine tensors (B,S,E,C): {2 * B * cfg.seq_len * E * C * BYTES / 1e6:.0f} MB per layer")

        # Targets for M3/M4: route only real assignments (k*N rows, no capacity padding).
        D, F, kN = cfg.d_model, cfg.d_ff, cfg.top_k * N
        w = BYTES * E * D * F  # one expert weight matrix stack
        gather = (0, BYTES * (N * D + kN * D))  # read x, write sorted copies
        scatter = (0, BYTES * (kN * D + N * D))  # read expert outputs, write combined y
        gmm_in = (2 * kN * D * F, BYTES * (kN * D + kN * F) + w)
        gmm_out = (2 * kN * F * D, BYTES * (kN * F + kN * D) + w)
        fused = (2 * 2 * kN * D * F, BYTES * (2 * N * D) + 2 * w)  # x in, y out, h never leaves VMEM
        rt = lambda *ops: sum(max(f / PEAK_FLOPS, b / HBM_BW) for f, b in ops)
        router = t("router")
        current = t("router", "moe dispatch", "expert in", "expert out", "moe combine")
        gmm = router + rt(gather, gmm_in, gmm_out, scatter)
        fus = router + rt(fused)
        print(f"\nMoE layer forward, roofline per layer:")
        print(f"| variant | us | vs current |")
        print(f"|---|---:|---:|")
        print(f"| current einsum dispatch | {current * 1e6:.1f} | 1.0x |")
        print(f"| sort + grouped matmul (M3) | {gmm * 1e6:.1f} | {current / gmm:.1f}x |")
        print(f"| fully fused gather-GMM-gelu-GMM-combine (M4) | {fus * 1e6:.1f} | {current / fus:.1f}x |")


def main():
    B = TrainConfig().batch_size
    print(f"# Napkin math. Target {HW}: {PEAK_FLOPS / 1e12:.0f} TFLOP/s, {HBM_BW / 1e9:.0f} GB/s, "
          f"critical intensity {CRITICAL:.0f} FLOPs/byte")
    for name in ("base", "dense_matched", "moe"):
        report(name, PRESETS[name], B)


if __name__ == "__main__":
    main()
