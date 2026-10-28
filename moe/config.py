from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256  # byte-level tokens
    seq_len: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536  # per expert when n_experts > 0
    n_experts: int = 0  # 0 = dense FFN
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01  # load balancing (Switch)
    z_loss_coef: float = 0.001  # router z-loss (ST-MoE)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    steps: int = 5000
    lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 0


PRESETS = {
    "tiny": ModelConfig(seq_len=64, d_model=64, n_layers=2, n_heads=2, d_ff=256),
    "tiny_moe": ModelConfig(seq_len=64, d_model=64, n_layers=2, n_heads=2, d_ff=64, n_experts=4),
    "base": ModelConfig(),  # ~10.8M params, dense baseline
    # ~10.8M total params (same as base), ~5.5M active per token
    "moe": ModelConfig(d_ff=192, n_experts=8, top_k=2),
    # dense FFN with d_ff = top_k * 192: same active params / FLOPs per token as "moe"
    "dense_matched": ModelConfig(d_ff=384),
}
