"""Plot val loss curves for several runs, plus expert load for each MoE run.

    uv run python scripts/plot.py runs/base runs/dense_matched runs/moe --out docs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(run_dir: Path):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in args.runs:
        rec = read(run)
        ax.plot([r["step"] for r in rec], [r["val_ce"] for r in rec], label=run.name)
    ax.set(xlabel="step", ylabel="val loss (nats/byte)", title="Validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "val_loss.png", dpi=150)
    print(f"wrote {args.out / 'val_loss.png'}")

    for run in args.runs:
        rec = [r for r in read(run) if "load" in r]
        if not rec:
            continue
        load = np.array(rec[-1]["load"])  # (L, E), fraction of assignments per expert
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(load * load.shape[1], cmap="RdBu_r", vmin=0, vmax=2)  # 1.0 = perfectly balanced
        ax.set(xticks=range(load.shape[1]), yticks=range(load.shape[0]),
               xlabel="expert", ylabel="layer", title=f"{run.name}: expert load x E at step {rec[-1]['step']}")
        fig.colorbar(im, ax=ax, label="1.0 = balanced")
        fig.tight_layout()
        path = args.out / f"{run.name}_expert_load.png"
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
