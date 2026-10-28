"""Save/load a run: params as a pickled numpy pytree, config as JSON."""

import dataclasses
import json
import pickle
from pathlib import Path

import jax

from moe.config import ModelConfig


def save(out_dir: Path, params, cfg: ModelConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    tmp = out_dir / "params.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    tmp.rename(out_dir / "params.pkl")


def load(out_dir: Path):
    cfg = ModelConfig(**json.loads((out_dir / "config.json").read_text()))
    with open(out_dir / "params.pkl", "rb") as f:
        return pickle.load(f), cfg
