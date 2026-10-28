import jax
import numpy as np

from moe import checkpoint
from moe.config import PRESETS
from moe.model import init_params
from moe.sample import generate


def test_roundtrip_and_generate(tmp_path):
    cfg = PRESETS["tiny_moe"]
    params = init_params(cfg, jax.random.key(0))
    checkpoint.save(tmp_path, params, cfg)
    loaded, loaded_cfg = checkpoint.load(tmp_path)
    assert loaded_cfg == cfg
    jax.tree.map(np.testing.assert_array_equal, params, loaded)

    out = generate(loaded, cfg, "Once", n_new=cfg.seq_len + 5, temperature=1.0, key=jax.random.key(0))
    assert out.startswith("Once")  # also exercises the sliding window past seq_len
