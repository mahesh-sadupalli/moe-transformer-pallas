import numpy as np

from moe.data import decode, encode, get_batch


def test_roundtrip_unicode():
    text = "Once upon a time — a naïve dragon 🐉."
    assert decode(encode(text)) == text


def test_batch_targets_are_shifted_inputs():
    tokens = np.arange(1000, dtype=np.int64) % 256
    x, y = get_batch(tokens.astype(np.uint8), batch_size=4, seq_len=16, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (4, 16)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
