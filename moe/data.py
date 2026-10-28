"""TinyStories with byte-level tokens.

Byte-level means the raw UTF-8 text file *is* the token array, so there is no
tokenizer to train and no preprocessing step: we memmap the file as uint8.
"""

import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{split}.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def decode(tokens) -> str:
    return bytes(np.asarray(tokens, dtype=np.uint8)).decode("utf-8", errors="replace")


def load_split(split: str) -> np.ndarray:
    """Return the tokens of `split` ("train" ~1.9GB or "valid" ~19MB), downloading on first use."""
    path = DATA_DIR / f"TinyStories-{split}.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading {path.name} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(split=split), tmp)
        tmp.rename(path)
    return np.memmap(path, dtype=np.uint8, mode="r")


def get_batch(tokens: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator):
    """Random contiguous windows; `y` is `x` shifted left by one token."""
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    window = np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int32)
    return window[:, :-1], window[:, 1:]
