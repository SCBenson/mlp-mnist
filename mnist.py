import gzip
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "mnist"


def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def load_mnist():
    """Return (x_train, y_train, x_test, y_test).

    Images are uint8 arrays of shape (N, 28, 28) with values 0-255.
    Labels are uint8 arrays of shape (N,) with values 0-9.
    """
    x_train = _read_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    y_train = _read_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    x_test = _read_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    y_test = _read_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test
