"""Loader for the checked-in trained INT8 parameters.

`golden/train_lenet5.py` produces `golden/trained_params.npz` offline. This
module is the read side: it is what the vector generator, the unit tests and
any demo import, so nothing else needs to know the archive's key names or that
the shifts are stored as a parallel array rather than a dict.

The parameters here are a real network -- trained on MNIST, quantized to the
exact fixed-point scheme `rtl/requantize.sv` implements -- as opposed to
`golden.deploy.random_deploy_parameters`, which is deterministic, shape-correct
and deliberately meaningless. Both are needed: the random set is better
*stimulus* (it exercises accumulator ranges a trained network never visits),
while this set is the only one that can answer "does the accelerator actually
recognise a digit".
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

PARAM_FILE = Path(__file__).resolve().parent / "trained_params.npz"

PARAM_KEYS = (
    "c1_w", "c1_b",
    "c3_w", "c3_b",
    "c5_w", "c5_b",
    "f6_w", "f6_b",
    "cls_w", "cls_b",
)


class TrainedModelMissing(FileNotFoundError):
    """Raised with a pointer at how to regenerate the artifact."""


def _load() -> dict[str, np.ndarray]:
    if not PARAM_FILE.exists():
        raise TrainedModelMissing(
            f"{PARAM_FILE} is missing. It is a checked-in artifact; regenerate it "
            "with:  python -m golden.train_lenet5 --data path/to/mnist.npz"
        )
    with np.load(PARAM_FILE) as archive:
        return {key: archive[key] for key in archive.files}


def trained_parameters() -> dict[str, np.ndarray]:
    """The int8 weights and int32 biases, keyed as deploy_forward_int8 expects."""

    archive = _load()
    params = {key: archive[key] for key in PARAM_KEYS}
    for key in PARAM_KEYS:
        expected = np.int8 if key.endswith("_w") else np.int32
        if params[key].dtype != expected:
            raise ValueError(f"{key} has dtype {params[key].dtype}, expected {expected}")
    return params


def trained_shifts() -> dict[str, int]:
    """Per-layer calibrated right shifts, replacing deploy.DEFAULT_SHIFTS' flat 7."""

    archive = _load()
    names = [str(n) for n in archive["shift_layers"]]
    return {name: int(value) for name, value in zip(names, archive["shifts"])}


def trained_metadata() -> Mapping[str, float]:
    """Accuracy and provenance recorded at training time."""

    archive = _load()
    return {
        "float_accuracy": float(archive["float_accuracy"]),
        "int8_accuracy": float(archive["int8_accuracy"]),
        "seed": int(archive["seed"]),
        "epochs": int(archive["epochs"]),
        "percentile": float(archive["percentile"]),
    }


def quantize_image(images: np.ndarray) -> np.ndarray:
    """MNIST uint8 pixels -> the int8 the RTL is fed, scale 1/127.

    Kept here rather than only in the trainer because the vector generator and
    any future demo have to reproduce it exactly; a second, subtly different
    copy of this line would silently change what the hardware is being asked.
    golden/train_lenet5.py imports this rather than restating it.

    Dispatch is on dtype, not on the data. An earlier version scaled when
    `x.max() > 1.0`, which takes the wrong branch for a legitimately dark uint8
    image whose brightest pixel is 0 or 1 -- rare, silent, and it would change
    the image the hardware is fed rather than raise anything.
    """

    x = np.asarray(images)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / 255.0     # raw MNIST pixels, 0..255
    else:
        x = x.astype(np.float64)             # already normalised to 0..1
    return np.clip(np.rint(x * 127.0), -128, 127).astype(np.int8)


def pad_to_32(images: np.ndarray) -> np.ndarray:
    """28x28 -> 32x32 with a 2-pixel zero border, matching training."""

    x = np.asarray(images)
    if x.ndim == 2:
        x = x[None, :, :]
    if x.shape[1:] == (32, 32):
        return x
    if x.shape[1:] != (28, 28):
        raise ValueError(f"expected [N,28,28] or [N,32,32], got {x.shape}")
    out = np.zeros((x.shape[0], 32, 32), dtype=x.dtype)
    out[:, 2:30, 2:30] = x
    return out
