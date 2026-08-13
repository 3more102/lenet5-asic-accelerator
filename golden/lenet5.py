"""Canonical LeNet-5 topology and a NumPy floating-point golden model.

This module models the 1998 paper rather than the commonly used modern
"LeNet-like" network. In particular, it includes C3's sparse connectivity,
trainable S2/S4 subsampling coefficients, scaled tanh activations, a
convolutional C5 layer, F6=84, and a Euclidean RBF output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def c3_connectivity() -> np.ndarray:
    """Return the exact 16x6 boolean C3 connection matrix."""

    channel_sets = (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 0),
        (5, 0, 1),
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 0),
        (4, 5, 0, 1),
        (5, 0, 1, 2),
        (0, 1, 3, 4),
        (1, 2, 4, 5),
        (0, 2, 3, 5),
        (0, 1, 2, 3, 4, 5),
    )
    matrix = np.zeros((16, 6), dtype=bool)
    for output_map, inputs in enumerate(channel_sets):
        matrix[output_map, list(inputs)] = True
    return matrix


@dataclass(frozen=True)
class LayerSpec:
    name: str
    output_shape: tuple[int, ...]
    trainable_parameters: int


CANONICAL_LAYERS = (
    LayerSpec("Input", (1, 32, 32), 0),
    LayerSpec("C1", (6, 28, 28), 156),
    LayerSpec("S2", (6, 14, 14), 12),
    LayerSpec("C3", (16, 10, 10), 1516),
    LayerSpec("S4", (16, 5, 5), 32),
    LayerSpec("C5", (120, 1, 1), 48120),
    LayerSpec("F6", (84,), 10164),
    LayerSpec("RBF output", (10,), 0),
)


def scaled_tanh(values: np.ndarray) -> np.ndarray:
    """LeNet-5 squashing function A*tanh(S*a), A=1.7159, S=2/3."""

    return 1.7159 * np.tanh((2.0 / 3.0) * values)


def conv2d_valid_float(
    activations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    connectivity: np.ndarray | None = None,
) -> np.ndarray:
    """Valid CHW/OCHW convolution using float64 accumulation."""

    x = np.asarray(activations, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if x.ndim != 3 or w.ndim != 4 or w.shape[1] != x.shape[0]:
        raise ValueError("expected activations [C,H,W] and weights [O,C,KH,KW]")
    if b.shape != (w.shape[0],):
        raise ValueError("bias must have one value per output channel")

    if connectivity is not None:
        mask = np.asarray(connectivity, dtype=bool)
        if mask.shape != w.shape[:2]:
            raise ValueError("connectivity must have shape [O,C]")
        w = w * mask[:, :, None, None]

    kernel_h, kernel_w = w.shape[2:]
    patches = np.lib.stride_tricks.sliding_window_view(
        x, (kernel_h, kernel_w), axis=(1, 2)
    )
    output = np.einsum("cxykl,ockl->oxy", patches, w, optimize=True)
    return output + b[:, None, None]


def trainable_subsample(
    activations: np.ndarray,
    coefficient: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Original non-overlapping 2x2 sum, coefficient, bias, and tanh."""

    x = np.asarray(activations, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    channels, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("subsampling input dimensions must be even")
    if coefficient.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("subsampling needs one coefficient and bias per map")

    sums = x.reshape(channels, height // 2, 2, width // 2, 2).sum(axis=(2, 4))
    return scaled_tanh(
        sums * coefficient[:, None, None] + bias[:, None, None]
    )


def forward_canonical(
    image: np.ndarray,
    parameters: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | int]:
    """Run the canonical floating-point LeNet-5 forward pass."""

    x = np.asarray(image, dtype=np.float64)
    if x.shape == (32, 32):
        x = x[None, :, :]
    if x.shape != (1, 32, 32):
        raise ValueError("LeNet-5 input must have shape [32,32] or [1,32,32]")

    c1 = scaled_tanh(
        conv2d_valid_float(x, parameters["c1_w"], parameters["c1_b"])
    )
    s2 = trainable_subsample(c1, parameters["s2_coeff"], parameters["s2_b"])
    c3 = scaled_tanh(
        conv2d_valid_float(
            s2,
            parameters["c3_w"],
            parameters["c3_b"],
            c3_connectivity(),
        )
    )
    s4 = trainable_subsample(c3, parameters["s4_coeff"], parameters["s4_b"])
    c5 = scaled_tanh(
        conv2d_valid_float(s4, parameters["c5_w"], parameters["c5_b"])
    )
    f6 = scaled_tanh(
        parameters["f6_w"] @ c5.reshape(-1) + parameters["f6_b"]
    )
    rbf_codes = np.asarray(parameters["rbf_codes"], dtype=np.float64)
    scores = np.sum((f6[None, :] - rbf_codes) ** 2, axis=1)

    return {
        "C1": c1,
        "S2": s2,
        "C3": c3,
        "S4": s4,
        "C5": c5,
        "F6": f6,
        "scores": scores,
        "prediction": int(np.argmin(scores)),
    }


def random_canonical_parameters(seed: int = 5) -> dict[str, np.ndarray]:
    """Create deterministic shape-correct parameters for verification demos.

    These values are not trained and are not expected to classify MNIST.
    """

    rng = np.random.default_rng(seed)

    def weights(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(0.0, 0.05, size=shape).astype(np.float64)

    return {
        "c1_w": weights((6, 1, 5, 5)),
        "c1_b": weights((6,)),
        "s2_coeff": np.full(6, 0.25, dtype=np.float64),
        "s2_b": weights((6,)),
        "c3_w": weights((16, 6, 5, 5)),
        "c3_b": weights((16,)),
        "s4_coeff": np.full(16, 0.25, dtype=np.float64),
        "s4_b": weights((16,)),
        "c5_w": weights((120, 16, 5, 5)),
        "c5_b": weights((120,)),
        "f6_w": weights((84, 120)),
        "f6_b": weights((84,)),
        # Placeholder fixed codes for datapath/shape verification. Replace
        # these with the desired trained or hand-designed 7x12 class codes.
        "rbf_codes": rng.choice((-1.0, 1.0), size=(10, 84)),
    }

