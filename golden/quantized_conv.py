"""Bit-exact signed-int8 operators used by the RTL verification flow."""

from __future__ import annotations

import numpy as np


def requantize(
    values: np.ndarray | int,
    shift: int,
    relu: bool = False,
) -> np.ndarray:
    """Power-of-two requantization matching rtl/requantize.sv.

    The result is rounded to nearest with ties away from zero, optionally
    clamped by ReLU, saturated to int8, and returned as an ndarray.
    """

    if not 0 <= shift <= 31:
        raise ValueError("shift must be in the range 0..31")

    x = np.asarray(values, dtype=np.int64)
    if shift:
        offset = np.int64(1 << (shift - 1))
        magnitude = np.abs(x)
        rounded_magnitude = (magnitude + offset) >> shift
        x = np.where(x < 0, -rounded_magnitude, rounded_magnitude)

    if relu:
        x = np.maximum(x, 0)
    return np.clip(x, -128, 127).astype(np.int8)


def conv2d_valid_int8(
    activations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    shift: int,
    relu: bool = False,
    connectivity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference valid convolution in CHW/OCHW layout.

    Args:
        activations: int8 array shaped [in_channels, height, width].
        weights: int8 array shaped [out_channels, in_channels, 5, 5].
        bias: int32-compatible array shaped [out_channels].
        shift: right shift used for output requantization.
        relu: enable output ReLU.
        connectivity: optional bool array [out_channels, in_channels].

    Returns:
        Tuple ``(int8_output, int64_accumulators)``. The second item is useful
        for debugging quantization and accumulator overflow.
    """

    x = np.asarray(activations, dtype=np.int8)
    w = np.asarray(weights, dtype=np.int8)
    b = np.asarray(bias, dtype=np.int64)

    if x.ndim != 3:
        raise ValueError("activations must have shape [C,H,W]")
    if w.ndim != 4 or w.shape[2:] != (5, 5):
        raise ValueError("weights must have shape [O,C,5,5]")
    if w.shape[1] != x.shape[0]:
        raise ValueError("weight and activation channel counts do not match")
    if b.shape != (w.shape[0],):
        raise ValueError("bias must have one value per output channel")
    if x.shape[1] < 5 or x.shape[2] < 5:
        raise ValueError("input height and width must both be at least 5")

    out_channels, in_channels = w.shape[:2]
    out_h = x.shape[1] - 4
    out_w = x.shape[2] - 4

    if connectivity is None:
        mask = np.ones((out_channels, in_channels), dtype=bool)
    else:
        mask = np.asarray(connectivity, dtype=bool)
        if mask.shape != (out_channels, in_channels):
            raise ValueError("connectivity must have shape [O,C]")

    accumulators = np.empty((out_channels, out_h, out_w), dtype=np.int64)
    for out_ch in range(out_channels):
        connected_channels = np.flatnonzero(mask[out_ch])
        for out_y in range(out_h):
            for out_x in range(out_w):
                acc = np.int64(b[out_ch])
                for in_ch in connected_channels:
                    patch = x[in_ch, out_y : out_y + 5, out_x : out_x + 5]
                    acc += np.sum(
                        patch.astype(np.int64) * w[out_ch, in_ch].astype(np.int64),
                        dtype=np.int64,
                    )
                accumulators[out_ch, out_y, out_x] = acc

    return requantize(accumulators, shift, relu), accumulators


def avg_pool2x2_int8(activations: np.ndarray) -> np.ndarray:
    """Non-overlapping 2x2 signed-int8 average pooling."""

    x = np.asarray(activations, dtype=np.int8)
    if x.ndim != 3 or x.shape[1] % 2 or x.shape[2] % 2:
        raise ValueError("input must be [C,H,W] with even H and W")

    channels, height, width = x.shape
    sums = x.astype(np.int64).reshape(
        channels, height // 2, 2, width // 2, 2
    ).sum(axis=(2, 4), dtype=np.int64)
    return requantize(sums, shift=2, relu=False)


def dense_int8(
    activations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    shift: int,
    relu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed-int8 dense layer with int64 reference accumulation."""

    x = np.asarray(activations, dtype=np.int8).reshape(-1)
    w = np.asarray(weights, dtype=np.int8)
    b = np.asarray(bias, dtype=np.int64)
    if w.ndim != 2 or w.shape[1] != x.size or b.shape != (w.shape[0],):
        raise ValueError("incompatible dense-layer shapes")
    accumulators = w.astype(np.int64) @ x.astype(np.int64) + b
    return requantize(accumulators, shift, relu), accumulators


def argmax_classifier(
    activations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Dense(N->num_classes) + argmax, matching rtl/classifier_argmax.sv.

    The class decision compares the raw int64 accumulators returned by
    dense_int8 (shift=0, relu=False, so nothing is rounded, clamped, or
    zeroed before the comparison) rather than dense_int8's requantized int8
    scores. requantize()'s int8 saturation clamp is not injective: with no
    per-layer calibration step yet in this project, two classes can both
    exceed +127 and collapse to the same clamped score, which would make an
    int8-score argmax pick an arbitrary winner among exactly the classes
    most likely to be correct. Comparing the pre-saturation accumulator
    avoids that failure mode entirely. Ties resolve to the lowest class
    index, matching np.argmax's default tie-break.
    """

    _, accumulators = dense_int8(activations, weights, bias, shift=0, relu=False)
    return int(np.argmax(accumulators)), accumulators

