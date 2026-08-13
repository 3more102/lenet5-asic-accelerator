"""Bit-exact INT8 deployment forward pass for the full LeNet-5 network.

This chains the low-level INT8 kernels in golden/quantized_conv.py into the
full C1->S2->C3->S4->C5->F6->classifier sequence that rtl/lenet5_top.sv
implements. It is deliberately distinct from golden/lenet5.py's
forward_canonical, which models the 1998 paper's float64/scaled-tanh/RBF
path -- that canonical model is kept unmodified as a paper reference and is
not what the RTL is checked against. Every stage here uses ReLU (not
scaled-tanh) and the final classifier is an INT8 dense(84->10) + argmax
(not the paper's RBF/Euclidean-distance layer), per the deployment
decisions locked for this project.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from golden.lenet5 import c3_connectivity
from golden.quantized_conv import (
    argmax_classifier,
    avg_pool2x2_int8,
    conv2d_valid_int8,
    dense_int8,
)

DEFAULT_SHIFTS: Mapping[str, int] = {
    "c1": 7,
    "c3": 7,
    "c5": 7,
    "f6": 7,
}


def deploy_forward_int8(
    image: np.ndarray,
    params: Mapping[str, np.ndarray],
    *,
    shifts: Mapping[str, int] = DEFAULT_SHIFTS,
) -> dict[str, np.ndarray | int]:
    """Run the INT8/ReLU deployment forward pass.

    Args:
        image: int8 array shaped [1,32,32] or [32,32].
        params: mapping with c1_w/c1_b, c3_w/c3_b, c5_w/c5_b, f6_w/f6_b,
            cls_w/cls_b (see quantized_conv.conv2d_valid_int8/dense_int8 for
            each tensor's expected shape).
        shifts: per-layer right-shift amounts for c1/c3/c5/f6 requantization
            (the classifier compares raw accumulators and has no shift).

    Returns:
        Dict with every intermediate int8 tensor (C1, S2, C3, S4, C5, F6),
        the final classifier accumulators, and the predicted class index.
    """

    x = np.asarray(image, dtype=np.int8)
    if x.ndim == 2:
        x = x[None, :, :]
    if x.shape != (1, 32, 32):
        raise ValueError(
            "deploy_forward_int8 input must have shape [32,32] or [1,32,32]"
        )

    c1, _ = conv2d_valid_int8(
        x, params["c1_w"], params["c1_b"], shift=shifts["c1"], relu=True
    )
    s2 = avg_pool2x2_int8(c1)
    c3, _ = conv2d_valid_int8(
        s2,
        params["c3_w"],
        params["c3_b"],
        shift=shifts["c3"],
        relu=True,
        connectivity=c3_connectivity(),
    )
    s4 = avg_pool2x2_int8(c3)
    c5, _ = conv2d_valid_int8(
        s4, params["c5_w"], params["c5_b"], shift=shifts["c5"], relu=True
    )
    f6, _ = dense_int8(
        c5, params["f6_w"], params["f6_b"], shift=shifts["f6"], relu=True
    )
    prediction, cls_accumulators = argmax_classifier(
        f6, params["cls_w"], params["cls_b"]
    )

    return {
        "C1": c1,
        "S2": s2,
        "C3": c3,
        "S4": s4,
        "C5": c5,
        "F6": f6,
        "classifier_accumulators": cls_accumulators,
        "prediction": prediction,
    }


def random_deploy_parameters(seed: int = 11) -> dict[str, np.ndarray]:
    """Deterministic shape-correct int8 parameters for verification/demo use.

    These values are not trained and are not expected to classify MNIST.
    """

    rng = np.random.default_rng(seed)

    def w8(shape: tuple[int, ...], lo: int = -16, hi: int = 16) -> np.ndarray:
        return rng.integers(lo, hi, size=shape, dtype=np.int8)

    def b32(shape: tuple[int, ...], lo: int = -300, hi: int = 300) -> np.ndarray:
        return rng.integers(lo, hi, size=shape, dtype=np.int32)

    return {
        "c1_w": w8((6, 1, 5, 5)),
        "c1_b": b32((6,)),
        "c3_w": w8((16, 6, 5, 5)),
        "c3_b": b32((16,)),
        "c5_w": w8((120, 16, 5, 5)),
        "c5_b": b32((120,)),
        "f6_w": w8((84, 120)),
        "f6_b": b32((84,)),
        "cls_w": w8((10, 84)),
        "cls_b": b32((10,)),
    }
