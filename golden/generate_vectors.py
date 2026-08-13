#!/usr/bin/env python3
"""Generate deterministic bit-exact vectors for the tb/*.sv testbenches.

Each generate_*_vectors() function owns one module's vector set and one
namespaced block of `TV_*` macros in the single regenerated
vectors/config.svh. The original (first) module's macros stay unprefixed
(tb_conv2d_engine.sv's working, already-passing contract) -- new blocks are
namespaced (TV_POOL_*, TV_F6_*, TV_CLS_*, TV_TOP_*) to avoid collisions.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golden.deploy import DEFAULT_SHIFTS, deploy_forward_int8, random_deploy_parameters
from golden.quantized_conv import argmax_classifier, avg_pool2x2_int8, conv2d_valid_int8, dense_int8

VECTOR_DIR = ROOT / "vectors"


def write_hex(path: Path, values: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    width = bits // 4
    flat = np.asarray(values).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(value) & mask:0{width}x}\n" for value in flat),
        encoding="ascii",
    )


def generate_conv2d_engine_vectors() -> str:
    """Original tb_conv2d_engine.sv case; output/macros unchanged."""

    IN_H, IN_W, IN_CH, OUT_CH = 7, 8, 3, 4
    SHIFT, RELU, SEED = 7, 1, 2026

    rng = np.random.default_rng(SEED)
    activations = rng.integers(-32, 32, size=(IN_CH, IN_H, IN_W), dtype=np.int8)
    weights = rng.integers(-16, 16, size=(OUT_CH, IN_CH, 5, 5), dtype=np.int8)
    bias = rng.integers(-300, 300, size=(OUT_CH,), dtype=np.int32)
    connectivity = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )

    expected, accumulators = conv2d_valid_int8(
        activations, weights, bias, shift=SHIFT, relu=bool(RELU), connectivity=connectivity
    )

    write_hex(VECTOR_DIR / "act.hex", activations, 8)
    write_hex(VECTOR_DIR / "wgt.hex", weights, 8)
    write_hex(VECTOR_DIR / "bias.hex", bias, 32)
    write_hex(VECTOR_DIR / "conn.hex", connectivity.astype(np.uint8), 8)
    write_hex(VECTOR_DIR / "expected.hex", expected, 8)
    write_hex(VECTOR_DIR / "accumulator.hex", accumulators, 64)

    out_h, out_w = IN_H - 4, IN_W - 4
    print(f"conv2d_engine: {IN_CH}x{IN_H}x{IN_W} -> {OUT_CH}x{out_h}x{out_w}")

    return f"""// ---- tb_conv2d_engine.sv (first/legacy module; macros unprefixed) ----
`define TV_IN_H {IN_H}
`define TV_IN_W {IN_W}
`define TV_IN_CH {IN_CH}
`define TV_OUT_CH {OUT_CH}
`define TV_OUT_H {out_h}
`define TV_OUT_W {out_w}
`define TV_SHIFT {SHIFT}
`define TV_RELU 1'b{RELU}
`define TV_ACT_COUNT {activations.size}
`define TV_WGT_COUNT {weights.size}
`define TV_BIAS_COUNT {bias.size}
`define TV_CONN_COUNT {connectivity.size}
`define TV_OUT_COUNT {expected.size}
"""


def generate_pool_vectors() -> str:
    """tb_avg_pool2x2_stream.sv case."""

    IN_H, IN_W, IN_CH, SEED = 4, 6, 2, 3001

    rng = np.random.default_rng(SEED)
    activations = rng.integers(-32, 32, size=(IN_CH, IN_H, IN_W), dtype=np.int8)
    expected = avg_pool2x2_int8(activations)

    write_hex(VECTOR_DIR / "pool" / "act.hex", activations, 8)
    write_hex(VECTOR_DIR / "pool" / "expected.hex", expected, 8)

    out_h, out_w = IN_H // 2, IN_W // 2
    print(f"pool: {IN_CH}x{IN_H}x{IN_W} -> {IN_CH}x{out_h}x{out_w}")

    return f"""// ---- tb_avg_pool2x2_stream.sv ----
`define TV_POOL_IN_H {IN_H}
`define TV_POOL_IN_W {IN_W}
`define TV_POOL_IN_CH {IN_CH}
`define TV_POOL_OUT_H {out_h}
`define TV_POOL_OUT_W {out_w}
`define TV_POOL_ACT_COUNT {activations.size}
`define TV_POOL_OUT_COUNT {expected.size}
"""


def generate_f6_vectors() -> str:
    """tb_dense_engine.sv case (used to verify the F6 configuration)."""

    IN_LEN, OUT_LEN = 10, 5
    SHIFT, RELU, SEED = 6, 1, 3002

    rng = np.random.default_rng(SEED)
    activations = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)
    weights = rng.integers(-16, 16, size=(OUT_LEN, IN_LEN), dtype=np.int8)
    bias = rng.integers(-300, 300, size=(OUT_LEN,), dtype=np.int32)

    expected, accumulators = dense_int8(
        activations, weights, bias, shift=SHIFT, relu=bool(RELU)
    )

    write_hex(VECTOR_DIR / "f6" / "act.hex", activations, 8)
    write_hex(VECTOR_DIR / "f6" / "wgt.hex", weights, 8)
    write_hex(VECTOR_DIR / "f6" / "bias.hex", bias, 32)
    write_hex(VECTOR_DIR / "f6" / "expected.hex", expected, 8)
    write_hex(VECTOR_DIR / "f6" / "accumulator.hex", accumulators, 64)

    print(f"dense_engine (F6 config): {IN_LEN} -> {OUT_LEN}")

    return f"""// ---- tb_dense_engine.sv (F6 configuration) ----
`define TV_F6_IN_LEN {IN_LEN}
`define TV_F6_OUT_LEN {OUT_LEN}
`define TV_F6_SHIFT {SHIFT}
`define TV_F6_RELU 1'b{RELU}
`define TV_F6_ACT_COUNT {activations.size}
`define TV_F6_WGT_COUNT {weights.size}
`define TV_F6_BIAS_COUNT {bias.size}
`define TV_F6_OUT_COUNT {expected.size}
"""


def generate_classifier_vectors() -> str:
    """tb_classifier_argmax.sv case.

    classifier_argmax.sv does not expose its internal dense_engine's
    per-class scores externally (only the final class_o/valid_o), so only
    the final class index is a checkable expected value here.
    """

    IN_LEN, NUM_CLASSES = 6, 4
    SEED = 3003

    rng = np.random.default_rng(SEED)
    activations = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)
    weights = rng.integers(-16, 16, size=(NUM_CLASSES, IN_LEN), dtype=np.int8)
    bias = rng.integers(-300, 300, size=(NUM_CLASSES,), dtype=np.int32)

    winner, _ = argmax_classifier(activations, weights, bias)

    write_hex(VECTOR_DIR / "classifier" / "act.hex", activations, 8)
    write_hex(VECTOR_DIR / "classifier" / "wgt.hex", weights, 8)
    write_hex(VECTOR_DIR / "classifier" / "bias.hex", bias, 32)

    print(f"classifier_argmax: {IN_LEN} -> {NUM_CLASSES} classes, winner={winner}")

    return f"""// ---- tb_classifier_argmax.sv ----
`define TV_CLS_IN_LEN {IN_LEN}
`define TV_CLS_NUM_CLASSES {NUM_CLASSES}
`define TV_CLS_ACT_COUNT {activations.size}
`define TV_CLS_WGT_COUNT {weights.size}
`define TV_CLS_BIAS_COUNT {bias.size}
`define TV_CLS_EXPECTED_CLASS {winner}
"""


def generate_top_vectors() -> str:
    """tb_lenet5_top.sv case.

    The C1->S2->C3->S4->C5 chain only collapses to a 1x1 C5 output at
    exactly the canonical proportions (fixed 5x5/2x2 stage geometry), so
    this vector set is forced to the full canonical 32x32x1 input -- there
    is no smaller synthetic shape that stays valid end to end.
    """

    SEED = 3004
    WATCHDOG_CYCLES = 600_000

    rng = np.random.default_rng(SEED)
    image = rng.integers(-16, 16, size=(32, 32), dtype=np.int8)
    params = random_deploy_parameters(seed=SEED)

    result = deploy_forward_int8(image, params, shifts=DEFAULT_SHIFTS)
    expected_class = result["prediction"]

    top_dir = VECTOR_DIR / "top"
    write_hex(top_dir / "image.hex", image, 8)
    write_hex(top_dir / "c1_wgt.hex", params["c1_w"], 8)
    write_hex(top_dir / "c1_bias.hex", params["c1_b"], 32)
    write_hex(top_dir / "c3_wgt.hex", params["c3_w"], 8)
    write_hex(top_dir / "c3_bias.hex", params["c3_b"], 32)
    write_hex(top_dir / "c5_wgt.hex", params["c5_w"], 8)
    write_hex(top_dir / "c5_bias.hex", params["c5_b"], 32)
    write_hex(top_dir / "f6_wgt.hex", params["f6_w"], 8)
    write_hex(top_dir / "f6_bias.hex", params["f6_b"], 32)
    write_hex(top_dir / "cls_wgt.hex", params["cls_w"], 8)
    write_hex(top_dir / "cls_bias.hex", params["cls_b"], 32)

    print(f"lenet5_top: 1x32x32 -> class {expected_class}")

    return f"""// ---- tb_lenet5_top.sv ----
`define TV_TOP_EXPECTED_CLASS {expected_class}
`define TV_TOP_WATCHDOG_CYCLES {WATCHDOG_CYCLES}
"""


def main() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    blocks = [
        generate_conv2d_engine_vectors(),
        generate_pool_vectors(),
        generate_f6_vectors(),
        generate_classifier_vectors(),
        generate_top_vectors(),
    ]

    header = "// Generated by golden/generate_vectors.py; do not hand edit.\n"
    (VECTOR_DIR / "config.svh").write_text(header + "\n".join(blocks), encoding="ascii")


if __name__ == "__main__":
    main()
