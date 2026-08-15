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
from golden.quantized_conv import (
    argmax_classifier,
    avg_pool2x2_int8,
    conv2d_valid_int8,
    dense_int8,
    requantize,
)

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


def generate_requantize_vectors() -> str:
    """Directed + randomized differential vectors for tb_requantize.sv.

    Every other testbench exercises `requantize` only indirectly, and only on
    whatever accumulator values the fixed convolution vectors happen to
    produce at shift = 7. That leaves the parts most likely to be wrong
    untested: the exact half-way values the round-away-from-zero rule is
    defined by, both saturation boundaries, the shift = 0 bypass, and the
    extremes of the int32 accumulator. This sweeps them deliberately, then
    adds random cases across the whole (accumulator, shift) space.
    """

    cases: list[tuple[int, int, int]] = []  # (acc, shift, relu)

    int32_min, int32_max = -(1 << 31), (1 << 31) - 1

    def add(acc: int, shift: int) -> None:
        # The port is 32 bits wide, so anything outside that range would be
        # silently truncated on the way into the hex file while the expected
        # value here was computed from Python's unbounded integer -- which
        # would report a mismatch against correct RTL. Skip rather than lie.
        if not int32_min <= acc <= int32_max:
            return
        cases.append((acc, shift, 0))
        cases.append((acc, shift, 1))

    for shift in range(0, 32):
        # Half-way values: the tie cases the rounding rule is defined by. At
        # shift s the tie is at +/-2^(s-1), and it must round away from zero.
        if shift:
            half = 1 << (shift - 1)
            for base in (0, 1 << shift, 2 << shift):
                add(base + half, shift)
                add(-(base + half), shift)
                add(base + half - 1, shift)  # just below the tie
                add(base + half + 1, shift)  # just above it
        # Saturation boundaries, approached from both sides.
        for target in (-129, -128, -127, 126, 127, 128):
            add(target << shift, shift)
        # Accumulator extremes and small values.
        for acc in (0, 1, -1, int32_max, int32_min, int32_max - 1, int32_min + 1):
            add(acc, shift)

    # Worst-case real accumulation from the architecture: C5 is 16 channels x
    # 25 taps, and the largest-magnitude int8 product is (-128) * (-128) =
    # 16,384 rather than 127 * 127 = 16,129, because -128 has no positive
    # counterpart (docs/ARCHITECTURE.md). Bracketing 127^2 instead -- as this
    # did until 2026-08-15 -- leaves the true architectural worst case, and the
    # 102,000 above it, untested.
    worst = 16 * 25 * 128 * 128
    for shift in (0, 1, 7, 15, 31):
        for acc in (worst, -worst, worst + 1, -worst - 1):
            add(acc, shift)

    rng = np.random.default_rng(20260815)
    for _ in range(4000):
        acc = int(rng.integers(int32_min, int32_max, dtype=np.int64))
        shift = int(rng.integers(0, 32))
        cases.append((acc, shift, int(rng.integers(0, 2))))

    accs = np.array([c[0] for c in cases], dtype=np.int64)
    shifts = np.array([c[1] for c in cases], dtype=np.int64)
    relus = np.array([c[2] for c in cases], dtype=np.int64)

    expected = np.array(
        [
            int(requantize(np.int64(acc), int(shift), bool(relu)))
            for acc, shift, relu in zip(accs, shifts, relus)
        ],
        dtype=np.int64,
    )

    rq_dir = VECTOR_DIR / "requant"
    write_hex(rq_dir / "acc.hex", accs, 32)
    write_hex(rq_dir / "shift.hex", shifts, 8)
    write_hex(rq_dir / "relu.hex", relus, 8)
    write_hex(rq_dir / "expected.hex", expected, 8)

    print(f"requantize: {len(cases)} differential cases")

    return f"""// ---- tb_requantize.sv ----
`define TV_RQ_COUNT {len(cases)}
"""


def _extreme_weight_rows(out_ch: int, taps: int, split: int) -> np.ndarray:
    """Four deliberately chosen int8-extreme weight patterns, one per row.

    `split` is the index where the activation pattern flips from -128 to +127,
    so rows 0 and 1 can be built to drive the accumulator to its largest
    positive and largest negative value for this layer shape.
    """

    rows = np.zeros((out_ch, taps), dtype=np.int8)

    for k in range(taps):
        low = k < split
        # Row 0: every product positive -- (-128)(-128) then (+127)(+127).
        rows[0, k] = -128 if low else 127
        # Row 1: every product negative -- (-128)(+127) then (+127)(-128).
        rows[1, k] = 127 if low else -128
        # Row 2: alternating, so the running sum swings across zero rather
        # than marching monotonically to the extreme. A saturating or
        # wrongly-signed partial sum shows up here and not in rows 0/1.
        rows[2, k] = -128 if (k % 2 == 0) else 127

    # Row 3: a single isolated -128 tap against a -128 activation. The product
    # is +16384, which is the one case a multiplier that negates -128 (not
    # representable in int8) gets wrong while every other pattern still looks
    # plausible.
    rows[3, 0] = -128

    return rows


def generate_extremes_vectors() -> str:
    """Directed int8-extreme vectors for tb_extremes.sv.

    Every other engine-level vector set draws activations from [-32, 32) and
    weights from [-16, 16), so the largest accumulator any engine ever sees is
    around 39k of a 32-bit range, and the int8 extremes -128/+127 are never
    driven at all. Two things go untested as a result: the sign handling of
    -128, whose negation is not representable in int8 and which is the classic
    place a signed multiplier is wrong; and whether ACC_WIDTH is actually wide
    enough for the largest layer the engines accept.

    The shapes here are the worst case the design permits rather than the
    LeNet-5 layers: 16 input channels of 5x5 taps is 400 MACs per output, which
    is what C5 costs and the most conv2d_engine can be configured for at
    MAX_IN_CH=16.

    SHIFT is 16 rather than the deployment 6/7 on purpose. At shift 7 an
    accumulator of this size requantizes far past +-127 and every output
    saturates, so out_data_o would be pinned to the rail and completely
    insensitive to the accumulator underneath it -- the test would pass on
    almost any wrong answer. Shift 16 brings the extremes back inside int8, so
    the output is a faithful function of the full accumulator. Saturation
    itself is already covered exhaustively by tb_requantize.
    """

    IN_H = IN_W = 5           # smallest legal valid-conv input: 5x5 -> 1x1
    IN_CH, OUT_CH = 16, 2     # MAX_IN_CH; 400 MACs per output pixel
    SHIFT, RELU = 16, 0       # relu off so negative extremes stay visible
    TAPS = IN_CH * 25
    SPLIT = (IN_CH // 2) * 25

    # Only the two saturating patterns are used here, and that is deliberate.
    # At shift 16 one output LSB is worth 65,536 of accumulator, so a pattern
    # whose whole contribution is a few thousand requantizes to zero and would
    # keep requantizing to zero with its sign inverted -- the check would be
    # vacuous exactly where it looks most interesting. The sign-structure
    # patterns live in the dense case below instead, where out_acc_o is
    # compared directly and magnitude does not matter. Nothing is lost for
    # conv: rows 0 and 1 already drive (-128)(-128) and (-128)(+127) 200 times
    # each, so a multiplier that mishandles -128 moves the accumulator by
    # millions here, not by a rounding step.

    # Activations: the first half of the input channels are -128, the rest
    # +127, so every output sees both extremes and all four sign combinations
    # of the multiply are exercised within a single configuration.
    activations = np.empty((IN_CH, IN_H, IN_W), dtype=np.int8)
    activations[: IN_CH // 2] = -128
    activations[IN_CH // 2 :] = 127

    weights = _extreme_weight_rows(4, TAPS, SPLIT)[:OUT_CH].reshape(OUT_CH, IN_CH, 5, 5)

    # Large but deliberately safe: the worst-case MAC sum is 400 * 16384 =
    # 6,553,600, so +-1,000,000 of bias keeps the total inside int32 with room
    # to spare. Driving the accumulator past int32 would not be a bug found,
    # it would be an out-of-contract input -- the RTL wraps at 32 bits while
    # this oracle computes in Python's unbounded integers, so they would
    # disagree for a reason that is not a defect.
    bias = np.array([1_000_000, -1_000_000], dtype=np.int32)

    connectivity = np.ones((OUT_CH, IN_CH), dtype=bool)

    expected, accumulators = conv2d_valid_int8(
        activations, weights, bias, shift=SHIFT, relu=bool(RELU), connectivity=connectivity
    )

    ex_dir = VECTOR_DIR / "extremes"
    write_hex(ex_dir / "act.hex", activations, 8)
    write_hex(ex_dir / "wgt.hex", weights, 8)
    write_hex(ex_dir / "bias.hex", bias, 32)
    write_hex(ex_dir / "conn.hex", connectivity.astype(np.uint8), 8)
    write_hex(ex_dir / "expected.hex", expected, 8)
    write_hex(ex_dir / "accumulator.hex", accumulators, 64)

    # ---- extreme operands, cancelling sum, shift 0 ------------------------
    # The case above proves the accumulator carries millions without wrapping,
    # but it cannot see a *small* arithmetic error: at shift 16 one output LSB
    # is worth 65,536 of accumulator, so a mutation that perturbs a few taps by
    # a few thousand requantizes to exactly the same byte. (Measured, not
    # assumed -- a multiplier that reads -128 as -127 moves this accumulator by
    # 5,120 and the output not at all.)
    #
    # So the same extreme operands are run again with weights chosen to cancel
    # to exactly zero, leaving the bias as the whole accumulator. At shift 0
    # the output *is* the accumulator, so a single wrong product moves it by at
    # least 128 and a sign-flipped one drives it into the rail. Between the two
    # runs, conv2d_engine is checked for both magnitude and resolution.
    #
    # The weights must cancel, but they must NOT cancel symmetrically with
    # respect to position in the 5-tap row. conv5x5_row_mac computes five
    # products per row and a bug in one of them perturbs every affected tap by
    # an amount proportional to that tap's weight. A first attempt here used
    # three equal contiguous blocks of 50, which put an equal share of each
    # weight group in every row column -- so a mutation confined to column 0
    # summed to exactly zero and the pass stayed green on broken RTL. Measured,
    # not theorised: that is precisely what happened.
    #
    # So the cancellation is arranged *across* the row columns. Each of the
    # five tap positions in a row gets its own weight value, chosen to sum to
    # zero over the five:
    #     -128 + 127 + 1 - 1 + 1 = 0
    # Every tap in the layer sees the same activation, so the products sum to
    # activation * sum-of-weights = 0 exactly, while no individual column sums
    # to zero. A bug confined to any one of the five products therefore shifts
    # the accumulator, and at shift 0 a shift of even 1 fails the comparison.
    # Leaving a column at zero would reopen the hole: a wrong product times a
    # zero weight is still zero.
    COL_W_LOW  = [-128, 127, 1, -1, 1]   # against the -128 activations
    COL_W_HIGH = [127, -128, 1, -1, 1]   # against the +127 activations

    cancel_w = np.zeros((OUT_CH, TAPS), dtype=np.int8)
    for k in range(SPLIT):
        cancel_w[0, k] = COL_W_LOW[k % 5]
    for k in range(SPLIT, TAPS):
        cancel_w[1, k] = COL_W_HIGH[k % 5]

    cancel_bias = np.array([100, -100], dtype=np.int32)
    cancel_weights = cancel_w.reshape(OUT_CH, IN_CH, 5, 5)

    cancel_expected, cancel_acc = conv2d_valid_int8(
        activations, cancel_weights, cancel_bias, shift=0, relu=False,
        connectivity=connectivity,
    )

    # If this ever stops holding the run is no longer a resolution test, so
    # fail here rather than emit vectors that quietly prove less than claimed.
    if list(np.asarray(cancel_acc).reshape(-1)) != list(cancel_bias):
        raise AssertionError(
            f"cancelling weights did not cancel: accumulators {cancel_acc} "
            f"should equal the bias {cancel_bias}"
        )

    cn_dir = VECTOR_DIR / "extremes" / "cancel"
    write_hex(cn_dir / "wgt.hex", cancel_weights, 8)
    write_hex(cn_dir / "bias.hex", cancel_bias, 32)
    write_hex(cn_dir / "expected.hex", cancel_expected, 8)
    write_hex(cn_dir / "accumulator.hex", cancel_acc, 64)

    # ---- minimum legal configuration -------------------------------------
    # One input channel, one output channel, 5x5 -> 1x1: a single MAC group
    # and a single output beat. Nested loop counters that are off by one at
    # the bottom of their range have nowhere to hide in a one-beat stream,
    # and no other testbench runs a shape this small.
    min_act = np.full((1, 5, 5), -128, dtype=np.int8)
    min_wgt = np.full((1, 1, 5, 5), -128, dtype=np.int8)
    min_bias = np.array([-16384], dtype=np.int32)
    min_conn = np.ones((1, 1), dtype=bool)

    # Shift 12 for the same reason the main case uses 16: this accumulator is
    # 25 * 16384 - 16384 = 393,216, which at any small shift saturates and
    # leaves out_data_o pinned at +127 regardless of what the engine actually
    # computed. 393,216 >> 12 = 96, comfortably inside int8.
    min_expected, min_accumulators = conv2d_valid_int8(
        min_act, min_wgt, min_bias, shift=12, relu=False, connectivity=min_conn
    )

    mn_dir = VECTOR_DIR / "extremes" / "min"
    write_hex(mn_dir / "act.hex", min_act, 8)
    write_hex(mn_dir / "wgt.hex", min_wgt, 8)
    write_hex(mn_dir / "bias.hex", min_bias, 32)
    write_hex(mn_dir / "conn.hex", min_conn.astype(np.uint8), 8)
    write_hex(mn_dir / "expected.hex", min_expected, 8)
    write_hex(mn_dir / "accumulator.hex", min_accumulators, 64)

    # ---- dense engine at its maximum input length ------------------------
    D_IN_LEN, D_OUT_LEN = 120, 4      # MAX_IN_LEN
    D_SHIFT, D_RELU = 16, 0
    D_SPLIT = D_IN_LEN // 2

    d_act = np.empty((D_IN_LEN,), dtype=np.int8)
    d_act[:D_SPLIT] = -128
    d_act[D_SPLIT:] = 127

    d_wgt = _extreme_weight_rows(D_OUT_LEN, D_IN_LEN, D_SPLIT)
    d_bias = np.array([1_000_000, -1_000_000, 0, 1], dtype=np.int32)

    d_expected, d_accumulators = dense_int8(
        d_act, d_wgt, d_bias, shift=D_SHIFT, relu=bool(D_RELU)
    )

    dn_dir = VECTOR_DIR / "extremes" / "dense"
    write_hex(dn_dir / "act.hex", d_act, 8)
    write_hex(dn_dir / "wgt.hex", d_wgt, 8)
    write_hex(dn_dir / "bias.hex", d_bias, 32)
    write_hex(dn_dir / "expected.hex", d_expected, 8)
    write_hex(dn_dir / "accumulator.hex", d_accumulators, 64)

    peak = int(np.max(np.abs(np.asarray(accumulators, dtype=np.int64))))
    d_peak = int(np.max(np.abs(np.asarray(d_accumulators, dtype=np.int64))))
    print(
        f"extremes: conv {IN_CH}x{IN_H}x{IN_W} -> {OUT_CH}x1x1 "
        f"(peak |acc| {peak:,}), dense {D_IN_LEN} -> {D_OUT_LEN} "
        f"(peak |acc| {d_peak:,})"
    )

    return f"""// ---- tb_extremes.sv (int8-extreme operands at maximum layer size) ----
`define TV_EX_IN_H {IN_H}
`define TV_EX_IN_W {IN_W}
`define TV_EX_IN_CH {IN_CH}
`define TV_EX_OUT_CH {OUT_CH}
`define TV_EX_SHIFT {SHIFT}
`define TV_EX_RELU 1'b{RELU}
`define TV_EX_ACT_COUNT {activations.size}
`define TV_EX_WGT_COUNT {weights.size}
`define TV_EX_BIAS_COUNT {bias.size}
`define TV_EX_CONN_COUNT {connectivity.size}
`define TV_EX_OUT_COUNT {expected.size}
`define TV_EX_PEAK_ACC {peak}

// Same DUT and same activations, weights cancelling to zero, shift 0, so the
// output is the accumulator itself and one wrong product is visible.
`define TV_EXC_SHIFT 0
`define TV_EXC_WGT_COUNT {cancel_weights.size}
`define TV_EXC_BIAS_COUNT {cancel_bias.size}
`define TV_EXC_OUT_COUNT {cancel_expected.size}

// Minimum legal conv configuration: 1 channel, 5x5 -> 1x1, one output beat.
`define TV_EXMIN_IN_H 5
`define TV_EXMIN_IN_W 5
`define TV_EXMIN_IN_CH 1
`define TV_EXMIN_OUT_CH 1
`define TV_EXMIN_SHIFT 12
`define TV_EXMIN_RELU 1'b0
`define TV_EXMIN_ACT_COUNT {min_act.size}
`define TV_EXMIN_WGT_COUNT {min_wgt.size}
`define TV_EXMIN_BIAS_COUNT {min_bias.size}
`define TV_EXMIN_CONN_COUNT {min_conn.size}
`define TV_EXMIN_OUT_COUNT {min_expected.size}

// ---- tb_extremes.sv dense case ----
`define TV_EXD_IN_LEN {D_IN_LEN}
`define TV_EXD_OUT_LEN {D_OUT_LEN}
`define TV_EXD_SHIFT {D_SHIFT}
`define TV_EXD_RELU 1'b{D_RELU}
`define TV_EXD_ACT_COUNT {d_act.size}
`define TV_EXD_WGT_COUNT {d_wgt.size}
`define TV_EXD_BIAS_COUNT {d_bias.size}
`define TV_EXD_OUT_COUNT {d_expected.size}
`define TV_EXD_PEAK_ACC {d_peak}
"""


def main() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    blocks = [
        generate_conv2d_engine_vectors(),
        generate_pool_vectors(),
        generate_f6_vectors(),
        generate_classifier_vectors(),
        generate_requantize_vectors(),
        generate_extremes_vectors(),
        generate_top_vectors(),
    ]

    header = "// Generated by golden/generate_vectors.py; do not hand edit.\n"
    (VECTOR_DIR / "config.svh").write_text(header + "\n".join(blocks), encoding="ascii")


if __name__ == "__main__":
    main()
