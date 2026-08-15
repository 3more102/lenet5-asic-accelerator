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


# Background operands for the lane sweeps below. A sweep of one lane has to
# leave the other lanes holding *something*, and zero would be the worst
# possible choice: with the rest of the row zeroed, a design that dropped a
# lane, duplicated one, or paired lane i's activation with lane j's weight
# would produce exactly the same sum as a correct one. These are non-zero, of
# mixed sign, and chosen so the eight lane products (87, -155, -259, 451, 559,
# 799, 1007, -1357) all have distinct magnitudes -- so removing any single lane
# from the sum lands on a different value than removing any other.
_MAC_BG_ACT = (3, -5, 7, -11, 13, -17, 19, -23)
_MAC_BG_WGT = (29, 31, -37, -41, 43, -47, 53, 59)

# The int8 values a MAC lane is most likely to be wrong on: both ends of the
# range, the neighbours of both ends, both units, and zero. -128 matters twice
# over -- it has no positive counterpart, so (-128)*(-128) = 16,384 is the
# largest-magnitude int8 product and the one a 127-based bound misses.
_MAC_CORNERS = (-128, -127, -1, 0, 1, 126, 127)


def _pack_lanes(values) -> int:
    """Pack per-lane int8 values into one word, lane 0 in the low byte."""

    word = 0
    for index, value in enumerate(values):
        word |= (int(value) & 0xFF) << (8 * index)
    return word


def _row_mac_reference(acts: list[list[int]], wgts: list[list[int]]) -> np.ndarray:
    """Expected lane sum for each (activation row, weight row) pair.

    Taken from golden/quantized_conv.py's dense_int8 accumulator output rather
    than recomputed here. One lane group of a dense layer with zero bias *is* a
    row MAC, so this reuses the same model every other testbench in the suite
    is checked against, instead of standing up a second implementation of the
    arithmetic the DUT is being compared against.
    """

    zero_bias = np.zeros(1, dtype=np.int64)
    sums = []
    for act_row, wgt_row in zip(acts, wgts):
        _, accumulator = dense_int8(
            np.array(act_row, dtype=np.int8),
            np.array(wgt_row, dtype=np.int8).reshape(1, -1),
            zero_bias,
            shift=0,
        )
        sums.append(int(accumulator[0]))
    return np.array(sums, dtype=np.int64)


def _mac_stimulus(lanes: int, seed: int, random_count: int):
    """Directed then randomized (activation, weight) rows for an N-lane MAC."""

    acts: list[list[int]] = []
    wgts: list[list[int]] = []
    background_act = list(_MAC_BG_ACT[:lanes])
    background_wgt = list(_MAC_BG_WGT[:lanes])

    def add(act_row, wgt_row) -> None:
        acts.append(list(act_row))
        wgts.append(list(wgt_row))

    # 1. Per-lane full int8 sweep, one lane at a time against the background.
    #    Every lane is swept on both operands, so a lane that is stuck, tied
    #    off, sign-extended wrongly or missing entirely cannot hide behind the
    #    other four (or seven) carrying the sum.
    for lane in range(lanes):
        for value in range(-128, 128):
            act_row = list(background_act)
            act_row[lane] = value
            add(act_row, background_wgt)

            wgt_row = list(background_wgt)
            wgt_row[lane] = value
            add(background_act, wgt_row)

    # 2. Per-lane corner grid: both operands of one lane at their extremes
    #    simultaneously, which the single-operand sweeps above never reach.
    for lane in range(lanes):
        for act_value in _MAC_CORNERS:
            for wgt_value in _MAC_CORNERS:
                act_row = list(background_act)
                wgt_row = list(background_wgt)
                act_row[lane] = act_value
                wgt_row[lane] = wgt_value
                add(act_row, wgt_row)

    # 3. Weight rotation against fixed activations. Rotating *both* rows would
    #    leave the multiset of products unchanged and so leave the sum
    #    unchanged -- a test that passes on a mis-wired lane pairing. Rotating
    #    the weights alone re-pairs every lane, so lane i's activation must
    #    meet lane i's weight to reproduce these sums.
    rotation_bases = [
        ([1, 2, 4, 8, 16, 32, 64, 127][:lanes], [1, 3, 9, 27, 81, 121, 5, 17][:lanes]),
        ([-128, 127, -64, 63, -32, 31, -16, 15][:lanes], [2, -3, 5, -7, 11, -13, 17, -19][:lanes]),
    ]
    for base_act, base_wgt in rotation_bases:
        for rotation in range(lanes):
            add(base_act, base_wgt[rotation:] + base_wgt[:rotation])

    # 4. Whole-row extremes: the four corners of the sum's range, plus one lane
    #    driven to an extreme while the rest hold the background, which is the
    #    case that tells a lane-local sign-extension bug from a tree-wide one.
    for act_value, wgt_value in (
        (-128, -128),  # +16,384 per lane -- the largest positive sum
        (-128, 127),   # -16,256 per lane -- the largest negative sum
        (127, 127),
        (127, -128),
        (-128, 1),
        (1, -128),
    ):
        add([act_value] * lanes, [wgt_value] * lanes)

    for lane in range(lanes):
        for act_value, wgt_value in ((-128, -128), (-128, 127), (127, 127)):
            act_row = list(background_act)
            wgt_row = list(background_wgt)
            act_row[lane] = act_value
            wgt_row[lane] = wgt_value
            add(act_row, wgt_row)

    # 5. Products that cancel to exactly zero, with a distinct magnitude in
    #    every lane. A zero sum is where a wrong sign, a lost carry or a
    #    dropped lane shows up most plainly -- the correct answer has no bits
    #    set, so any error is the whole output. But that only holds if the
    #    cancellation is asymmetric: a row whose lanes all carry the same
    #    product still sums correctly under a bug that swaps or duplicates
    #    them, and a lane left at zero is a lane not being tested at all. Each
    #    set below is written as explicit (activation, weight) pairs per lane
    #    and cannot be built by truncating a longer one, since dropping a lane
    #    would destroy the cancellation. Set A anchors on the two extreme
    #    products, (-128)*(-128) = +16,384 and 127*(-128) = -16,256.
    cancelling_pairs = {
        5: [
            [(-128, -128), (127, -128), (10, -10), (4, -5), (2, -4)],
            [(25, 40), (-25, 32), (-15, 10), (12, -5), (2, 5)],
        ],
        8: [
            [(-128, -128), (127, -128), (10, -10), (4, -5), (2, -4),
             (9, 7), (3, -7), (6, -7)],
            [(25, 40), (-25, 32), (-15, 10), (12, -5), (2, 5),
             (11, 11), (-11, 10), (-1, 11)],
        ],
    }
    for pairs in cancelling_pairs[lanes]:
        act_row = [pair[0] for pair in pairs]
        wgt_row = [pair[1] for pair in pairs]
        products = [a * w for a, w in zip(act_row, wgt_row)]
        # Assert the two properties the set is here for, rather than trusting
        # the arithmetic above to stay right through an edit.
        if sum(products) != 0:
            raise ValueError(f"cancelling set for {lanes} lanes sums to {sum(products)}")
        if 0 in products or len(set(abs(p) for p in products)) != lanes:
            raise ValueError(f"cancelling set for {lanes} lanes has a zero or repeated product")
        add(act_row, wgt_row)
        add(wgt_row, act_row)
        # The same products re-paired across lanes: still zero if the lanes are
        # summed, but a different set of per-lane values reaching each
        # multiplier.
        add(act_row[::-1], wgt_row[::-1])

    # 6. Uniform random over the whole int8 x int8 lane space, plus a mixed
    #    distribution that pins half the lanes to extremes and leaves the rest
    #    small -- uniform sampling almost never produces that shape, and it is
    #    the shape a real quantized layer produces after ReLU.
    rng = np.random.default_rng(seed)
    for _ in range(random_count):
        add(
            rng.integers(-128, 128, size=lanes).tolist(),
            rng.integers(-128, 128, size=lanes).tolist(),
        )
    for _ in range(random_count // 4):
        act_row = rng.integers(-128, 128, size=lanes).tolist()
        wgt_row = rng.integers(-4, 5, size=lanes).tolist()
        for lane in rng.choice(lanes, size=max(1, lanes // 2), replace=False):
            act_row[int(lane)] = int(rng.choice([-128, 127]))
            wgt_row[int(lane)] = int(rng.choice([-128, 127]))
        add(act_row, wgt_row)

    return acts, wgts


def _mac_lane_extreme_coverage(acts, wgts, lanes: int) -> tuple[int, int]:
    """How many lanes were driven to -128 and to +127 on either operand.

    Reported so the generator cannot quietly narrow to a subset of lanes while
    the testbench keeps passing; the testbenches assert on these counts.
    """

    saw_min = 0
    saw_max = 0
    for lane in range(lanes):
        values = {row[lane] for row in acts} | {row[lane] for row in wgts}
        saw_min += -128 in values
        saw_max += 127 in values
    return saw_min, saw_max


def generate_mac_vectors() -> str:
    """Direct differential vectors for tb_conv5x5_row_mac and tb_dense_row_mac.

    Both MAC blocks were reachable only through a wrapper: conv5x5_row_mac
    through tb_conv5x5_pe, dense_row_mac through tb_dense_engine's five F6
    output values. For a combinational block with five or eight independent
    multiply lanes that is thin stimulus, and the gate-level tier measured
    exactly how thin -- mutation scores of 3/4 and 2/5 against 5/5 for the
    blocks that have a testbench of their own (results/gls_20260815.log).
    These are also the two blocks unbounded equivalence cannot reach
    (synth/equiv_mapped_requantize.ys explains why), so simulation is the only
    evidence they have and it needs to be the good kind.

    Both rows are packed lane 0 in the low byte, matching the RTL's comment.
    """

    conv_acts, conv_wgts = _mac_stimulus(lanes=5, seed=20260816, random_count=3000)
    dense_acts, dense_wgts = _mac_stimulus(lanes=8, seed=20260817, random_count=4000)

    conv_expected = _row_mac_reference(conv_acts, conv_wgts)
    dense_expected = _row_mac_reference(dense_acts, dense_wgts)

    # The ports are 32 bits wide; a reference value outside that range would be
    # truncated on the way into the hex file and then reported as a mismatch
    # against correct RTL. Five or eight int8 products cannot get near it, but
    # assert rather than assume, because LANES is a parameter.
    for name, expected in (("conv", conv_expected), ("dense", dense_expected)):
        if expected.min() < -(1 << 31) or expected.max() > (1 << 31) - 1:
            raise ValueError(f"{name} row MAC reference overflows the 32-bit port")

    mac_dir = VECTOR_DIR / "mac"
    write_hex(mac_dir / "conv_act.hex", np.array([_pack_lanes(r) for r in conv_acts], dtype=object), 40)
    write_hex(mac_dir / "conv_wgt.hex", np.array([_pack_lanes(r) for r in conv_wgts], dtype=object), 40)
    write_hex(mac_dir / "conv_expected.hex", conv_expected, 32)
    write_hex(mac_dir / "dense_act.hex", np.array([_pack_lanes(r) for r in dense_acts], dtype=object), 64)
    write_hex(mac_dir / "dense_wgt.hex", np.array([_pack_lanes(r) for r in dense_wgts], dtype=object), 64)
    write_hex(mac_dir / "dense_expected.hex", dense_expected, 32)

    conv_min_lanes, conv_max_lanes = _mac_lane_extreme_coverage(conv_acts, conv_wgts, 5)
    dense_min_lanes, dense_max_lanes = _mac_lane_extreme_coverage(dense_acts, dense_wgts, 8)

    print(
        f"conv5x5_row_mac: {len(conv_acts)} differential cases, "
        f"peak {conv_expected.max()} / {conv_expected.min()}"
    )
    print(
        f"dense_row_mac: {len(dense_acts)} differential cases, "
        f"peak {dense_expected.max()} / {dense_expected.min()}"
    )

    return f"""// ---- tb_conv5x5_row_mac.sv ----
`define TV_MACC_COUNT {len(conv_acts)}
`define TV_MACC_LANES 5
`define TV_MACC_PEAK_POS {conv_expected.max()}
`define TV_MACC_PEAK_NEG {conv_expected.min()}
`define TV_MACC_LANES_AT_MIN {conv_min_lanes}
`define TV_MACC_LANES_AT_MAX {conv_max_lanes}

// ---- tb_dense_row_mac.sv ----
`define TV_MACD_COUNT {len(dense_acts)}
`define TV_MACD_LANES 8
`define TV_MACD_PEAK_POS {dense_expected.max()}
`define TV_MACD_PEAK_NEG {dense_expected.min()}
`define TV_MACD_LANES_AT_MIN {dense_min_lanes}
`define TV_MACD_LANES_AT_MAX {dense_max_lanes}
"""


def generate_pe_vectors() -> str:
    """Streaming stimulus for tb_conv5x5_pe_stream.sv.

    `conv5x5_pe` is the row MAC plus the control around it: bias injected at
    `first_i`, an int32 accumulator carried across rows, requantization at
    `last_i`, and an output held under backpressure. The MAC now has its own
    sweep (generate_mac_vectors), but until this existed the *wrapper* was
    tested by exactly one output pixel -- five rows of all-ones activations,
    weights 1..5, bias 5, **shift 0, ReLU off** -- so the requantizer inside the
    PE never saw a non-zero shift, never saturated and never clamped, the bias
    was never negative or large, and no pixel ever followed another, which
    leaves the accumulator's restart between pixels unexercised. A gate-level
    mutation survives in that wrapper (results/gls_20260815.log), which is what
    an untested control path looks like from the outside.

    The corners are reached by **choosing the accumulator first and solving for
    the bias**: `bias = target - sum(row sums)`. That drives the exact
    round-half-away-from-zero ties, both saturation rails and the ReLU clamp
    through a genuine multi-row accumulation with non-trivial rows, rather than
    through a single hand-picked row that would leave the accumulation itself
    barely exercised.
    """

    int32_min, int32_max = -(1 << 31), (1 << 31) - 1
    rng = np.random.default_rng(20260818)

    pixels: list[dict] = []

    def row_sum(act_row, wgt_row) -> int:
        return int(_row_mac_reference([list(act_row)], [list(wgt_row)])[0])

    def random_rows(count: int, lo: int = -128, hi: int = 128):
        return [
            (rng.integers(lo, hi, size=5).tolist(), rng.integers(lo, hi, size=5).tolist())
            for _ in range(count)
        ]

    def add_pixel(rows, shift: int, relu: bool, *, bias=None, target=None) -> None:
        total = sum(row_sum(a, w) for a, w in rows)
        if target is not None:
            bias = target - total
        assert bias is not None
        if not int32_min <= bias <= int32_max:
            return
        accumulator = total + bias
        if not int32_min <= accumulator <= int32_max:
            return
        pixels.append(
            {
                "rows": rows,
                "bias": bias,
                "shift": shift,
                "relu": int(relu),
                "acc": accumulator,
                "expected": int(requantize(np.int64(accumulator), shift, bool(relu))),
            }
        )

    # 1. Every shift, both ReLU settings, and the tie values the rounding rule
    #    is *defined* by. At shift s the tie sits at 2^(s-1) and must round away
    #    from zero; the pixel either side of it pins which way.
    for shift in range(32):
        for relu in (False, True):
            if shift:
                half = 1 << (shift - 1)
                for target in (half, -half, half - 1, -(half - 1), half + 1, -(half + 1)):
                    add_pixel(random_rows(3), shift, relu, target=target)
            add_pixel(random_rows(4), shift, relu, target=int(rng.integers(-40, 41)) << shift)

    # 2. Both saturation rails, approached from either side, at several shifts.
    #    Nothing in the old pixel could saturate at all.
    for shift in (0, 1, 4, 7, 12, 20, 31):
        for out_target in (-129, -128, -127, 126, 127, 128, 400, -400):
            add_pixel(random_rows(2), shift, False, target=out_target << shift)

    # 3. The ReLU clamp: negative accumulators with ReLU on.
    for shift in (0, 3, 7, 16):
        for target in (-1, -2, -1000, -(1 << 20), -(1 << 30)):
            add_pixel(random_rows(3), shift, True, target=target)

    # 4. Row counts. One row means first_i and last_i on the same beat -- a
    #    single-beat pixel, which the original testbench never produced and
    #    which is the shape most likely to break a first/last decode.
    for count in (1, 2, 3, 4, 5, 6, 7, 8):
        for shift in (0, 5, 9):
            add_pixel(random_rows(count), shift, bool(count % 2), bias=int(rng.integers(-500, 501)))

    # 5. Bias corners, including bias = 0 (pure accumulation) and a bias large
    #    enough to dominate every row product.
    for bias in (0, 1, -1, 1 << 20, -(1 << 20), 1 << 28, -(1 << 28)):
        for shift in (0, 8, 24):
            add_pixel(random_rows(5), shift, False, bias=bias)

    # 6. Randomized bulk: every operand, bias, shift and ReLU setting random,
    #    over the whole legal space.
    for _ in range(300):
        add_pixel(
            random_rows(int(rng.integers(1, 9))),
            int(rng.integers(0, 32)),
            bool(rng.integers(0, 2)),
            bias=int(rng.integers(-(1 << 22), 1 << 22)),
        )

    # The expected values above compose two shipped oracles -- dense_int8's
    # accumulator for each row and requantize for the output -- in the order
    # the PE's contract specifies. Check that composition against the *layer*
    # oracle rather than trusting it: a genuine single-channel 5x5 convolution
    # is exactly one PE pixel of five rows, so conv2d_valid_int8 must agree.
    check_act = rng.integers(-128, 128, size=(1, 5, 5)).astype(np.int8)
    check_wgt = rng.integers(-128, 128, size=(1, 1, 5, 5)).astype(np.int8)
    for check_shift in (0, 3, 7, 15):
        for check_relu in (False, True):
            check_bias = np.array([int(rng.integers(-5000, 5001))], dtype=np.int64)
            layer_out, _ = conv2d_valid_int8(
                check_act, check_wgt, check_bias, shift=check_shift, relu=check_relu
            )
            rows = [
                (check_act[0, r, :].tolist(), check_wgt[0, 0, r, :].tolist())
                for r in range(5)
            ]
            composed = int(
                requantize(
                    np.int64(int(check_bias[0]) + sum(row_sum(a, w) for a, w in rows)),
                    check_shift,
                    check_relu,
                )
            )
            if composed != int(layer_out[0, 0, 0]):
                raise ValueError(
                    "PE oracle disagrees with conv2d_valid_int8 at "
                    f"shift={check_shift} relu={check_relu}: {composed} vs {int(layer_out[0, 0, 0])}"
                )
            add_pixel(rows, check_shift, check_relu, bias=int(check_bias[0]))

    # Flatten to the per-row and per-pixel files the testbench reads.
    act_words, wgt_words = [], []
    row_counts, biases, shifts, relus, expected = [], [], [], [], []
    for pixel in pixels:
        for act_row, wgt_row in pixel["rows"]:
            act_words.append(_pack_lanes(act_row))
            wgt_words.append(_pack_lanes(wgt_row))
        row_counts.append(len(pixel["rows"]))
        biases.append(pixel["bias"])
        shifts.append(pixel["shift"])
        relus.append(pixel["relu"])
        expected.append(pixel["expected"])

    pe_dir = VECTOR_DIR / "pe"
    write_hex(pe_dir / "act.hex", np.array(act_words, dtype=object), 40)
    write_hex(pe_dir / "wgt.hex", np.array(wgt_words, dtype=object), 40)
    write_hex(pe_dir / "rows.hex", np.array(row_counts, dtype=np.int64), 8)
    write_hex(pe_dir / "bias.hex", np.array(biases, dtype=np.int64), 32)
    write_hex(pe_dir / "shift.hex", np.array(shifts, dtype=np.int64), 8)
    write_hex(pe_dir / "relu.hex", np.array(relus, dtype=np.int64), 8)
    write_hex(pe_dir / "expected.hex", np.array(expected, dtype=np.int64), 8)

    # Coverage the testbench asserts on, so a generator narrowed by a later
    # edit fails the run instead of quietly testing less.
    sat_hi = sum(1 for p in pixels if p["expected"] == 127)
    sat_lo = sum(1 for p in pixels if p["expected"] == -128)
    relu_clamped = sum(1 for p in pixels if p["relu"] and p["acc"] < 0 and p["expected"] == 0)
    one_row = sum(1 for p in pixels if len(p["rows"]) == 1)
    max_rows = max(len(p["rows"]) for p in pixels)
    shifts_used = len(set(shifts))

    # `shift_i` and `relu_en_i` are sampled by the requantizer at the cycle the
    # output beat is registered, not at `last_i`, so they have to stay stable
    # while a pixel is in flight -- which is how `conv2d_engine` drives them
    # (constant for a whole layer). The testbench therefore streams a pixel
    # straight into the previous one only when both match, and drains first
    # otherwise. This is how many such adjacent pairs exist, and the testbench
    # asserts it saw exactly that many: the accumulator's restart between
    # pixels is only exercised on those, and it is the case the original
    # single-pixel testbench could never reach.
    back_to_back = sum(
        1
        for i in range(1, len(pixels))
        if pixels[i]["shift"] == pixels[i - 1]["shift"]
        and pixels[i]["relu"] == pixels[i - 1]["relu"]
    )

    print(
        f"conv5x5_pe stream: {len(pixels)} pixels / {len(act_words)} rows, "
        f"sat {sat_hi}+/{sat_lo}-, relu-clamped {relu_clamped}, "
        f"{one_row} single-beat, {shifts_used}/32 shifts, {back_to_back} back-to-back"
    )

    return f"""// ---- tb_conv5x5_pe_stream.sv ----
`define TV_PE_PIXELS {len(pixels)}
`define TV_PE_ROWS_TOTAL {len(act_words)}
`define TV_PE_MAX_ROWS {max_rows}
`define TV_PE_SAT_HI {sat_hi}
`define TV_PE_SAT_LO {sat_lo}
`define TV_PE_RELU_CLAMPED {relu_clamped}
`define TV_PE_ONE_ROW {one_row}
`define TV_PE_SHIFTS_USED {shifts_used}
`define TV_PE_BACK_TO_BACK {back_to_back}
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
        generate_mac_vectors(),
        generate_pe_vectors(),
        generate_extremes_vectors(),
        generate_top_vectors(),
    ]

    header = "// Generated by golden/generate_vectors.py; do not hand edit.\n"
    (VECTOR_DIR / "config.svh").write_text(header + "\n".join(blocks), encoding="ascii")


if __name__ == "__main__":
    main()
