"""Train the deployment LeNet-5 on MNIST, then post-training-quantize it to INT8.

Every other tier in this project checks the RTL against `deploy_forward_int8`
running on `random_deploy_parameters` -- deterministic, shape-correct, and
explicitly "not trained and not expected to classify MNIST". That proves the
hardware computes what the model computes. It cannot show the hardware computes
anything *useful*, because the model it is being compared against does not.

This module closes that gap offline. It trains a float LeNet-5 whose topology is
exactly what `rtl/lenet5_top.sv` implements -- including the LeCun Table I sparse
C3 connectivity, which is a real capacity constraint on training and not a
detail that can be added afterwards -- and then quantizes it into the *specific*
fixed-point scheme the RTL implements: symmetric int8 weights, int32 biases, and
a per-layer power-of-two right shift. There are no per-channel scales and no
zero points, because `rtl/requantize.sv` has neither.

Training is offline and is deliberately NOT part of the regression or CI. Its
output, `golden/trained_params.npz`, is a checked-in artifact; the vector
generator reads that artifact and emits hex, so CI stays deterministic and never
retrains. Rerun this by hand only when the trained network itself should change.

Usage:
    python -m golden.train_lenet5 --data path/to/mnist.npz

The quantization scheme, stated once so the arithmetic below is checkable:

    a_real ~= a_int * s_a          activations, s_a a positive float
    w_real ~= w_int * s_w          weights, s_w = max|W| / 127
    b_int   = round(b_real / (s_a * s_w))
    acc_int = sum(a_int * w_int) + b_int          exact in int32
    out_int = requantize(acc_int, shift)          == round(acc_int / 2^shift)
    s_out   = s_a * s_w * 2^shift

`shift` is the only free knob per layer, which is exactly why it has to be
calibrated from measured accumulator statistics rather than left at 7.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from golden.lenet5 import c3_connectivity
from golden.quantized_conv import requantize
# The accuracy figure this file reports and the vectors the RTL is actually
# driven with have to come from the same input mapping. Two copies that agree
# today are two copies that can drift, and the symptom would be a silent change
# in what the hardware is asked, not a failure.
from golden.trained import quantize_image

KERNEL = 5
CALIBRATION_PERCENTILE = 99.9

# The classifier's shift is absent on purpose: rtl/classifier_argmax.sv compares
# raw accumulators, so any positive scale gives the same argmax.
SHIFT_LAYERS = ("c1", "c3", "c5", "f6")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_mnist(path: Path) -> dict[str, np.ndarray]:
    """Load mnist.npz and pad 28x28 to the 32x32 the RTL expects.

    Pixels become floats in [0,1]; the INT8 input scale below is 1/127, so a
    saturated white pixel maps to int8 127 and the zero background maps to 0.
    Padding with zero is therefore also zero after quantization, which matters:
    the RTL has no notion of a padding value.
    """

    with np.load(path) as raw:
        keys = set(raw.files)
        if {"x_train", "y_train", "x_test", "y_test"} <= keys:
            x_train, y_train = raw["x_train"], raw["y_train"]
            x_test, y_test = raw["x_test"], raw["y_test"]
        else:  # some mirrors ship the tuple-style layout
            raise KeyError(f"unrecognised mnist.npz layout, keys were {sorted(keys)}")

    def prep(images: np.ndarray) -> np.ndarray:
        x = images.astype(np.float64) / 255.0
        padded = np.zeros((x.shape[0], 1, 32, 32), dtype=np.float64)
        padded[:, 0, 2:30, 2:30] = x
        return padded

    return {
        "x_train": prep(x_train),
        "y_train": y_train.astype(np.int64),
        "x_test": prep(x_test),
        "y_test": y_test.astype(np.int64),
    }


# --------------------------------------------------------------------------
# float layers
# --------------------------------------------------------------------------

def im2col(x: np.ndarray, k: int = KERNEL) -> np.ndarray:
    """[N,C,H,W] -> [N, out_h*out_w, C*k*k] for a valid convolution."""

    n, c, h, w = x.shape
    out_h, out_w = h - k + 1, w - k + 1
    sn, sc, sh, sw = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, c, out_h, out_w, k, k),
        strides=(sn, sc, sh, sw, sh, sw),
        writeable=False,
    )
    # -> [N, out_h, out_w, C, k, k] so the flattened tap order is C-major,
    # matching the [O,C,5,5] weight layout the RTL and golden model use.
    windows = windows.transpose(0, 2, 3, 1, 4, 5)
    return np.ascontiguousarray(windows).reshape(n, out_h * out_w, c * k * k)


def col2im(dcols: np.ndarray, x_shape: tuple[int, ...], k: int = KERNEL) -> np.ndarray:
    """Scatter im2col gradients back to input space (25 slice-adds, no np.add.at)."""

    n, c, h, w = x_shape
    out_h, out_w = h - k + 1, w - k + 1
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    d = dcols.reshape(n, out_h, out_w, c, k, k)
    for ky in range(k):
        for kx in range(k):
            dx[:, :, ky : ky + out_h, kx : kx + out_w] += d[:, :, :, :, ky, kx].transpose(
                0, 3, 1, 2
            )
    return dx


def conv_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    """x [N,C,H,W], w [O,C,5,5], b [O] -> [N,O,out_h,out_w] plus the im2col cache.

    Every matmul here is collapsed to a single 2-D call. Leaving the batch axis
    on gives NumPy a stacked 3-D matmul -- N separate BLAS calls on matrices too
    small to amortise the call overhead -- and the equivalent einsum is worse
    still: it never reaches BLAS at all. Profiling the naive version put 94% of
    training time inside one einsum.
    """

    n, _, h, width = x.shape
    out_ch = w.shape[0]
    out_h, out_w = h - KERNEL + 1, width - KERNEL + 1
    cols = im2col(x)                                          # [N, P, K]
    k = cols.shape[-1]
    w_flat = np.ascontiguousarray(w.reshape(out_ch, k).T)     # [K, O]
    out = cols.reshape(-1, k) @ w_flat                        # one BLAS call
    out += b
    out = out.reshape(n, out_h * out_w, out_ch).transpose(0, 2, 1)
    return out.reshape(n, out_ch, out_h, out_w), cols


def conv_backward(dout: np.ndarray, cols: np.ndarray, w: np.ndarray, x_shape):
    n, out_ch = dout.shape[0], dout.shape[1]
    k = cols.shape[-1]
    d2 = np.ascontiguousarray(
        dout.reshape(n, out_ch, -1).transpose(0, 2, 1).reshape(-1, out_ch)
    )                                                          # [N*P, O]
    c2 = cols.reshape(-1, k)                                   # [N*P, K]
    # (c2.T @ d2).T, not d2.T @ c2, though the two are the same matrix.
    # Multithreaded OpenBLAS has a pathological path for A.T @ B when A is
    # narrow and the reduction axis is long: at C3's shapes -- [12800,16].T @
    # [12800,150] -- the natural spelling measures 632 ms against 4 ms for the
    # reversed one, a 158x difference for identical arithmetic. Forcing one
    # thread also fixes it, but that is an environment knob this file cannot
    # rely on; the operand order travels with the code.
    dw = (c2.T @ d2).T.reshape(w.shape)                        # [O, K] -> w.shape
    db = d2.sum(axis=0)
    dcols = d2 @ w.reshape(out_ch, k)                          # [N*P, K]
    return col2im(dcols.reshape(n, -1, k), x_shape), dw, db


def avgpool_forward(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


def avgpool_backward(dout: np.ndarray, x_shape) -> np.ndarray:
    n, c, h, w = x_shape
    d = dout[:, :, :, None, :, None] / 4.0
    return np.broadcast_to(d, (n, c, h // 2, 2, w // 2, 2)).reshape(x_shape).copy()


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = labels.shape[0]
    loss = -np.log(np.maximum(probs[np.arange(n), labels], 1e-12)).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    return loss, dlogits / n


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class LeNet5Float:
    """Float LeNet-5 with exactly the RTL's topology, including the C3 mask."""

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.mask = c3_connectivity().astype(np.float64)      # [16,6]

        def he(shape, fan_in):
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)

        self.p = {
            "c1_w": he((6, 1, 5, 5), 25),
            "c1_b": np.zeros(6),
            "c3_w": he((16, 6, 5, 5), 60 * 25 / 16),
            "c3_b": np.zeros(16),
            "c5_w": he((120, 16, 5, 5), 400),
            "c5_b": np.zeros(120),
            "f6_w": he((84, 120), 120),
            "f6_b": np.zeros(84),
            "cls_w": he((10, 84), 84),
            "cls_b": np.zeros(10),
        }
        self._apply_mask()

    def _apply_mask(self) -> None:
        """Force unconnected C3 taps to exactly zero, in weights and gradients.

        The RTL does not multiply these at all; a nonzero float weight here
        would train information into a connection the hardware cannot express.
        """
        self.p["c3_w"] *= self.mask[:, :, None, None]

    def forward(self, x: np.ndarray):
        cache = {"x": x}
        c1_pre, cache["c1_cols"] = conv_forward(x, self.p["c1_w"], self.p["c1_b"])
        c1 = np.maximum(c1_pre, 0.0)
        cache["c1_pre"] = c1_pre
        s2 = avgpool_forward(c1)
        cache["c1_shape"], cache["s2"] = c1.shape, s2

        c3_w = self.p["c3_w"] * self.mask[:, :, None, None]
        c3_pre, cache["c3_cols"] = conv_forward(s2, c3_w, self.p["c3_b"])
        c3 = np.maximum(c3_pre, 0.0)
        cache["c3_pre"] = c3_pre
        s4 = avgpool_forward(c3)
        cache["c3_shape"], cache["s4"] = c3.shape, s4

        c5_pre, cache["c5_cols"] = conv_forward(s4, self.p["c5_w"], self.p["c5_b"])
        c5 = np.maximum(c5_pre, 0.0)
        cache["c5_pre"] = c5_pre
        c5_flat = c5.reshape(c5.shape[0], -1)
        cache["c5_flat"] = c5_flat

        f6_pre = c5_flat @ self.p["f6_w"].T + self.p["f6_b"]
        f6 = np.maximum(f6_pre, 0.0)
        cache["f6_pre"], cache["f6"] = f6_pre, f6

        logits = f6 @ self.p["cls_w"].T + self.p["cls_b"]
        return logits, cache

    def backward(self, dlogits: np.ndarray, cache: dict):
        g = {}
        g["cls_w"] = dlogits.T @ cache["f6"]
        g["cls_b"] = dlogits.sum(axis=0)
        df6 = dlogits @ self.p["cls_w"]
        df6 *= cache["f6_pre"] > 0

        g["f6_w"] = df6.T @ cache["c5_flat"]
        g["f6_b"] = df6.sum(axis=0)
        dc5 = df6 @ self.p["f6_w"]
        dc5 = dc5.reshape(cache["c5_pre"].shape)
        dc5 *= cache["c5_pre"] > 0

        ds4, g["c5_w"], g["c5_b"] = conv_backward(
            dc5, cache["c5_cols"], self.p["c5_w"], cache["s4"].shape
        )
        dc3 = avgpool_backward(ds4, cache["c3_shape"])
        dc3 *= cache["c3_pre"] > 0

        c3_w = self.p["c3_w"] * self.mask[:, :, None, None]
        ds2, g["c3_w"], g["c3_b"] = conv_backward(
            dc3, cache["c3_cols"], c3_w, cache["s2"].shape
        )
        g["c3_w"] *= self.mask[:, :, None, None]      # never learn a dead tap

        dc1 = avgpool_backward(ds2, cache["c1_shape"])
        dc1 *= cache["c1_pre"] > 0
        _, g["c1_w"], g["c1_b"] = conv_backward(
            dc1, cache["c1_cols"], self.p["c1_w"], cache["x"].shape
        )
        return g


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.decay = lr, beta1, beta2, eps, decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, g in grads.items():
            if self.decay and k.endswith("_w"):
                g = g + self.decay * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            params[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)


# --------------------------------------------------------------------------
# INT8 evaluation (vectorised; bit-checked against deploy_forward_int8)
# --------------------------------------------------------------------------

def _exact_matmul(a_int: np.ndarray, b_int: np.ndarray) -> np.ndarray:
    """Integer matmul via float64 BLAS, exact because every product fits 2^53.

    NumPy has no integer BLAS, so an int64 @ int64 falls back to a slow generic
    loop. Every value here is an integer whose magnitude stays far below 2^53
    (worst case is C5: 400 taps * 127 * 127 ~ 6.5e6), so float64 arithmetic is
    exact and hundreds of times faster. The assertion keeps that argument
    honest rather than assumed.
    """

    out = a_int.astype(np.float64) @ b_int.astype(np.float64)
    assert np.max(np.abs(out)) < 2.0**53, "float64 matmul would lose integer exactness"
    return np.rint(out).astype(np.int64)


def int8_forward_batch(images: np.ndarray, q: dict, shifts: dict) -> np.ndarray:
    """Vectorised int8 forward returning predicted classes for a batch."""

    x = images.astype(np.int64)                      # [N,1,32,32]

    def conv(inp, w, b, shift, relu):
        n = inp.shape[0]
        out_ch = w.shape[0]
        h, width = inp.shape[2], inp.shape[3]
        out_h, out_w = h - 4, width - 4
        cols = im2col(inp.astype(np.float64))        # [N,P,C*25]
        acc = _exact_matmul(cols, w.reshape(out_ch, -1).T.astype(np.int64))
        acc += b.astype(np.int64)
        out = requantize(acc, shift, relu)
        return out.transpose(0, 2, 1).reshape(n, out_ch, out_h, out_w).astype(np.int64)

    def pool(inp):
        n, c, h, width = inp.shape
        s = inp.reshape(n, c, h // 2, 2, width // 2, 2).sum(axis=(3, 5))
        return requantize(s, shift=2, relu=False).astype(np.int64)

    c1 = conv(x, q["c1_w"], q["c1_b"], shifts["c1"], True)
    s2 = pool(c1)
    # Unconnected C3 taps are already exactly zero in q["c3_w"], so the dense
    # matmul contributes nothing for them -- bit-identical to skipping them.
    c3 = conv(s2, q["c3_w"], q["c3_b"], shifts["c3"], True)
    s4 = pool(c3)
    c5 = conv(s4, q["c5_w"], q["c5_b"], shifts["c5"], True)

    flat = c5.reshape(c5.shape[0], -1)
    f6_acc = _exact_matmul(flat, q["f6_w"].T.astype(np.int64)) + q["f6_b"].astype(np.int64)
    f6 = requantize(f6_acc, shifts["f6"], True).astype(np.int64)
    cls_acc = _exact_matmul(f6, q["cls_w"].T.astype(np.int64)) + q["cls_b"].astype(np.int64)
    return np.argmax(cls_acc, axis=1)


# quantize_image is imported at the top of this file from golden/trained.py
# rather than reimplemented here -- see the note there.


# --------------------------------------------------------------------------
# post-training quantization
# --------------------------------------------------------------------------

def _quantize_weights(w: np.ndarray, mask: np.ndarray | None = None):
    scale = float(np.max(np.abs(w))) / 127.0
    if scale == 0.0:
        scale = 1.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    if mask is not None:
        q = (q * mask.astype(np.int8)[:, :, None, None]).astype(np.int8)
    return q, scale


def quantize_network(model: LeNet5Float, calib_images: np.ndarray, percentile: float):
    """Quantize weights, then calibrate one power-of-two shift per layer.

    The shift for a layer is chosen from the measured accumulator distribution
    on real images: 2^shift ~= P/127 where P is a high percentile of |acc|.
    Using a percentile rather than the max trades a handful of saturated
    outliers for a bit more resolution on everything else, which is the whole
    reason `requantize` has a saturating clamp.
    """

    mask = c3_connectivity()
    q: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}

    for name, msk in (("c1", None), ("c3", mask), ("c5", None), ("f6", None), ("cls", None)):
        w = model.p[f"{name}_w"]
        qw, sw = _quantize_weights(w, msk) if msk is not None else _quantize_weights(w)
        q[f"{name}_w"], scales[name] = qw, sw

    s_in = 1.0 / 127.0                      # input activation scale
    x = quantize_image(calib_images).astype(np.int64)
    shifts: dict[str, int] = {}
    report: list[tuple[str, float, int, float]] = []

    def pick_shift(acc: np.ndarray) -> tuple[int, float]:
        p = float(np.percentile(np.abs(acc), percentile))
        if p <= 0:
            return 0, p
        shift = int(max(0, min(31, round(np.log2(max(p, 1.0) / 127.0)))))
        return shift, p

    def conv_stage(inp, name, relu, w_shape_ch):
        nonlocal s_in
        sw = scales[name]
        b_int = np.rint(model.p[f"{name}_b"] / (s_in * sw)).astype(np.int32)
        q[f"{name}_b"] = b_int
        cols = im2col(inp.astype(np.float64))
        acc = _exact_matmul(cols, q[f"{name}_w"].reshape(w_shape_ch, -1).T.astype(np.int64))
        acc += b_int.astype(np.int64)
        shift, p = pick_shift(acc)
        shifts[name] = shift
        report.append((name, p, shift, s_in * sw * (2.0**shift)))
        n = inp.shape[0]
        out_h, out_w = inp.shape[2] - 4, inp.shape[3] - 4
        out = requantize(acc, shift, relu)
        s_in = s_in * sw * (2.0**shift)
        return out.transpose(0, 2, 1).reshape(n, w_shape_ch, out_h, out_w).astype(np.int64)

    def pool(inp):
        n, c, h, width = inp.shape
        s = inp.reshape(n, c, h // 2, 2, width // 2, 2).sum(axis=(3, 5))
        return requantize(s, shift=2, relu=False).astype(np.int64)

    c1 = conv_stage(x, "c1", True, 6)
    s2 = pool(c1)
    c3 = conv_stage(s2, "c3", True, 16)
    s4 = pool(c3)
    c5 = conv_stage(s4, "c5", True, 120)

    flat = c5.reshape(c5.shape[0], -1)
    sw = scales["f6"]
    q["f6_b"] = np.rint(model.p["f6_b"] / (s_in * sw)).astype(np.int32)
    acc = _exact_matmul(flat, q["f6_w"].T.astype(np.int64)) + q["f6_b"].astype(np.int64)
    shift, p = pick_shift(acc)
    shifts["f6"] = shift
    report.append(("f6", p, shift, s_in * sw * (2.0**shift)))
    f6 = requantize(acc, shift, True).astype(np.int64)
    s_in = s_in * sw * (2.0**shift)

    sw = scales["cls"]
    q["cls_b"] = np.rint(model.p["cls_b"] / (s_in * sw)).astype(np.int32)

    for key in ("c1_b", "c3_b", "c5_b", "f6_b", "cls_b"):
        span = int(np.max(np.abs(q[key]))) if q[key].size else 0
        assert span < 2**31, f"{key} does not fit the RTL's 32-bit bias port"

    return q, shifts, report


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def evaluate_float(model: LeNet5Float, x: np.ndarray, y: np.ndarray, batch: int = 500) -> float:
    correct = 0
    for i in range(0, x.shape[0], batch):
        logits, _ = model.forward(x[i : i + batch])
        correct += int((np.argmax(logits, axis=1) == y[i : i + batch]).sum())
    return correct / x.shape[0]


def evaluate_int8(q: dict, shifts: dict, x: np.ndarray, y: np.ndarray, batch: int = 500) -> float:
    correct = 0
    for i in range(0, x.shape[0], batch):
        pred = int8_forward_batch(quantize_image(x[i : i + batch]), q, shifts)
        correct += int((pred == y[i : i + batch]).sum())
    return correct / x.shape[0]


def train(
    data: dict,
    epochs: int = 14,
    batch: int = 128,
    lr: float = 1.5e-3,
    seed: int = 7,
    decay: float = 1e-4,
) -> LeNet5Float:
    model = LeNet5Float(seed=seed)
    opt = Adam(model.p, lr=lr, decay=decay)
    rng = np.random.default_rng(seed + 1)
    x, y = data["x_train"], data["y_train"]
    n = x.shape[0]
    steps = n // batch

    for epoch in range(epochs):
        order = rng.permutation(n)
        # cosine decay: the last epochs matter most for how quantizable the
        # weights end up being, and a hot learning rate leaves them ragged.
        epoch_lr = lr * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs))
        started = time.time()
        running = 0.0
        for step in range(steps):
            idx = order[step * batch : (step + 1) * batch]
            logits, cache = model.forward(x[idx])
            loss, dlogits = softmax_cross_entropy(logits, y[idx])
            grads = model.backward(dlogits, cache)
            opt.step(model.p, grads, lr=epoch_lr)
            model._apply_mask()
            running += loss
        acc = evaluate_float(model, data["x_test"][:2000], data["y_test"][:2000])
        print(
            f"  epoch {epoch + 1:2d}/{epochs}  loss {running / steps:.4f}  "
            f"test(2k) {acc * 100:.2f}%  lr {epoch_lr:.2e}  {time.time() - started:.1f}s",
            flush=True,
        )
    return model


def emit_demo_set(mnist_path: Path, count: int, out: Path) -> None:
    """Cache the first `count` MNIST test digits as a small checked-in artifact.

    The vector generator needs real digits, but it runs in CI, where mnist.npz
    is not available and retraining would be absurd. Caching the handful of
    images the testbench actually drives keeps `make vectors` deterministic and
    dependency-free while leaving the images genuinely unmodified.

    "First N in dataset order" is load-bearing, not laziness: picking images
    because the network gets them right would make the accuracy assertion in
    tb_trained_mnist.sv tautological.
    """

    with np.load(mnist_path) as raw:
        images = raw["x_test"][:count].astype(np.uint8)
        labels = raw["y_test"][:count].astype(np.uint8)
    np.savez_compressed(out, images=images, labels=labels)
    print(f"wrote {out}  ({count} MNIST test digits, dataset order: {list(labels)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../mnist_nn_scratch/data/mnist.npz"),
        help="path to mnist.npz",
    )
    parser.add_argument(
        "--emit-demo",
        action="store_true",
        help="only write golden/mnist_demo.npz, do not train",
    )
    parser.add_argument("--demo-count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calib", type=int, default=512, help="calibration images")
    parser.add_argument("--percentile", type=float, default=CALIBRATION_PERCENTILE)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "trained_params.npz",
    )
    args = parser.parse_args()

    if args.emit_demo:
        emit_demo_set(
            args.data,
            args.demo_count,
            Path(__file__).resolve().parent / "mnist_demo.npz",
        )
        return

    print(f"loading {args.data}")
    data = load_mnist(args.data)
    print(f"  train {data['x_train'].shape}  test {data['x_test'].shape}")

    print("training float LeNet-5 (RTL topology, C3 masked to 60/96)")
    model = train(data, epochs=args.epochs, seed=args.seed)

    float_acc = evaluate_float(model, data["x_test"], data["y_test"])
    print(f"float test accuracy: {float_acc * 100:.2f}%")

    print(f"post-training quantization, {args.percentile}th-percentile calibration")
    q, shifts, report = quantize_network(
        model, data["x_train"][: args.calib], args.percentile
    )
    print(f"  {'layer':<6}{'|acc| pct':>14}{'shift':>7}{'out scale':>14}")
    for name, p, shift, s_out in report:
        print(f"  {name:<6}{p:>14.1f}{shift:>7d}{s_out:>14.3e}")

    int8_acc = evaluate_int8(q, shifts, data["x_test"], data["y_test"])
    print(f"int8 test accuracy:  {int8_acc * 100:.2f}%  (float {float_acc * 100:.2f}%)")

    np.savez_compressed(
        args.out,
        **q,
        shifts=np.array([shifts[k] for k in SHIFT_LAYERS], dtype=np.int32),
        shift_layers=np.array(SHIFT_LAYERS),
        float_accuracy=np.array(float_acc),
        int8_accuracy=np.array(int8_acc),
        seed=np.array(args.seed),
        epochs=np.array(args.epochs),
        percentile=np.array(args.percentile),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
