#!/usr/bin/env python3
"""Unit tests for the NumPy golden models (standard-library unittest)."""

from __future__ import annotations

import unittest

import numpy as np

from golden.deploy import deploy_forward_int8, random_deploy_parameters
from golden.lenet5 import (
    CANONICAL_LAYERS,
    c3_connectivity,
    forward_canonical,
    random_canonical_parameters,
)
from golden.quantized_conv import (
    argmax_classifier,
    avg_pool2x2_int8,
    conv2d_valid_int8,
    dense_int8,
    requantize,
)


class GoldenModelTests(unittest.TestCase):
    def test_requantize_rounding_and_saturation(self) -> None:
        values = np.array([-1000, -192, -64, 64, 192, 1000])
        actual = requantize(values, shift=7, relu=False)
        expected = np.array([-8, -2, -1, 1, 2, 8], dtype=np.int8)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(requantize(50000, shift=0)), 127)
        self.assertEqual(int(requantize(-50000, shift=0)), -128)
        self.assertEqual(int(requantize(-10, shift=0, relu=True)), 0)

    def test_hand_calculated_convolution(self) -> None:
        x = np.arange(25, dtype=np.int8).reshape(1, 5, 5) - 12
        w = np.ones((1, 1, 5, 5), dtype=np.int8)
        y, acc = conv2d_valid_int8(x, w, np.array([7]), shift=0)
        self.assertEqual(int(acc[0, 0, 0]), 7)
        self.assertEqual(int(y[0, 0, 0]), 7)

    def test_disconnected_channel_is_ignored(self) -> None:
        x = np.ones((2, 5, 5), dtype=np.int8)
        x[1] = 100
        w = np.ones((1, 2, 5, 5), dtype=np.int8)
        mask = np.array([[1, 0]], dtype=bool)
        y, acc = conv2d_valid_int8(
            x, w, np.array([0]), shift=0, connectivity=mask
        )
        self.assertEqual(int(acc[0, 0, 0]), 25)
        self.assertEqual(int(y[0, 0, 0]), 25)

    def test_c3_table_has_60_connections(self) -> None:
        matrix = c3_connectivity()
        self.assertEqual(matrix.shape, (16, 6))
        self.assertEqual(int(matrix.sum()), 60)
        np.testing.assert_array_equal(matrix[15], np.ones(6, dtype=bool))

    def test_average_pool_rounding(self) -> None:
        x = np.array([[[1, 2], [3, 4]]], dtype=np.int8)
        self.assertEqual(int(avg_pool2x2_int8(x)[0, 0, 0]), 3)
        x = -x
        self.assertEqual(int(avg_pool2x2_int8(x)[0, 0, 0]), -3)

    def test_canonical_shapes_and_parameter_total(self) -> None:
        image = np.zeros((32, 32), dtype=np.float64)
        result = forward_canonical(image, random_canonical_parameters(seed=7))
        for spec in CANONICAL_LAYERS[1:-1]:
            self.assertEqual(tuple(result[spec.name].shape), spec.output_shape)
        total = sum(spec.trainable_parameters for spec in CANONICAL_LAYERS)
        self.assertEqual(total, 60000)

    def test_dense_hand_calculated(self) -> None:
        x = np.array([1, 2, 3], dtype=np.int8)
        w = np.array([[1, 1, 1], [2, 0, 0]], dtype=np.int8)
        b = np.array([0, 5], dtype=np.int32)
        y, acc = dense_int8(x, w, b, shift=0)
        np.testing.assert_array_equal(acc, [6, 7])
        np.testing.assert_array_equal(y, [6, 7])

    def test_dense_matches_matmul_plus_bias(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.integers(-32, 32, size=(5,), dtype=np.int8)
        w = rng.integers(-16, 16, size=(7, 5), dtype=np.int8)
        b = rng.integers(-300, 300, size=(7,), dtype=np.int32)
        _, acc = dense_int8(x, w, b, shift=0)
        independent = w.astype(np.int64) @ x.astype(np.int64) + b.astype(np.int64)
        np.testing.assert_array_equal(acc, independent)

    def test_dense_shift_and_relu_boundaries(self) -> None:
        values = [-1000, -192, -64, 64, 192, 1000]
        expected = [-8, -2, -1, 1, 2, 8]
        for value, want in zip(values, expected):
            y, _ = dense_int8(
                np.array([0], dtype=np.int8),
                np.array([[0]], dtype=np.int8),
                np.array([value], dtype=np.int64),
                shift=7,
            )
            self.assertEqual(int(y[0]), want)

        y, _ = dense_int8(
            np.array([0], dtype=np.int8),
            np.array([[0]], dtype=np.int8),
            np.array([50000], dtype=np.int64),
            shift=0,
        )
        self.assertEqual(int(y[0]), 127)

        y, _ = dense_int8(
            np.array([0], dtype=np.int8),
            np.array([[0]], dtype=np.int8),
            np.array([-50000], dtype=np.int64),
            shift=0,
        )
        self.assertEqual(int(y[0]), -128)

        y, _ = dense_int8(
            np.array([0], dtype=np.int8),
            np.array([[0]], dtype=np.int8),
            np.array([-10], dtype=np.int64),
            shift=0,
            relu=True,
        )
        self.assertEqual(int(y[0]), 0)

    def test_dense_rejects_incompatible_shapes(self) -> None:
        x = np.zeros(3, dtype=np.int8)
        w = np.zeros((2, 4), dtype=np.int8)
        b = np.zeros(2, dtype=np.int32)
        with self.assertRaises(ValueError):
            dense_int8(x, w, b, shift=0)
        w_ok = np.zeros((2, 3), dtype=np.int8)
        b_bad = np.zeros(3, dtype=np.int32)
        with self.assertRaises(ValueError):
            dense_int8(x, w_ok, b_bad, shift=0)

    def test_argmax_classifier_picks_clear_winner(self) -> None:
        x = np.array([1, 1, 1], dtype=np.int8)
        w = np.array([[1, 1, 1], [10, 10, 10]], dtype=np.int8)
        b = np.array([0, 0], dtype=np.int32)
        winner, acc = argmax_classifier(x, w, b)
        self.assertEqual(winner, 1)
        _, expected_acc = dense_int8(x, w, b, shift=0, relu=False)
        np.testing.assert_array_equal(acc, expected_acc)

    def test_argmax_classifier_tie_breaks_to_lowest_index(self) -> None:
        x = np.array([1], dtype=np.int8)
        w = np.array([[5], [1], [5]], dtype=np.int8)
        b = np.array([0, 0, 0], dtype=np.int32)
        winner, acc = argmax_classifier(x, w, b)
        np.testing.assert_array_equal(acc, [5, 1, 5])
        self.assertEqual(winner, 0)

    def test_argmax_classifier_avoids_saturation_misclassification(self) -> None:
        # Both accumulators saturate to the same clamped int8 score (127),
        # but the true accumulators are far apart. An argmax over the
        # requantized int8 scores would incorrectly pick class 0 (first
        # occurrence in a tie); argmax_classifier must pick class 1 since it
        # compares raw accumulators instead.
        x = np.array([0], dtype=np.int8)
        w = np.array([[0], [0]], dtype=np.int8)
        b = np.array([200, 999999], dtype=np.int64)
        scores, acc = dense_int8(x, w, b, shift=0, relu=False)
        np.testing.assert_array_equal(acc, [200, 999999])
        np.testing.assert_array_equal(scores, [127, 127])
        self.assertEqual(int(np.argmax(scores)), 0)  # documents the bug this avoids

        winner, cls_acc = argmax_classifier(x, w, b)
        np.testing.assert_array_equal(cls_acc, [200, 999999])
        self.assertEqual(winner, 1)

    def test_argmax_classifier_matches_dense_int8_end_to_end(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.integers(-32, 32, size=(6,), dtype=np.int8)
        w = rng.integers(-16, 16, size=(10, 6), dtype=np.int8)
        b = rng.integers(-300, 300, size=(10,), dtype=np.int32)
        winner, acc = argmax_classifier(x, w, b)
        _, expected_acc = dense_int8(x, w, b, shift=0, relu=False)
        np.testing.assert_array_equal(acc, expected_acc)
        self.assertEqual(winner, int(np.argmax(expected_acc)))

    def test_deploy_forward_int8_shape_chain(self) -> None:
        image = np.zeros((32, 32), dtype=np.int8)
        result = deploy_forward_int8(image, random_deploy_parameters(seed=13))
        self.assertEqual(result["C1"].shape, (6, 28, 28))
        self.assertEqual(result["S2"].shape, (6, 14, 14))
        self.assertEqual(result["C3"].shape, (16, 10, 10))
        self.assertEqual(result["S4"].shape, (16, 5, 5))
        self.assertEqual(result["C5"].shape, (120, 1, 1))
        self.assertEqual(result["F6"].shape, (84,))
        self.assertEqual(result["classifier_accumulators"].shape, (10,))
        self.assertIn(result["prediction"], range(10))

    def test_input_quantization_dispatches_on_dtype_not_data(self) -> None:
        """A dark uint8 image must not be read as an already-normalised one.

        `quantize_image` accepts both raw MNIST pixels (uint8, 0..255) and
        normalised floats (0..1), so it has to decide which it was handed. An
        earlier version decided by inspecting the data -- scale iff
        `x.max() > 1.0` -- which takes the wrong branch for a legitimately dark
        uint8 image whose brightest pixel is 0 or 1, turning a near-black pixel
        into a saturated 127. It fails silently: no exception, just a different
        image driven into the hardware. Deciding by dtype cannot do that, and
        this pins it.
        """
        from golden.trained import quantize_image

        pixels = np.array([[0, 1, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(
            quantize_image(pixels),
            quantize_image(pixels.astype(np.float64) / 255.0),
            "uint8 and pre-normalised float inputs must quantize identically",
        )

        dark = np.zeros((1, 4), dtype=np.uint8)
        dark[0, 0] = 1  # max() == 1, the exact case the old heuristic misread
        np.testing.assert_array_equal(
            quantize_image(dark), np.zeros((1, 4), dtype=np.int8)
        )


class TrainedModelTests(unittest.TestCase):
    """Checks on the checked-in trained artifact.

    These are cheap and they guard a claim the rest of the suite cannot: that
    the network shipped in golden/trained_params.npz is a real trained model
    quantized to the scheme the RTL implements, rather than merely a
    shape-correct array bundle.
    """

    def setUp(self) -> None:
        from golden.trained import TrainedModelMissing

        try:
            from golden.trained import trained_parameters, trained_shifts

            self.params = trained_parameters()
            self.shifts = trained_shifts()
        except TrainedModelMissing as exc:  # pragma: no cover - artifact is committed
            self.skipTest(str(exc))

    def test_trained_shapes_and_dtypes(self) -> None:
        expected = {
            "c1_w": (6, 1, 5, 5), "c1_b": (6,),
            "c3_w": (16, 6, 5, 5), "c3_b": (16,),
            "c5_w": (120, 16, 5, 5), "c5_b": (120,),
            "f6_w": (84, 120), "f6_b": (84,),
            "cls_w": (10, 84), "cls_b": (10,),
        }
        for key, shape in expected.items():
            self.assertEqual(self.params[key].shape, shape, key)
            self.assertEqual(
                self.params[key].dtype,
                np.int8 if key.endswith("_w") else np.int32,
                key,
            )

    def test_trained_c3_respects_the_sparse_table(self) -> None:
        """Unconnected C3 taps must be exactly zero, not merely small.

        The RTL does not multiply these at all. A trained weight surviving in a
        connection the hardware cannot express would make the Python model and
        the accelerator disagree for a reason no arithmetic test would find.
        """
        mask = c3_connectivity()
        dead = self.params["c3_w"][~mask]
        np.testing.assert_array_equal(dead, np.zeros_like(dead))
        live = self.params["c3_w"][mask]
        self.assertTrue(np.any(live != 0), "every connected C3 tap is zero")

    def test_calibrated_shifts_are_legal_and_not_all_seven(self) -> None:
        self.assertEqual(set(self.shifts), {"c1", "c3", "c5", "f6"})
        for name, shift in self.shifts.items():
            self.assertGreaterEqual(shift, 0, name)
            self.assertLessEqual(shift, 31, name)
        # The whole point of calibration: if every layer still wants 7, the
        # calibration step did nothing and DEFAULT_SHIFTS would have done.
        self.assertNotEqual(
            list(self.shifts.values()), [7, 7, 7, 7],
            "calibration produced the flat default; it is not doing any work",
        )

    def test_trained_network_classifies_the_demo_digits(self) -> None:
        """End to end on real digits, through the same model the RTL is checked against."""
        from golden.trained import PARAM_FILE, pad_to_32, quantize_image

        demo = PARAM_FILE.parent / "mnist_demo.npz"
        if not demo.exists():  # pragma: no cover - artifact is committed
            self.skipTest(f"{demo} is missing")
        with np.load(demo) as archive:
            images = quantize_image(pad_to_32(archive["images"]))
            labels = archive["labels"].astype(int)

        predictions = np.array(
            [
                deploy_forward_int8(image, self.params, shifts=self.shifts)["prediction"]
                for image in images
            ]
        )
        correct = int((predictions == labels).sum())
        self.assertGreaterEqual(
            correct,
            len(labels) - 1,
            f"trained network got {correct}/{len(labels)} demo digits right",
        )

    def test_fast_int8_path_is_bit_identical_to_the_reference(self) -> None:
        """The vectorised forward used for accuracy must equal deploy_forward_int8.

        train_lenet5.evaluate_int8 scores 10,000 images with a batched im2col
        path, because the reference does its convolution in a Python triple
        loop and would take hours. That makes the published accuracy figure a
        claim about the *fast* function. This is the test that ties the two
        together; without it the headline number is unverified.
        """
        from golden.trained import PARAM_FILE, pad_to_32, quantize_image
        from golden.train_lenet5 import int8_forward_batch

        demo = PARAM_FILE.parent / "mnist_demo.npz"
        if not demo.exists():  # pragma: no cover - artifact is committed
            self.skipTest(f"{demo} is missing")
        with np.load(demo) as archive:
            images = quantize_image(pad_to_32(archive["images"]))

        fast = int8_forward_batch(images[:, None, :, :], self.params, self.shifts)
        reference = np.array(
            [
                deploy_forward_int8(image, self.params, shifts=self.shifts)["prediction"]
                for image in images
            ]
        )
        np.testing.assert_array_equal(fast, reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)

