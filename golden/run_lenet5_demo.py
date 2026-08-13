#!/usr/bin/env python3
"""Exercise every canonical LeNet-5 layer using deterministic parameters."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golden.lenet5 import CANONICAL_LAYERS, forward_canonical, random_canonical_parameters


def main() -> None:
    rng = np.random.default_rng(1998)
    image = rng.normal(0.0, 0.25, size=(32, 32))
    result = forward_canonical(image, random_canonical_parameters())

    print("Canonical LeNet-5 golden-model shape check")
    for spec in CANONICAL_LAYERS[1:-1]:
        actual = result[spec.name]
        print(
            f"{spec.name:>2}: shape={tuple(actual.shape)!s:<14} "
            f"parameters={spec.trainable_parameters}"
        )
        if tuple(actual.shape) != spec.output_shape:
            raise RuntimeError(f"{spec.name} shape mismatch")
    print(f"RBF scores: {np.array2string(result['scores'], precision=4)}")
    print(f"Demo prediction (untrained weights): {result['prediction']}")


if __name__ == "__main__":
    main()
