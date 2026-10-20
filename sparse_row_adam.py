"""Bounded sparse updates for 64-wide bf16 CPU memory tables."""

import ctypes
import functools
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np


class StepOptions(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_size_t),
        ("lookups", ctypes.c_size_t),
        ("learning_rate", ctypes.c_double),
        ("weight_decay", ctypes.c_double),
    ]


@functools.cache
def kernel() -> Callable[..., int]:
    root = Path(__file__).resolve().parent
    configured = os.environ.get("SPARSE_ROW_ADAM_LIBRARY")
    path = Path(configured) if configured else root / ".sparse-row-adam/lib/libsparse_row_adam.so"
    if not configured and not path.is_file():
        path = root / "target/release/libsparse_row_adam.so"
    if not path.is_file():
        raise RuntimeError(
            "Build with cargo build --release or nix build path:. --out-link .sparse-row-adam. "
            "Set SPARSE_ROW_ADAM_LIBRARY for an installed library"
        )
    library = ctypes.CDLL(str(path))
    function = library.sparse_row_adam_step
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.POINTER(StepOptions)]
    function.restype = ctypes.c_int64
    return function


def step(
    bits: np.ndarray,
    moments: np.ndarray,
    indices: np.ndarray,
    gradients: np.ndarray,
    learning_rate: float,
    weight_decay: float,
) -> int:
    for name, value in (("bits", bits), ("moments", moments), ("indices", indices), ("gradients", gradients)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Sparse row Adam: {name} must be a NumPy array")
    if bits.ndim != 2 or bits.shape[1] != 64:
        raise ValueError("Sparse row Adam: bits must have shape (rows, 64)")
    if not 0 < len(bits) <= 128_000_000 or indices.size > 1_048_576:
        raise ValueError("Sparse row Adam: table or lookup count exceeds workspace bounds")
    if gradients.size != indices.size * 64:
        raise ValueError("Sparse row Adam: each lookup needs 64 gradient values")
    if bits.dtype != np.uint16 or moments.dtype != np.float32:
        raise TypeError("Sparse row Adam: bits must be uint16 and moments must be float32")
    if moments.shape != (len(bits),):
        raise ValueError("Sparse row Adam: moments must have one value per table row")
    for name, value in (("bits", bits), ("moments", moments)):
        if not value.flags.c_contiguous or not value.flags.writeable or not value.flags.aligned:
            raise ValueError(f"Sparse row Adam: {name} must be contiguous, writable and aligned")
    if indices.dtype.kind not in "iu":
        raise TypeError("Sparse row Adam: indices must be integers")
    if gradients.dtype.kind not in "fiu":
        raise TypeError("Sparse row Adam: gradients must be real numbers")
    indices = np.require(indices, dtype=np.int64, requirements=["C", "A"]).reshape(-1)
    gradients = np.require(gradients, dtype=np.float32, requirements=["C", "A"]).reshape(-1, 64)
    if np.shares_memory(bits, moments):
        raise ValueError("Sparse row Adam: bits and moments must not overlap")
    for value in (indices, gradients):
        if np.shares_memory(bits, value) or np.shares_memory(moments, value):
            raise ValueError("Sparse row Adam: inputs and writable state must not overlap")
    options = StepOptions(len(bits), len(indices), learning_rate, weight_decay)
    result = kernel()(
        bits.ctypes.data, moments.ctypes.data, indices.ctypes.data, gradients.ctypes.data, ctypes.byref(options)
    )
    if result < 0:
        messages = {
            -1: "invalid dimensions/rates",
            -2: "row index outside table",
            -3: "nonfinite/invalid update state",
            -4: "workspace allocation or hash insertion failed",
        }
        raise ValueError("Sparse row Adam: " + messages.get(result, f"native error {result}"))
    return result
