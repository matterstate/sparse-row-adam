"""Native sparse updates preserve the NumPy optimizer rule, including repeated-row accumulation."""

import subprocess
import sys

import numpy as np
import pytest

from engram_cpu import step
from reference import reference, round_to_bfloat16_bits


@pytest.mark.parametrize("learning_rate,weight_decay", [(0.01, 0.025), (1e-6, 0.0), (0.1, 0.5)])
def test_native_matches_reference_over_repeated_updates(learning_rate, weight_decay):
    rng = np.random.default_rng(87)
    bits = round_to_bfloat16_bits(rng.normal(0, 0.1, (193, 64)).astype(np.float32))
    moments = rng.uniform(0, 0.01, 193).astype(np.float32)
    expected_bits, expected_moments = bits.copy(), moments.copy()
    for _ in range(12):
        indices = rng.integers(0, 173, (2, 39, 8), dtype=np.int64)
        gradients = rng.normal(0, 0.01, (2, 39, 512)).astype(np.float32)
        expected = reference(expected_bits, expected_moments, indices, gradients, learning_rate, weight_decay)
        assert step(bits, moments, indices, gradients, learning_rate, weight_decay) == expected
        np.testing.assert_array_equal(bits, expected_bits)
        np.testing.assert_array_equal(moments, expected_moments)


@pytest.mark.parametrize(
    "failure", ["negative_index", "large_index", "nan_gradient", "negative_moment", "infinite_weight"]
)
def test_invalid_update_never_partially_mutates_table(failure):
    indices = np.array([[0, 1]], np.int64)
    gradients = np.ones((1, 128), np.float32)
    bits = np.zeros((24, 64), np.uint16)
    moments = np.zeros(24, np.float32)
    if failure == "negative_index":
        indices[0, 1] = -1
    elif failure == "large_index":
        indices[0, 1] = len(bits)
    elif failure == "nan_gradient":
        gradients[0, -1] = np.nan
    elif failure == "negative_moment":
        moments[1] = -1
    else:
        bits[1, -1] = 0x7F80
    before = bits.copy(), moments.copy()
    with pytest.raises(ValueError):
        step(bits, moments, indices, gradients, 0.01, 0.025)
    np.testing.assert_array_equal(bits, before[0])
    np.testing.assert_array_equal(moments, before[1])


def test_empty_update_leaves_table_unchanged():
    bits = np.zeros((24, 64), np.uint16)
    moments = np.zeros(24, np.float32)
    assert step(bits, moments, np.empty((0, 8), np.int64), np.empty((0, 512), np.float32), 0.01, 0.025) == 0
    assert not bits.any()
    assert not moments.any()


@pytest.mark.parametrize(
    "failure",
    [
        "shape",
        "weight_dtype",
        "moment_dtype",
        "moment_shape",
        "strided",
        "readonly",
        "unaligned",
        "state_alias",
        "gradient_alias",
        "fractional_indices",
        "complex_gradients",
        "gradient_size",
        "lookup_bound",
    ],
)
def test_unsafe_buffers_are_rejected_without_mutation(failure):
    bits = np.zeros((3, 64), np.uint16)
    moments = np.zeros(3, np.float32)
    indices = np.array([0, 1], np.int64)
    gradients = np.ones((2, 64), np.float32)
    if failure == "shape":
        bits = bits[:, :32]
    elif failure == "weight_dtype":
        bits = bits.astype(np.float32)
    elif failure == "moment_dtype":
        moments = moments.astype(np.float64)
    elif failure == "moment_shape":
        moments = moments[:2]
    elif failure == "strided":
        bits = np.zeros((3, 128), np.uint16)[:, ::2]
    elif failure == "readonly":
        bits.flags.writeable = False
    elif failure == "unaligned":
        bits = np.ndarray((3, 64), np.uint16, buffer=bytearray(385), offset=1)
    elif failure == "state_alias":
        moments = bits.view(np.float32).reshape(-1)[:3]
    elif failure == "gradient_alias":
        gradients = bits.view(np.float32).reshape(-1)[:64].reshape(1, 64)
        indices = indices[:1]
    elif failure == "fractional_indices":
        indices = np.array([0.5, 1], np.float32)
    elif failure == "complex_gradients":
        gradients = gradients.astype(np.complex64)
    elif failure == "gradient_size":
        gradients = gradients[:1]
    else:
        indices = np.broadcast_to(np.array(0, np.int64), (1048577,))
    before = bits.copy(), moments.copy()
    with pytest.raises((TypeError, ValueError)):
        step(bits, moments, indices, gradients, 0.01, 0.025)
    np.testing.assert_array_equal(bits, before[0])
    np.testing.assert_array_equal(moments, before[1])


def test_buffer_validation_survives_python_optimization():
    source = """
import numpy as np
from engram_cpu import step
bits = np.zeros((1, 32), np.uint16)
try:
    step(bits, np.zeros(1, np.float32), np.array([0]), np.ones((1, 64), np.float32), 0.01, 0)
except ValueError:
    pass
else:
    raise RuntimeError("Invalid shape reached the native kernel")
"""
    subprocess.run([sys.executable, "-O", "-c", source], check=True, timeout=20)


@pytest.mark.parametrize("learning_rate,weight_decay", [(0, 0), (float("nan"), 0), (0.01, -1), (0.01, float("inf"))])
def test_invalid_rates_leave_state_unchanged(learning_rate, weight_decay):
    bits = np.zeros((2, 64), np.uint16)
    moments = np.zeros(2, np.float32)
    with pytest.raises(ValueError):
        step(bits, moments, np.array([0]), np.ones((1, 64), np.float32), learning_rate, weight_decay)
    assert not bits.any()
    assert not moments.any()


@pytest.mark.parametrize(
    "gradient_kind", ["positive_infinity", "negative_infinity", "infinity_cancellation", "overflow"]
)
def test_nonfinite_aggregates_never_mutate_state(gradient_kind):
    bits = np.zeros((2, 64), np.uint16)
    moments = np.zeros(2, np.float32)
    indices = np.array([0, 1, 1], np.int64)
    gradients = np.ones((3, 64), np.float32)
    if gradient_kind == "positive_infinity":
        gradients[1, 0] = np.inf
    elif gradient_kind == "negative_infinity":
        gradients[1, 0] = -np.inf
    elif gradient_kind == "infinity_cancellation":
        gradients[1, 0], gradients[2, 0] = np.inf, -np.inf
    else:
        gradients[1:, 0] = np.finfo(np.float32).max
    with pytest.raises(ValueError):
        step(bits, moments, indices, gradients, 0.01, 0.025)
    assert not bits.any()
    assert not moments.any()


def test_bfloat16_overflow_predicate_matches_rounding():
    rng = np.random.default_rng(83)
    edges = np.arange(65536, dtype=np.uint32)[:, None] * np.uint32(65536)
    patterns = np.concatenate(
        [
            (edges + np.array([0, 32767, 32768, 65535], np.uint32)).reshape(-1),
            rng.integers(0, 2**32, 1048576, dtype=np.uint32),
        ]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = round_to_bfloat16_bits(patterns.view(np.float32))
    expected = np.isfinite(patterns.view(np.float32)) & ((rounded & 0x7F80) != 0x7F80)
    np.testing.assert_array_equal((patterns & 0x7FFFFFFF) < 0x7F7F8000, expected)
