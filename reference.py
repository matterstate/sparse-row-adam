"""NumPy reference update for correctness checks and CPU benchmarks."""

import numpy as np

BETA2 = 0.95
EPS = 1e-8
HEAD_WIDTH = 64


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def round_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def reference(
    bits: np.ndarray,
    moments: np.ndarray,
    indices: np.ndarray,
    gradients: np.ndarray,
    learning_rate: float,
    weight_decay: float,
) -> int:
    rows, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    summed = np.zeros((len(rows), HEAD_WIDTH), dtype=np.float32)
    np.add.at(summed, inverse, gradients.reshape(-1, HEAD_WIDTH).astype(np.float32))
    moment = moments[rows] * BETA2 + (1 - BETA2) * (summed * summed).mean(-1)
    moments[rows] = moment
    weights = bfloat16_bits_to_float32(bits[rows])
    update = learning_rate * summed / (np.sqrt(moment)[:, None] + EPS)
    update += np.where(update * weights > 0, weights * (learning_rate * weight_decay), 0.0)
    bits[rows] = round_to_bfloat16_bits((weights - update).astype(np.float32))
    return len(rows)
