"""Synthetic one-core CPU benchmark with no tokenizer, model, GPU or private data."""

import argparse
import json
import os
import resource
import statistics
from pathlib import Path
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from reference import reference, round_to_bfloat16_bits
from sparse_row_adam import step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookups", type=int, default=524288)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.lookups <= 1048576:
        parser.error("--lookups must be between 1 and 1048576")
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[-1]})
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    cases = []
    for name in ("hot_rows", "distinct_rows"):
        rng = np.random.default_rng(64)
        rows = min(8192, args.lookups) if name == "hot_rows" else args.lookups
        indices = (
            rng.integers(0, rows, args.lookups, dtype=np.int64)
            if name == "hot_rows"
            else rng.permutation(args.lookups).astype(np.int64)
        )
        cases.append(measure(name, indices, rows, rng))
    result = {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "numpy_version": np.__version__,
        "gpu_used": False,
        "address_space_limit_gib": 8,
        "caveats": ["Synthetic gradients and small table working sets, not full-table or end-to-end training timings"],
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def measure(name: str, indices: np.ndarray, rows: int, rng: np.random.Generator) -> dict:
    gradients = rng.normal(0, 1e-4, (len(indices), 64)).astype(np.float32)
    initial = round_to_bfloat16_bits(rng.normal(0, 0.01, (rows, 64)).astype(np.float32))
    moments = rng.uniform(0, 1e-6, rows).astype(np.float32)
    timings, outputs = {}, []
    for label, function in (("numpy", reference), ("rust", step)):
        bits, state = initial.copy(), moments.copy()
        function(bits, state, indices, gradients, 0.01, 0.025)
        elapsed = []
        for _ in range(3):
            bits[:], state[:] = initial, moments
            started = perf_counter()
            touched = function(bits, state, indices, gradients, 0.01, 0.025)
            elapsed.append(perf_counter() - started)
        outputs.append((bits, state))
        timings[label] = {"seconds": elapsed, "median_seconds": statistics.median(elapsed), "rows_updated": touched}
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    result = {
        "name": name,
        "lookups": len(indices),
        "rows": rows,
        "table_mib": initial.nbytes / 1024**2,
        "exact_table_and_moment_parity": True,
        **timings,
        "speedup": timings["numpy"]["median_seconds"] / timings["rust"]["median_seconds"],
    }
    print(json.dumps({"case": name, "speedup": result["speedup"]}), flush=True)
    return result


if __name__ == "__main__":
    main()
