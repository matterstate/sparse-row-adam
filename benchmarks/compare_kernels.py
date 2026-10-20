"""Compare native library revisions with alternating order and NumPy parity checks."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import statistics
from pathlib import Path
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from reference import reference, round_to_bfloat16_bits
from sparse_row_adam import StepOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lookups", type=int, default=524288)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.lookups <= 1048576 or not 1 <= args.repeats <= 30:
        parser.error("Expected 1 to 1048576 lookups and 1 to 30 repeats")
    if args.baseline.resolve() == args.candidate.resolve():
        parser.error("Choose two different library files")
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    libraries, functions = {}, {}
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        libraries[name] = ctypes.CDLL(str(path.resolve()))
        functions[name] = libraries[name].sparse_row_adam_step
        functions[name].argtypes = [ctypes.c_void_p] * 4 + [ctypes.POINTER(StepOptions)]
        functions[name].restype = ctypes.c_int64
    results = []
    for name in ("hot", "mixed", "distinct"):
        rng = np.random.default_rng(87)
        rows = min(8192 if name == "hot" else 350000, args.lookups)
        if name == "distinct":
            rows = args.lookups
        indices = (
            rng.permutation(args.lookups).astype(np.int64)
            if name == "distinct"
            else rng.integers(0, rows, args.lookups, dtype=np.int64)
        )
        gradients = rng.normal(0, 1e-4, (args.lookups, 64)).astype(np.float32)
        initial = round_to_bfloat16_bits(rng.normal(0, 0.01, (rows, 64)).astype(np.float32))
        initial_moments = rng.uniform(0, 1e-6, rows).astype(np.float32)
        expected_bits, expected_moments = initial.copy(), initial_moments.copy()
        touched = reference(expected_bits, expected_moments, indices, gradients, 0.01, 0.025)
        bits, moments = initial.copy(), initial_moments.copy()
        options = StepOptions(rows, args.lookups, 0.01, 0.025)
        times = {"baseline": [], "candidate": []}
        for repeat in range(args.repeats + 1):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for backend in order:
                bits[:], moments[:] = initial, initial_moments
                started = perf_counter()
                count = functions[backend](
                    bits.ctypes.data,
                    moments.ctypes.data,
                    indices.ctypes.data,
                    gradients.ctypes.data,
                    ctypes.byref(options),
                )
                elapsed = perf_counter() - started
                if count != touched:
                    raise RuntimeError(f"{backend}: expected {touched} touched rows, received {count}")
                np.testing.assert_array_equal(bits, expected_bits)
                np.testing.assert_array_equal(moments, expected_moments)
                if repeat:
                    times[backend].append(elapsed)
        medians = {backend: statistics.median(values) for backend, values in times.items()}
        result = {
            "case": name,
            "lookups": args.lookups,
            "rows": rows,
            "seconds": times,
            "medians": medians,
            "speedup": medians["baseline"] / medians["candidate"],
            "exact_parity": True,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    report = {
        "cases": results,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "library_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("baseline", args.baseline), ("candidate", args.candidate))
        },
        "scope": "Native C ABI updates with synthetic gradients and small table working sets",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
