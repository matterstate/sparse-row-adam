# engram-cpu

A Rust CPU kernel for sparse updates to 64-wide bf16 memory rows, with a NumPy
adapter and C interface. No Rust dependencies.

Repeated row gradients are accumulated in occurrence order. Only touched rows
change. The update uses FP32 gradients and rowwise second moments, beta2 = 0.95,
epsilon = 1e-8, no first moment or bias correction, and cautious weight decay.
Weights are rounded to bf16.

## Build

```sh
nix build path:. --out-link .engram-cpu
nix develop path:.
```

Or use `cargo build --release` with Rust 1.88 or later. The Python adapter needs
NumPy 2.x. Set `ENGRAM_CPU_LIBRARY` for a library outside the default build
paths. Linux is the tested platform.

## Use

```python
import numpy as np

from engram_cpu import step

bits = np.zeros((1024, 64), dtype=np.uint16)
moments = np.zeros(1024, dtype=np.float32)
indices = np.array([7, 7, 12], dtype=np.int64)
gradients = np.ones((3, 64), dtype=np.float32)
assert step(bits, moments, indices, gradients, 0.01, 0.025) == 2
```

`bits` contains bf16 bit patterns. Weights and moments must be writable,
aligned, contiguous and non-overlapping. See `include/engram_cpu.h` for the C
interface. Indices must be integers and gradients must be real numbers. Do not
mutate or resize input arrays while a call is running.

## Verify

Inside the development shell:

```sh
python -m pytest -q
cargo clippy --all-targets -- -D warnings
PYTHONPATH=. python benchmarks/sparse_update.py --output benchmark-results.json
```

For kernel revisions, `benchmarks/compare_kernels.py` accepts `--baseline` and
`--candidate` library paths and checks parity in every timing round.

Tests require bitwise parity with `reference.py` and reject invalid updates
without partial mutation. Verify parity when changing NumPy or platform.

The single-threaded kernel supports up to 128 million rows and 1,048,576 lookups
per call. Temporary storage scales with the batch and stays below approximately
280 MiB. Learning rate must be in (0, 1\] and weight decay in [0, 1].

Benchmarks use synthetic gradients and small working sets. Full-table
performance and integrated GPU validation are pending. CPU update timings do not
measure overall training speed.

Related work: [Engram](https://github.com/deepseek-ai/Engram) and
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

MIT licensed. See [LICENSE](LICENSE).
