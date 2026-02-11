# GEMMul8

GEMMul8 is a custom JAX CUDA primitive for optimized General Matrix Multiply (GEMM) operations, specifically designed for double-precision (`float64`) computations using modular arithmetic techniques.

It exposes a `gemmul8_dgemm` function accessible through JAX's FFI (Foreign Function Interface), allowing for integration into JAX-based high-performance computing workflows.

## Features

- **Custom CUDA Kernel**: optimized for specific modular arithmetic workloads.
- **JAX Integration**: seamless usage with JAX arrays and transformations (e.g., `vmap` support via `broadcast_all`).
- **Configurable Parameters**: supports various tuning parameters like `num_moduli`, `fastmode`, and workspace usage.

## Installation

This project is structured as a subdirectory within the repository. To install it directly from GitHub using `pip`, you must specify the subdirectory.

### From GitHub

```bash
pip install "git+https://github.com/isaa-sudweeks/GEMMul8-JAX-Python.git#subdirectory=GEMMul8"
```

For Hopper GPUs (H100/H200), set CUDA architecture explicitly during build:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" \
pip install "git+https://github.com/isaa-sudweeks/GEMMul8-JAX-Python.git#subdirectory=GEMMul8"
```

### From Local Source

If you have cloned the repository, navigate to the `GEMMul8` directory and run:

```bash
pip install .
```

On Hopper GPUs (H100/H200), use:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" pip install .
```

`pip install` now declares runtime dependencies on `jax[cuda12]` and `jaxlib`,
so CUDA-enabled JAX packages are installed automatically when available for your
platform.

## Usage

```python
import jax
import jax.numpy as jnp
from GEMMul8.api import gemmul8_dgemm

# Example usage
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
A = jax.random.normal(k1, (128, 128), dtype=jnp.float64)
B = jax.random.normal(k2, (128, 128), dtype=jnp.float64)

# Perform matrix multiplication using GEMMul8
C = gemmul8_dgemm(A, B)
```

`gemmul8_dgemm` auto-registers the CUDA FFI target on first use.
