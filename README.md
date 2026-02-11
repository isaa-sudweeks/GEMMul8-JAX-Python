# GEMMul8

GEMMul8 is a custom JAX CUDA primitive for optimized General Matrix Multiply (GEMM) operations, specifically designed for double-precision (`float64`) computations using modular arithmetic techniques.

It exposes a `gemmul8_dgemm` function accessible through JAX's FFI (Foreign Function Interface), allowing for integration into JAX-based high-performance computing workflows.

## Features

- **Custom CUDA Kernel**: optimized for specific modular arithmetic workloads.
- **JAX Integration**: seamless usage with JAX arrays and transformations (e.g., `vmap` support via `broadcast_all`).
- **Configurable Parameters**: supports various tuning parameters like `num_moduli`, `fastmode`, and workspace usage.

## Installation

### From GitHub

```bash
pip install "git+https://github.com/isaa-sudweeks/GEMMul8-JAX-Python.git"
```

For Hopper GPUs (H100/H200), set CUDA architecture explicitly during build:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" \
pip install "git+https://github.com/isaa-sudweeks/GEMMul8-JAX-Python.git"
```

### From Local Source

If you have cloned the repository, run:

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

## `gemmul8_dgemm` Parameters and Defaults

Signature:

```python
gemmul8_dgemm(
    A, B,
    transa=0,
    transb=0,
    num_moduli=14,
    fastmode=1,
    enable_skip_scalA=0,
    enable_skip_scalB=0,
    skip_scalA=0,
    skip_scalB=0,
    use_extra_workspace=1,
    alpha=1.0,
    beta=0.0,
)
```

- `A` (required): left input matrix (`float64` JAX array).
- `B` (required): right input matrix (`float64` JAX array).
- `transa` (default `0`): whether to transpose `A` before multiplication (`0` no, `1` yes).
- `transb` (default `0`): whether to transpose `B` before multiplication (`0` no, `1` yes).
- `num_moduli` (default `14`): number of modular channels used internally by GEMMul8.
- `fastmode` (default `1`): enables the fast execution path when set to `1`.
- `enable_skip_scalA` (default `0`): enables skip-scaling control for `A`.
- `enable_skip_scalB` (default `0`): enables skip-scaling control for `B`.
- `skip_scalA` (default `0`): skip-scaling selector/value for `A` (used when `enable_skip_scalA=1`).
- `skip_scalB` (default `0`): skip-scaling selector/value for `B` (used when `enable_skip_scalB=1`).
- `use_extra_workspace` (default `1`): use extra temporary workspace to improve kernel behavior/performance.
- `alpha` (default `1.0`): scalar multiplier applied to `A @ B`.
- `beta` (default `0.0`): scalar multiplier for any accumulated output term.
