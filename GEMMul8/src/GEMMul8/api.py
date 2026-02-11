import numpy as np
import jax
import jax.numpy as jnp

try:
    from . import gemmul8_ffi
except ImportError:
    import gemmul8_ffi


def _get_ffi_module():
    # JAX exposes FFI under different modules across releases.
    if hasattr(jax, "ffi"):
        return jax.ffi
    try:
        from jax.extend import ffi as ffi_mod
    except Exception as exc:
        raise RuntimeError(
            "GEMMul8 requires JAX FFI, but neither `jax.ffi` nor "
            "`jax.extend.ffi` is available. Upgrade JAX/JAXLIB to a "
            "version with FFI support."
        ) from exc
    return ffi_mod


def register():
    ffi_mod = _get_ffi_module()
    ffi_mod.register_ffi_target(
        "gemmul8_gemm_f64",
        gemmul8_ffi.gemmul8_gemm_f64(),
        platform="CUDA",
    )


def gemmul8_dgemm(
    A: jax.Array,
    B: jax.Array,
    *,
    transa: int = 0,
    transb: int = 0,
    num_moduli: int = 14,
    fastmode: int = 1,
    enable_skip_scalA: int = 0,
    enable_skip_scalB: int = 0,
    skip_scalA: int = 0,
    skip_scalB: int = 0,
    use_extra_workspace: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
):
    if A.dtype != jnp.float64 or B.dtype != jnp.float64:
        raise TypeError("gemmul8_dgemm: float64 only")

    m = A.shape[1] if transa else A.shape[0]
    kA = A.shape[0] if transa else A.shape[1]
    kB = B.shape[1] if transb else B.shape[0]
    n = B.shape[0] if transb else B.shape[1]
    if kA != kB:
        raise ValueError(f"Inner dims mismatch: {kA} vs {kB}")

    out_type = jax.ShapeDtypeStruct((m, n), jnp.float64)

    ffi_mod = _get_ffi_module()
    call = ffi_mod.ffi_call(
        "gemmul8_gemm_f64",
        out_type,
        vmap_method="broadcast_all",
    )

    return call(
        A, B,
        transa=np.int32(transa),
        transb=np.int32(transb),
        num_moduli=np.int32(num_moduli),
        fastmode=np.int32(fastmode),
        enable_skip_scalA=np.int32(enable_skip_scalA),
        enable_skip_scalB=np.int32(enable_skip_scalB),
        skip_scalA=np.int32(skip_scalA),
        skip_scalB=np.int32(skip_scalB),
        use_extra_workspace=np.int32(use_extra_workspace),
        alpha=np.float64(alpha),
        beta=np.float64(beta),
    )
