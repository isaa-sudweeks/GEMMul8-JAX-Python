#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "gemmul8.hpp"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

struct ThreadLocalState {
  cublasHandle_t cublas = nullptr;

  void *work_base = nullptr; // single big allocation
  size_t work_bytes = 0;

  // cached split sizes for current allocation regime
  size_t workA_bytes = 0;
  size_t workB_bytes = 0;
};

static thread_local ThreadLocalState tls;

static cublasHandle_t GetCublasHandle(cudaStream_t stream) {
  if (!tls.cublas) {
    cublasCreate(&tls.cublas);
  }
  cublasSetStream(tls.cublas, stream);
  return tls.cublas;
}

// NOTE: This is a grow-only allocator. We never free during runtime.
// (Safe for XLA long runs; you can add an explicit finalizer later if desired.)
static ffi::Error EnsureWorkspace(cudaStream_t stream, size_t required_bytes,
                                  size_t workA_bytes, size_t workB_bytes) {

  if (tls.work_base && tls.work_bytes >= required_bytes &&
      tls.workA_bytes == workA_bytes && tls.workB_bytes == workB_bytes) {
    return ffi::Error::Success();
  }

  // If we already had one but sizes changed, just grow/realloc.
  // For simplicity we leak the old buffer (rare if shapes are static under
  // jit).
  void *new_ptr = nullptr;
  cudaError_t st = cudaMallocAsync(&new_ptr, required_bytes, stream);
  if (st != cudaSuccess) {
    return ffi::Error::Internal(
        "cudaMallocAsync failed for GEMMul8 workspace.");
  }

  tls.work_base = new_ptr;
  tls.work_bytes = required_bytes;
  tls.workA_bytes = workA_bytes;
  tls.workB_bytes = workB_bytes;
  return ffi::Error::Success();
}

ffi::Error Gemmul8GemmF64Impl(
    cudaStream_t stream,
    // attrs
    int32_t transa,              // 0: N, 1: T
    int32_t transb,              // 0: N, 1: T
    int32_t num_moduli,          // 2..20 (else GEMMul8 may fall back)
    int32_t fastmode,            // 0/1
    int32_t enable_skip_scalA,   // 0/1
    int32_t enable_skip_scalB,   // 0/1
    int32_t skip_scalA,          // 0/1 (only meaningful if enable_skip_scalA=1)
    int32_t skip_scalB,          // 0/1
    int32_t use_extra_workspace, // 0/1: template parameter UseExtraWorkspace
    double alpha, double beta,
    // buffers
    ffi::Buffer<ffi::F64> A, ffi::Buffer<ffi::F64> B,
    ffi::ResultBuffer<ffi::F64> C) {
  if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
    return ffi::Error::InvalidArgument("Only rank-2 matrices supported.");
  }

  // Output shape
  const int64_t m = C->dimensions()[0];
  const int64_t n = C->dimensions()[1];

  // Infer k
  const int64_t k = (transa == 0) ? A.dimensions()[1] : A.dimensions()[0];

  // Basic consistency check (shape only)
  const int64_t kB = (transb == 0) ? B.dimensions()[0] : B.dimensions()[1];
  if (kB != k) {
    return ffi::Error::InvalidArgument(
        "Inner dimension mismatch between A and B.");
  }

  // cuBLAS handle on XLA-provided stream
  cublasHandle_t handle = GetCublasHandle(stream);

  const cublasOperation_t opA = (transa == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opB = (transb == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // NOTE: GEMMul8 API expects leading dims like cuBLAS.
  // This minimal version assumes column-major interpretation.
  // If you pass JAX arrays, use jnp.asfortranarray-like layouts or accept a
  // transpose convention.
  const int64_t lda = (opA == CUBLAS_OP_N) ? m : k;
  const int64_t ldb = (opB == CUBLAS_OP_N) ? k : n;
  const int64_t ldc = m;

  // Workspace sizing (reserve extra space for skip-scaling)
  size_t workSizeA = 0, workSizeB = 0;

  // UseExtraWorkspace is a compile-time bool in GEMMul8. We dispatch both
  // cases.
  size_t total_ws = 0;
  if (use_extra_workspace) {
    total_ws =
        gemmul8::workSize</*is_Complex=*/false, /*UseExtraWorkspace=*/true>(
            (size_t)m, (size_t)n, (size_t)k, (unsigned)num_moduli,
            (bool)(enable_skip_scalA != 0), (bool)(enable_skip_scalB != 0),
            &workSizeA, &workSizeB);
  } else {
    total_ws =
        gemmul8::workSize</*is_Complex=*/false, /*UseExtraWorkspace=*/false>(
            (size_t)m, (size_t)n, (size_t)k, (unsigned)num_moduli,
            (bool)(enable_skip_scalA != 0), (bool)(enable_skip_scalB != 0),
            &workSizeA, &workSizeB);
  }

  // Ensure we have a persistent buffer large enough
  auto err = EnsureWorkspace(stream, total_ws, workSizeA, workSizeB);
  if (!err.ok())
    return err;

  // Split workspace (matches README pattern)
  auto *base = reinterpret_cast<int8_t *>(tls.work_base);
  void *workA = base;             // fixed workspace for A
  void *workB = base + workSizeA; // fixed workspace for B
  void *work_rem = base + workSizeA + workSizeB;

  // Call GEMMul8 (DGEMM emulation)
  if (use_extra_workspace) {
    (void)gemmul8::gemm<double, /*UseExtraWorkspace=*/true>(
        handle, opA, opB, (size_t)m, (size_t)n, (size_t)k, &alpha,
        (const double *)A.data(), (size_t)lda, (const double *)B.data(),
        (size_t)ldb, &beta, (double *)C->data(), (size_t)ldc,
        (unsigned)num_moduli, (bool)(fastmode != 0), work_rem, workA, workB,
        (bool)(enable_skip_scalA != 0), (bool)(enable_skip_scalB != 0),
        (bool)(skip_scalA != 0), (bool)(skip_scalB != 0));
  } else {
    (void)gemmul8::gemm<double, /*UseExtraWorkspace=*/false>(
        handle, opA, opB, (size_t)m, (size_t)n, (size_t)k, &alpha,
        (const double *)A.data(), (size_t)lda, (const double *)B.data(),
        (size_t)ldb, &beta, (double *)C->data(), (size_t)ldc,
        (unsigned)num_moduli, (bool)(fastmode != 0), work_rem, workA, workB,
        (bool)(enable_skip_scalA != 0), (bool)(enable_skip_scalB != 0),
        (bool)(skip_scalA != 0), (bool)(skip_scalB != 0));
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(Gemmul8GemmF64, Gemmul8GemmF64Impl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int32_t>("transa")
                           .Attr<int32_t>("transb")
                           .Attr<int32_t>("num_moduli")
                           .Attr<int32_t>("fastmode")
                           .Attr<int32_t>("enable_skip_scalA")
                           .Attr<int32_t>("enable_skip_scalB")
                           .Attr<int32_t>("skip_scalA")
                           .Attr<int32_t>("skip_scalB")
                           .Attr<int32_t>("use_extra_workspace")
                           .Attr<double>("alpha")
                           .Attr<double>("beta")
                           .Arg<ffi::Buffer<ffi::F64>>() // A
                           .Arg<ffi::Buffer<ffi::F64>>() // B
                           .Ret<ffi::Buffer<ffi::F64>>() // C
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "gemmul8_gemm_f64", "CUDA",
                         Gemmul8GemmF64);
