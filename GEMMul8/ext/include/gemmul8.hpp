#pragma once
#if defined(__NVCC__)
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#endif
#include <vector>

namespace gemmul8 {

/***
 * workSize returns the required workspace size in bytes.
 */
template <bool is_Complex = false, bool UseExtraWorkspace = true>
size_t
workSize(size_t m,            // Number of rows of C
         size_t n,            // Number of columns of C
         size_t k,            // Inner dimension <= 2^17
         unsigned num_moduli, // #moduli, 2 <= num_moduli <= 20
         bool enable_skip_scalA =
             false, // [option] Reserve extra space for A to allow skip_scalA
         bool enable_skip_scalB =
             false, // [option] Reserve extra space for B to allow skip_scalB
         size_t *workSizeA =
             nullptr, // [option] Output: workspace size used for A8i and sftA
         size_t *workSizeB =
             nullptr // [option] Output: workspace size used for B8i and sftB
);

/***
 * GEMM emulation using INT8 Tensor Cores
 */
#if defined(__NVCC__)
// CUDA definitions are already included via common.hpp or similar, but just in
// case: Check if we need to include anything specific here or if it's handled
// by the caller/common.hpp
#elif defined(__HIPCC__)
// HIP definitions
#else
// Fallback for IDEs and host compilers
using cublasHandle_t = void *;
using cublasOperation_t = int;
using hipblasHandle_t = void *;
using hipblasOperation_t = int;

struct cuFloatComplex {
  float x, y;
};
struct cuDoubleComplex {
  double x, y;
};
struct hipFloatComplex {
  float x, y;
};
struct hipDoubleComplex {
  double x, y;
};
#endif

template <typename T, bool UseExtraWorkspace = true>
std::vector<double> gemm(
#if defined(__NVCC__)
    cublasHandle_t handle, cublasOperation_t op_A, cublasOperation_t op_B,
#elif defined(__HIPCC__)
    hipblasHandle_t handle, hipblasOperation_t op_A, hipblasOperation_t op_B,
#else
    cublasHandle_t handle, cublasOperation_t op_A, cublasOperation_t op_B,
#endif
    size_t m,         // Number of rows of C
    size_t n,         // Number of columns of C
    size_t k,         // Inner dimension <= 2^17
    const T *alpha,   // Scaling factor for op(A)*op(B)
    const T *const A, // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or
                      // lda*m (CUBLAS_OP_T)
    size_t lda,       // Leading dimension of A
    const T *const B, // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or
                      // ldb*k (CUBLAS_OP_T)
    size_t ldb,       // Leading dimension of B
    const T *beta,    // Scaling factor for C
    T *const C,       // 1-D device array of dimensions ldc*n
    size_t ldc,       // Leading dimension of C
    unsigned num_moduli, // #moduli, 2 <= num_moduli <= 20
    bool fastmode,       // false (accurate mode) or true (fast mode)
    void *const work,    // Preallocated workspace
    void *const workA =
        nullptr, // [optional] Separate workspace for A (if nullptr, uses work)
    void *const workB =
        nullptr, // [optional] Separate workspace for B (if nullptr, uses work)
    bool enable_skip_scalA =
        false, // [optional] Enables scaling-skip mechanism for A
    bool enable_skip_scalB =
        false,               // [optional] Enables scaling-skip mechanism for B
    bool skip_scalA = false, // [optional] If true, skip preprocessing for A
    bool skip_scalB = false  // [optional] If true, skip preprocessing for B
);

} // namespace gemmul8
