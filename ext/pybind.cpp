#include <nanobind/nanobind.h>
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

extern "C" XLA_FFI_Error* gemmul8_gemm_f64(XLA_FFI_CallFrame*);

template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
  return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(gemmul8_ffi, m) {
  m.def("gemmul8_gemm_f64", []() { return EncapsulateFfiCall(gemmul8_gemm_f64); });
}
