cmake_minimum_required(VERSION 3.15...3.27)
project(GEMMul8 LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "CUDA is required to build GEMMul8, but no CUDA compiler (nvcc) was found.\n"
                      "Please ensure the CUDA Toolkit is installed and nvcc is in your PATH.\n"
                      "On a cluster, you may need to run 'module load cuda'.")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
    OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

nanobind_add_module(gemmul8_ffi dist_name gemmul8_ffi ext/pybind.cpp ext/gemmul8.cu ext/gemmul8_ffi_cuda.cu)

# Add include directories
target_include_directories(gemmul8_ffi PRIVATE ext/include ${JAXLIB_INCLUDE_DIR})
