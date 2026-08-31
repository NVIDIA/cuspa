# Native extension: builds the `_core` nanobind module with CUDA kernels.

find_package(CUDAToolkit REQUIRED)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

# Locate nanobind via its installed Python package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_DIR
  RESULT_VARIABLE _nb_status)
if(NOT _nb_status EQUAL 0)
  message(FATAL_ERROR "Could not locate nanobind. Install it first: pip install nanobind")
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _core
  STABLE_ABI
  LTO
  NB_STATIC
  src/aggregate/aggregate.cu
  src/index/index.cu
  src/module.cpp
  src/query/query.cu
)

target_link_libraries(_core PRIVATE CUDA::cudart)

target_compile_options(_core PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>"
)

# Drop the .so inside the `cuspa` Python package directory, wherever
# scikit-build-core's wheel layout places it.
install(TARGETS _core LIBRARY DESTINATION cuspa)
