/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

#include <cstdint>

#include "common/cuda_utils.cuh"
#include "ops.hpp"

namespace nb = nanobind;

namespace cuspa {

void* allocate_scratch(std::size_t bytes, cudaStream_t stream) {
  void* ptr = nullptr;
  auto err  = cudaMallocAsync(&ptr, bytes, stream);
  if (err == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    nb::gil_scoped_acquire guard;
    nb::module_::import_("cuspa._memory").attr("release_cached_blocks")();
    err = cudaMallocAsync(&ptr, bytes, stream);
  }
  CS_CHECK(err);
  return ptr;
}

void synchronize_stream(std::uintptr_t stream_handle) {
  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  CS_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuspa

NB_MODULE(_core, m) {
  m.doc() = "CUDA spatial indexing and aggregation.";

  cuspa::bind_index_ops(m);
  cuspa::bind_query_ops(m);
  cuspa::bind_aggregate_ops(m);
  cuspa::bind_host_plan(m);
  m.def("synchronize_stream", &cuspa::synchronize_stream, nb::arg("stream") = std::uintptr_t{0});
}
