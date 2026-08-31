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

void synchronize_stream(std::uintptr_t stream_handle) {
  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  CS_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuspa

NB_MODULE(_core, m) {
  m.doc() = "cuspa: self-contained CUDA spatial index + point assignment";

  cuspa::bind_index_ops(m);
  cuspa::bind_query_ops(m);
  cuspa::bind_aggregate_ops(m);
  m.def("synchronize_stream", &cuspa::synchronize_stream, nb::arg("stream") = std::uintptr_t{0});
}
