/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include "../common/ndarray_utils.cuh"
#include "../ops.hpp"
#include "kernels_aggregate.cuh"

namespace nb = nanobind;

namespace cuspa {

template <typename Device>
std::int64_t aggregate_to_cells_entry(DeviceArray<Device> cell_ids,
                                      DeviceArray<Device> gene_ids,
                                      std::int32_t num_cells,
                                      DeviceArray<Device> out_indptr,
                                      DeviceArray<Device> out_indices_scratch,
                                      DeviceArray<Device> out_data_scratch,
                                      std::uintptr_t stream_handle) {
  _require_1d(cell_ids, "cell_ids");
  _require_1d(gene_ids, "gene_ids");
  _require_1d(out_indptr, "out_indptr");
  _require_1d(out_indices_scratch, "out_indices_scratch");
  _require_1d(out_data_scratch, "out_data_scratch");
  _require_int32(cell_ids, "cell_ids");
  _require_int32(gene_ids, "gene_ids");
  _require_int32(out_indptr, "out_indptr");
  _require_int32(out_indices_scratch, "out_indices_scratch");
  _require_int32(out_data_scratch, "out_data_scratch");

  auto const N = static_cast<std::int64_t>(cell_ids.shape(0));
  if (N > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("aggregate_to_cells supports at most 2^31-1 input points");
  }
  if (static_cast<std::int64_t>(gene_ids.shape(0)) != N) {
    throw std::invalid_argument("cell_ids and gene_ids must have the same length");
  }
  if (num_cells < 0) {
    throw std::invalid_argument("num_cells must be non-negative");
  }
  if (static_cast<std::int64_t>(out_indptr.shape(0)) != num_cells + 1) {
    throw std::invalid_argument("out_indptr must have length num_cells + 1");
  }
  if (static_cast<std::int64_t>(out_indices_scratch.shape(0)) < N ||
      static_cast<std::int64_t>(out_data_scratch.shape(0)) < N) {
    throw std::invalid_argument(
        "out_indices_scratch and out_data_scratch must each be sized at least N");
  }

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  return run_aggregate_to_cells(static_cast<std::int32_t const*>(cell_ids.data()),
                                static_cast<std::int32_t const*>(gene_ids.data()),
                                N,
                                num_cells,
                                static_cast<std::int32_t*>(out_indptr.data()),
                                static_cast<std::int32_t*>(out_indices_scratch.data()),
                                static_cast<std::int32_t*>(out_data_scratch.data()),
                                stream);
}

template <typename Device>
void bind_aggregate_device(nb::module_& m) {
  m.def("aggregate_to_cells",
        &cuspa::aggregate_to_cells_entry<Device>,
        nb::arg("cell_ids"),
        nb::arg("gene_ids"),
        nb::arg("num_cells"),
        nb::arg("out_indptr"),
        nb::arg("out_indices_scratch"),
        nb::arg("out_data_scratch"),
        nb::arg("stream") = std::uintptr_t{0},
        "Build a CSR count matrix over (cell_id, gene_id) pairs. Returns K, "
        "the number of unique (cell, gene) pairs — caller slices the scratch "
        "buffers to [:K] to get the CSR indices/data arrays.");
}

void bind_aggregate_ops(nb::module_& m) {
  bind_aggregate_device<nb::device::cuda>(m);
  bind_aggregate_device<nb::device::cuda_managed>(m);
}

}  // namespace cuspa
