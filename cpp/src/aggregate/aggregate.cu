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

// The accumulated CSR and the merge output use int32 or int64 for indptr and
// indices; a batch CSR is always int32.
template <typename Device>
static bool _is_int64_index(DeviceArray<Device> const& indptr,
                            DeviceArray<Device> const& indices,
                            char const* name) {
  auto const dt = indptr.dtype();
  if (dt != nb::dtype<std::int32_t>() && dt != nb::dtype<std::int64_t>())
    throw std::invalid_argument(std::string{name} + " indptr must be int32 or int64");
  if (indices.dtype() != dt)
    throw std::invalid_argument(std::string{name} + " indptr and indices must have the same dtype");
  return dt == nb::dtype<std::int64_t>();
}

// Shared validation for the two CSR-merge entry points. Returns num_rows.
template <typename Device>
std::int32_t _require_csr_pair(DeviceArray<Device> const& a_indptr,
                               DeviceArray<Device> const& a_indices,
                               DeviceArray<Device> const& b_indptr,
                               DeviceArray<Device> const& b_indices,
                               DeviceArray<Device> const& rank_scratch) {
  _require_1d(a_indptr, "a_indptr");
  _require_1d(a_indices, "a_indices");
  _require_1d(b_indptr, "b_indptr");
  _require_1d(b_indices, "b_indices");
  _require_1d(rank_scratch, "rank_scratch");
  _require_int32(b_indptr, "b_indptr");
  _require_int32(b_indices, "b_indices");
  _require_int32(rank_scratch, "rank_scratch");
  if (a_indptr.shape(0) == 0 || a_indptr.shape(0) != b_indptr.shape(0)) {
    throw std::invalid_argument("a_indptr and b_indptr must have the same non-zero length");
  }
  _checked_int32_size(b_indices.shape(0), "b_indices");
  if (static_cast<std::int64_t>(rank_scratch.shape(0)) !=
      csr_merge_scratch_size(static_cast<std::int64_t>(b_indices.shape(0)))) {
    throw std::invalid_argument("rank_scratch must have length 2 * ceil(len(b_indices) / 32) + 1");
  }
  return _checked_int32_size(a_indptr.shape(0) - 1, "num_rows");
}

template <typename Device>
std::int64_t merge_csr_count_entry(DeviceArray<Device> a_indptr,
                                   DeviceArray<Device> a_indices,
                                   DeviceArray<Device> b_indptr,
                                   DeviceArray<Device> b_indices,
                                   DeviceArray<Device> out_rank_scratch,
                                   std::uintptr_t stream_handle) {
  auto const num_rows =
      _require_csr_pair(a_indptr, a_indices, b_indptr, b_indices, out_rank_scratch);
  auto const a_int64 = _is_int64_index(a_indptr, a_indices, "a");
  if (!a_int64)
    _checked_int32_size(a_indices.shape(0), "a_indices");
  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto run          = [&](auto const* a_ip, auto const* a_ix) {
    return run_csr_merge_count(a_ip,
                               a_ix,
                               static_cast<std::int64_t>(a_indices.shape(0)),
                               static_cast<std::int32_t const*>(b_indptr.data()),
                               static_cast<std::int32_t const*>(b_indices.data()),
                               static_cast<std::int64_t>(b_indices.shape(0)),
                               num_rows,
                               static_cast<std::int32_t*>(out_rank_scratch.data()),
                               stream);
  };
  if (a_int64)
    return run(static_cast<std::int64_t const*>(a_indptr.data()),
               static_cast<std::int64_t const*>(a_indices.data()));
  return run(static_cast<std::int32_t const*>(a_indptr.data()),
             static_cast<std::int32_t const*>(a_indices.data()));
}

template <typename Device>
void merge_csr_fill_entry(DeviceArray<Device> a_indptr,
                          DeviceArray<Device> a_indices,
                          DeviceArray<Device> a_data,
                          DeviceArray<Device> b_indptr,
                          DeviceArray<Device> b_indices,
                          DeviceArray<Device> b_data,
                          DeviceArray<Device> rank_scratch,
                          DeviceArray<Device> out_indptr,
                          DeviceArray<Device> out_indices,
                          DeviceArray<Device> out_data,
                          std::uintptr_t stream_handle) {
  auto const num_rows = _require_csr_pair(a_indptr, a_indices, b_indptr, b_indices, rank_scratch);
  auto const a_int64  = _is_int64_index(a_indptr, a_indices, "a");
  auto const o_int64  = _is_int64_index(out_indptr, out_indices, "out");
  if (a_int64 && !o_int64)
    throw std::invalid_argument("out indices must be int64 when a indices are int64");
  _require_1d(a_data, "a_data");
  _require_1d(b_data, "b_data");
  _require_1d(out_indptr, "out_indptr");
  _require_1d(out_indices, "out_indices");
  _require_1d(out_data, "out_data");
  _require_int32(a_data, "a_data");
  _require_int32(b_data, "b_data");
  _require_int32(out_data, "out_data");
  if (a_data.shape(0) != a_indices.shape(0) || b_data.shape(0) != b_indices.shape(0)) {
    throw std::invalid_argument("CSR data and indices must have the same length");
  }
  if (out_indptr.shape(0) != a_indptr.shape(0)) {
    throw std::invalid_argument("out_indptr must have length num_rows + 1");
  }
  // The caller sizes the output from merge_csr_count; an undersized buffer
  // would be written out of bounds. The exact size is checked on the host
  // side, so this only rules out buffers that cannot hold any valid result.
  if (!o_int64)
    _checked_int32_size(out_indices.shape(0), "out_indices");
  if (out_data.shape(0) != out_indices.shape(0) || out_indices.shape(0) < a_indices.shape(0) ||
      out_indices.shape(0) < b_indices.shape(0) ||
      out_indices.shape(0) > a_indices.shape(0) + b_indices.shape(0)) {
    throw std::invalid_argument("out_indices and out_data must be sized by merge_csr_count");
  }

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto run          = [&](auto const* a_ip, auto const* a_ix, auto* o_ip, auto* o_ix) {
    run_csr_merge_fill(a_ip,
                       a_ix,
                       static_cast<std::int32_t const*>(a_data.data()),
                       static_cast<std::int64_t>(a_indices.shape(0)),
                       static_cast<std::int32_t const*>(b_indptr.data()),
                       static_cast<std::int32_t const*>(b_indices.data()),
                       static_cast<std::int32_t const*>(b_data.data()),
                       static_cast<std::int64_t>(b_indices.shape(0)),
                       num_rows,
                       static_cast<std::int32_t const*>(rank_scratch.data()),
                       o_ip,
                       o_ix,
                       static_cast<std::int32_t*>(out_data.data()),
                       stream);
  };
  auto* o64_ip = static_cast<std::int64_t*>(out_indptr.data());
  auto* o64_ix = static_cast<std::int64_t*>(out_indices.data());
  if (a_int64) {
    run(static_cast<std::int64_t const*>(a_indptr.data()),
        static_cast<std::int64_t const*>(a_indices.data()),
        o64_ip,
        o64_ix);
  } else if (o_int64) {
    run(static_cast<std::int32_t const*>(a_indptr.data()),
        static_cast<std::int32_t const*>(a_indices.data()),
        o64_ip,
        o64_ix);
  } else {
    run(static_cast<std::int32_t const*>(a_indptr.data()),
        static_cast<std::int32_t const*>(a_indices.data()),
        static_cast<std::int32_t*>(out_indptr.data()),
        static_cast<std::int32_t*>(out_indices.data()));
  }
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
  m.def("merge_csr_count",
        &cuspa::merge_csr_count_entry<Device>,
        nb::arg("a_indptr"),
        nb::arg("a_indices"),
        nb::arg("b_indptr"),
        nb::arg("b_indices"),
        nb::arg("out_rank_scratch"),
        nb::arg("stream") = std::uintptr_t{0},
        "First pass of an exact sum of two canonical CSR matrices; A may use "
        "int32 or int64 indices, B is int32. "
        "Fills out_rank_scratch, sized 2 * ceil(nnz_b / 32) + 1, and returns the result's number "
        "of non-zeros.");
  m.def("merge_csr_fill",
        &cuspa::merge_csr_fill_entry<Device>,
        nb::arg("a_indptr"),
        nb::arg("a_indices"),
        nb::arg("a_data"),
        nb::arg("b_indptr"),
        nb::arg("b_indices"),
        nb::arg("b_data"),
        nb::arg("rank_scratch"),
        nb::arg("out_indptr"),
        nb::arg("out_indices"),
        nb::arg("out_data"),
        nb::arg("stream") = std::uintptr_t{0},
        "Second pass of merge_csr_count: writes the canonical sum. Raises "
        "OverflowError if a count exceeds 2^31-1.");
}

void bind_aggregate_ops(nb::module_& m) {
  bind_aggregate_device<nb::device::cuda>(m);
  bind_aggregate_device<nb::device::cuda_managed>(m);
}

}  // namespace cuspa
