/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <limits>

#include "../common/cuda_utils.cuh"

namespace cuspa {

// --- Aggregation: (cell_id, gene_id) pairs → sparse CSR counts -------------
//
// Strategy:
//   1. Pack each (cell_id, gene_id) into a uint64 key. Invalid assignments
//      (cell_id < 0) get UINT64_MAX as sentinel so they sort to the end and
//      drop out of the RLE result.
//   2. CUB radix-sort keys (ascending) → groups identical (cell, gene) pairs.
//   3. CUB run-length encode → unique keys + counts. A trailing sentinel run
//      (if any) is stripped.
//   4. Unpack unique keys back into sorted cell_ids + gene_ids. gene_ids and
//      counts are already in CSR column order (sorted within each cell).
//   5. Build indptr by lower_bound: indptr[c] = first index where
//      unique_cells[idx] >= c.  Runs in one kernel, num_cells+1 threads.

constexpr std::uint64_t CS_SENTINEL_KEY = ~std::uint64_t{0};

__global__ void build_pair_keys_kernel(std::int32_t const* cell_ids,
                                       std::int32_t const* gene_ids,
                                       std::int64_t N,
                                       std::uint64_t* keys) {
  auto const i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  auto const c = cell_ids[i];
  if (c < 0) {
    keys[i] = CS_SENTINEL_KEY;
    return;
  }
  auto const g = gene_ids[i];
  // NB: g < 0 is treated as unassigned too (defensive); normal gene IDs fit
  // in int32 positive range.
  if (g < 0) {
    keys[i] = CS_SENTINEL_KEY;
    return;
  }
  keys[i] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(c)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(g));
}

__global__ void decode_unique_keys_kernel(std::uint64_t const* unique_keys,
                                          std::int64_t K,
                                          std::int32_t* out_cells,
                                          std::int32_t* out_genes) {
  auto const i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= K)
    return;
  std::uint64_t const k = unique_keys[i];
  out_cells[i]          = static_cast<std::int32_t>(k >> 32);
  out_genes[i]          = static_cast<std::int32_t>(k & 0xFFFFFFFFu);
}

__global__ void lower_bound_indptr_kernel(std::int32_t const* sorted_cells,
                                          std::int64_t K,
                                          std::int32_t num_cells,
                                          std::int32_t* indptr /* size num_cells+1 */) {
  auto const c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c > num_cells)
    return;
  // Find first index into sorted_cells where sorted_cells[idx] >= c.
  std::int64_t lo = 0, hi = K;
  std::int32_t const target = static_cast<std::int32_t>(c);
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (sorted_cells[mid] < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  indptr[c] = static_cast<std::int32_t>(lo);
}

std::int64_t run_aggregate_to_cells(std::int32_t const* cell_ids,
                                    std::int32_t const* gene_ids,
                                    std::int64_t N,
                                    std::int32_t num_cells,
                                    std::int32_t* out_indptr,  /* (num_cells+1,) */
                                    std::int32_t* out_indices, /* scratch sized N */
                                    std::int32_t* out_data,    /* scratch sized N */
                                    cudaStream_t stream) {
  DeviceScratch scratch{stream};

  if (N == 0) {
    CS_CHECK(cudaMemsetAsync(out_indptr, 0, sizeof(std::int32_t) * (num_cells + 1), stream));
    return 0;
  }

  constexpr int block = 256;

  // 1--3. Pack, radix-sort, then RLE the keys. Scope the sort input/output
  // buffers to their last use: cudaFreeAsync preserves stream ordering, so
  // each large N-sized key buffer becomes reusable before the decode phase.
  DeviceBuffer<std::uint64_t> d_unique_keys{static_cast<std::size_t>(N), stream};
  DeviceBuffer<std::int32_t> d_run_counts{static_cast<std::size_t>(N), stream};
  DeviceBuffer<std::int64_t> d_num_runs_dev{1, stream};
  {
    DeviceBuffer<std::uint64_t> d_keys_out{static_cast<std::size_t>(N), stream};
    {
      DeviceBuffer<std::uint64_t> d_keys_in{static_cast<std::size_t>(N), stream};
      auto const grid = static_cast<unsigned>((N + block - 1) / block);
      build_pair_keys_kernel<<<grid, block, 0, stream>>>(cell_ids, gene_ids, N, d_keys_in.ptr);
      CS_CHECK(cudaGetLastError());

      // (cell_id, gene_id) packed in cell:high, gene:low, so the sort groups
      // genes within cells. Invalid entries (sentinel) land at the end.
      std::size_t temp_bytes = 0;
      CS_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, temp_bytes, d_keys_in.ptr, d_keys_out.ptr, N, 0, 64, stream));
      scratch.resize(temp_bytes);
      CS_CHECK(cub::DeviceRadixSort::SortKeys(
          scratch.ptr, temp_bytes, d_keys_in.ptr, d_keys_out.ptr, N, 0, 64, stream));
    }

    std::size_t temp_bytes = 0;
    CS_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                temp_bytes,
                                                d_keys_out.ptr,
                                                d_unique_keys.ptr,
                                                d_run_counts.ptr,
                                                d_num_runs_dev.ptr,
                                                N,
                                                stream));
    scratch.resize(temp_bytes);
    CS_CHECK(cub::DeviceRunLengthEncode::Encode(scratch.ptr,
                                                temp_bytes,
                                                d_keys_out.ptr,
                                                d_unique_keys.ptr,
                                                d_run_counts.ptr,
                                                d_num_runs_dev.ptr,
                                                N,
                                                stream));
  }

  // Pull num_runs back to the host.  One sync — the user will likely sync
  // after anyway since the output is a new CSR.
  std::int64_t num_runs_host = 0;
  CS_CHECK(cudaMemcpyAsync(
      &num_runs_host, d_num_runs_dev.ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream));
  CS_CHECK(cudaStreamSynchronize(stream));

  // Strip the trailing sentinel run, if any.  It's definitely at index
  // num_runs-1 because the sentinel is the largest uint64.
  std::int64_t K = num_runs_host;
  if (K > 0) {
    std::uint64_t last_key_host = 0;
    CS_CHECK(cudaMemcpyAsync(&last_key_host,
                             d_unique_keys.ptr + (K - 1),
                             sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CS_CHECK(cudaStreamSynchronize(stream));
    if (last_key_host == CS_SENTINEL_KEY)
      K -= 1;
  }

  if (K == 0) {
    CS_CHECK(cudaMemsetAsync(out_indptr, 0, sizeof(std::int32_t) * (num_cells + 1), stream));
  } else {
    // 4. Decode unique keys into sorted_cells (temp) + out_indices (gene).
    DeviceBuffer<std::int32_t> d_sorted_cells{static_cast<std::size_t>(K), stream};
    auto const dg = static_cast<unsigned>((K + block - 1) / block);
    decode_unique_keys_kernel<<<dg, block, 0, stream>>>(
        d_unique_keys.ptr, K, d_sorted_cells.ptr, out_indices);
    CS_CHECK(cudaGetLastError());

    // Counts → out_data (D-to-D copy; CUB RLE already sized these correctly
    // before the trim, and the trimmed tail we're dropping is past K).
    CS_CHECK(cudaMemcpyAsync(
        out_data, d_run_counts.ptr, sizeof(std::int32_t) * K, cudaMemcpyDeviceToDevice, stream));

    // 5. Build indptr via lower_bound, one thread per row.
    auto const ig = static_cast<unsigned>((num_cells + 1 + block - 1) / block);
    lower_bound_indptr_kernel<<<ig, block, 0, stream>>>(
        d_sorted_cells.ptr, K, num_cells, out_indptr);
    CS_CHECK(cudaGetLastError());
  }

  return K;
}

}  // namespace cuspa
