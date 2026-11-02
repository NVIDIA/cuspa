/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <stdexcept>

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
  // Treat negative gene IDs as unassigned.
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

  // Copy num_runs to size the decoded output; this synchronizes the stream.
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

// --- Batch merge: accumulated CSR + batch CSR → their exact sum --------------
//
// A (the accumulator) and B (one batch) have the same shape, sorted genes
// within each row, and no duplicates. Every output entry's position is computed
// directly, so the result is written in canonical order with one thread per
// input entry and no concatenate-then-sort buffer.
//
// B is always int32 because a batch has at most 2^31-1 points. A and the output
// use IA and IO for indptr and indices: int32 while the non-zeros fit, int64
// once they do not. The caller picks IO from the exact count of step 2, so the
// combinations are (int32, int32), (int32, int64), and (int64, int64).
//
//   1. For each B entry, flag whether its (row, gene) is new, i.e. absent from
//      A. The row comes from a binary search on b_indptr, and the gene from a
//      binary search in A's row. Flags are packed one bit per entry into
//      32-bit words, with a per-word popcount alongside.
//   2. Exclusive-scan the per-word popcounts. rank(j), the number of new B
//      entries before j, is then one scanned count plus one popcount. This
//      costs 8 bytes per 32 B entries instead of a 4-byte prefix per entry.
//      The output has nA + rank(nB) non-zeros.
//   3. out_indptr[r] = a_indptr[r] + rank(b_indptr[r]).
//   4. Each A entry lands after the A entries before it in its row and the new
//      B entries with a smaller gene; a matching B count is added to it.
//   5. Each new B entry lands after the A entries with a smaller gene and the
//      new B entries before it in its row.
//
// Counts are summed as uint32: both inputs are at most 2^31-1, so a sum never
// wraps, and a sum above 2^31-1 sets an overflow flag instead of being cast.

constexpr int CSR_MERGE_BLOCK = 256;  // a multiple of 32, so warps cover whole flag words

struct NewEntryRank {
  std::uint32_t const* words;  // bit j % 32 of words[j / 32] is set if B entry j is new
  std::int32_t const* counts;  // exclusive scan of popcount(words), plus the total

  __device__ std::int32_t operator()(std::int64_t j) const {
    auto const w   = j >> 5;
    auto const bit = static_cast<unsigned>(j & 31);
    // A whole-word boundary (including j == nB) needs no word lookup.
    if (bit == 0)
      return counts[w];
    return counts[w] + __popc(words[w] & ((1u << bit) - 1u));
  }

  __device__ bool is_new(std::int64_t j) const { return (words[j >> 5] >> (j & 31)) & 1u; }
};

// Largest r with indptr[r] <= j, i.e. the row that holds entry j.
template <typename I>
__device__ inline std::int32_t csr_row_of(I const* indptr, std::int32_t num_rows, std::int64_t j) {
  std::int64_t lo = 0, hi = static_cast<std::int64_t>(num_rows) + 1;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (static_cast<std::int64_t>(indptr[mid]) <= j)
      lo = mid + 1;
    else
      hi = mid;
  }
  return static_cast<std::int32_t>(lo - 1);
}

// First index in [begin, end) whose gene is >= g.
template <typename I>
__device__ inline std::int64_t csr_lower_bound(I const* indices,
                                               std::int64_t begin,
                                               std::int64_t end,
                                               std::int32_t g) {
  while (begin < end) {
    auto const mid = begin + (end - begin) / 2;
    if (static_cast<std::int32_t>(indices[mid]) < g)
      begin = mid + 1;
    else
      end = mid;
  }
  return begin;
}

template <typename IA>
__global__ void csr_merge_flag_new_kernel(IA const* a_indptr,
                                          IA const* a_indices,
                                          std::int32_t const* b_indptr,
                                          std::int32_t const* b_indices,
                                          std::int32_t num_rows,
                                          std::int64_t nB,
                                          std::uint32_t* words,
                                          std::int32_t* counts) {
  auto const j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool is_new  = false;
  if (j < nB) {
    auto const r     = csr_row_of(b_indptr, num_rows, j);
    auto const g     = b_indices[j];
    auto const a_end = static_cast<std::int64_t>(a_indptr[r + 1]);
    auto const pos   = csr_lower_bound(a_indices, a_indptr[r], a_end, g);
    is_new           = !(pos < a_end && static_cast<std::int32_t>(a_indices[pos]) == g);
  }
  // Every lane reaches the ballot; lane 0 of each warp owns word j / 32.
  auto const word = __ballot_sync(0xFFFFFFFFu, is_new);
  if ((threadIdx.x & 31) == 0 && j < nB) {
    words[j >> 5]  = word;
    counts[j >> 5] = __popc(word);
  }
}

template <typename IA, typename IO>
__global__ void csr_merge_indptr_kernel(IA const* a_indptr,
                                        std::int32_t const* b_indptr,
                                        NewEntryRank rank,
                                        std::int32_t num_rows,
                                        IO* out_indptr) {
  auto const r = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r > num_rows)
    return;
  out_indptr[r] = static_cast<IO>(static_cast<std::int64_t>(a_indptr[r]) + rank(b_indptr[r]));
}

template <typename IA, typename IO>
__global__ void csr_merge_scatter_a_kernel(IA const* a_indptr,
                                           IA const* a_indices,
                                           std::int32_t const* a_data,
                                           std::int32_t const* b_indptr,
                                           std::int32_t const* b_indices,
                                           std::int32_t const* b_data,
                                           NewEntryRank rank,
                                           IO const* out_indptr,
                                           std::int32_t num_rows,
                                           std::int64_t nA,
                                           IO* out_indices,
                                           std::int32_t* out_data,
                                           int* overflow) {
  auto const i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nA)
    return;
  auto const r       = csr_row_of(a_indptr, num_rows, i);
  auto const g       = static_cast<std::int32_t>(a_indices[i]);
  auto const b_start = b_indptr[r];
  auto const b_end   = b_indptr[r + 1];
  auto const pos     = csr_lower_bound(b_indices, b_start, b_end, g);
  auto count         = static_cast<std::uint32_t>(a_data[i]);
  if (pos < b_end && b_indices[pos] == g)
    count += static_cast<std::uint32_t>(b_data[pos]);
  if (count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    atomicOr(overflow, 1);
  auto const out =
      static_cast<std::int64_t>(out_indptr[r]) + (i - a_indptr[r]) + (rank(pos) - rank(b_start));
  out_indices[out] = static_cast<IO>(g);
  out_data[out]    = static_cast<std::int32_t>(count);
}

template <typename IA, typename IO>
__global__ void csr_merge_scatter_b_kernel(IA const* a_indptr,
                                           IA const* a_indices,
                                           std::int32_t const* b_indptr,
                                           std::int32_t const* b_indices,
                                           std::int32_t const* b_data,
                                           NewEntryRank rank,
                                           IO const* out_indptr,
                                           std::int32_t num_rows,
                                           std::int64_t nB,
                                           IO* out_indices,
                                           std::int32_t* out_data) {
  auto const j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= nB || !rank.is_new(j))
    return;
  auto const r       = csr_row_of(b_indptr, num_rows, j);
  auto const g       = b_indices[j];
  auto const a_start = static_cast<std::int64_t>(a_indptr[r]);
  auto const pos     = csr_lower_bound(a_indices, a_start, a_indptr[r + 1], g);
  auto const out =
      static_cast<std::int64_t>(out_indptr[r]) + (pos - a_start) + (rank(j) - rank(b_indptr[r]));
  out_indices[out] = static_cast<IO>(g);
  out_data[out]    = b_data[j];
}

// Words and counts for nB entries, packed into one int32 scratch buffer.
inline std::int64_t csr_merge_num_words(std::int64_t nB) {
  return (nB + 31) / 32;
}
inline std::int64_t csr_merge_scratch_size(std::int64_t nB) {
  return 2 * csr_merge_num_words(nB) + 1;
}
inline NewEntryRank csr_merge_rank(std::int32_t const* scratch, std::int64_t nB) {
  return NewEntryRank{reinterpret_cast<std::uint32_t const*>(scratch),
                      scratch + csr_merge_num_words(nB)};
}

inline unsigned csr_merge_grid(std::int64_t n) {
  return static_cast<unsigned>((n + CSR_MERGE_BLOCK - 1) / CSR_MERGE_BLOCK);
}

// Steps 1--2. Returns the exact number of non-zeros in A + B.
template <typename IA>
std::int64_t run_csr_merge_count(IA const* a_indptr,
                                 IA const* a_indices,
                                 std::int64_t nA,
                                 std::int32_t const* b_indptr,
                                 std::int32_t const* b_indices,
                                 std::int64_t nB,
                                 std::int32_t num_rows,
                                 std::int32_t* rank_scratch, /* csr_merge_scratch_size(nB) */
                                 cudaStream_t stream) {
  auto const num_words = csr_merge_num_words(nB);
  auto* words          = reinterpret_cast<std::uint32_t*>(rank_scratch);
  auto* counts         = rank_scratch + num_words;
  // Zero the trailing count, which becomes the total after the scan.
  CS_CHECK(cudaMemsetAsync(counts, 0, sizeof(std::int32_t) * (num_words + 1), stream));
  if (nB > 0) {
    csr_merge_flag_new_kernel<<<csr_merge_grid(nB), CSR_MERGE_BLOCK, 0, stream>>>(
        a_indptr, a_indices, b_indptr, b_indices, num_rows, nB, words, counts);
    CS_CHECK(cudaGetLastError());

    DeviceScratch scratch{stream};
    std::size_t temp_bytes = 0;
    CS_CHECK(
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, counts, num_words + 1, stream));
    scratch.resize(temp_bytes);
    CS_CHECK(cub::DeviceScan::ExclusiveSum(
        scratch.ptr, temp_bytes, counts, counts, num_words + 1, stream));
  }

  std::int32_t num_new = 0;
  CS_CHECK(cudaMemcpyAsync(
      &num_new, counts + num_words, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
  CS_CHECK(cudaStreamSynchronize(stream));
  return nA + num_new;
}

// Steps 3--5. Throws std::overflow_error if any summed count exceeds 2^31-1.
template <typename IA, typename IO>
void run_csr_merge_fill(IA const* a_indptr,
                        IA const* a_indices,
                        std::int32_t const* a_data,
                        std::int64_t nA,
                        std::int32_t const* b_indptr,
                        std::int32_t const* b_indices,
                        std::int32_t const* b_data,
                        std::int64_t nB,
                        std::int32_t num_rows,
                        std::int32_t const* rank_scratch,
                        IO* out_indptr,
                        IO* out_indices,
                        std::int32_t* out_data,
                        cudaStream_t stream) {
  constexpr int block = CSR_MERGE_BLOCK;
  auto const rank     = csr_merge_rank(rank_scratch, nB);
  csr_merge_indptr_kernel<<<csr_merge_grid(static_cast<std::int64_t>(num_rows) + 1),
                            block,
                            0,
                            stream>>>(a_indptr, b_indptr, rank, num_rows, out_indptr);
  CS_CHECK(cudaGetLastError());

  DeviceBuffer<int> d_overflow{1, stream};
  CS_CHECK(cudaMemsetAsync(d_overflow.ptr, 0, sizeof(int), stream));
  if (nA > 0) {
    csr_merge_scatter_a_kernel<<<csr_merge_grid(nA), block, 0, stream>>>(a_indptr,
                                                                         a_indices,
                                                                         a_data,
                                                                         b_indptr,
                                                                         b_indices,
                                                                         b_data,
                                                                         rank,
                                                                         out_indptr,
                                                                         num_rows,
                                                                         nA,
                                                                         out_indices,
                                                                         out_data,
                                                                         d_overflow.ptr);
    CS_CHECK(cudaGetLastError());
  }
  if (nB > 0) {
    csr_merge_scatter_b_kernel<<<csr_merge_grid(nB), block, 0, stream>>>(a_indptr,
                                                                         a_indices,
                                                                         b_indptr,
                                                                         b_indices,
                                                                         b_data,
                                                                         rank,
                                                                         out_indptr,
                                                                         num_rows,
                                                                         nB,
                                                                         out_indices,
                                                                         out_data);
    CS_CHECK(cudaGetLastError());
  }

  int overflow_host = 0;
  CS_CHECK(
      cudaMemcpyAsync(&overflow_host, d_overflow.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CS_CHECK(cudaStreamSynchronize(stream));
  if (overflow_host) {
    throw std::overflow_error(
        "aggregate_to_cells: a (cell, gene) count exceeds the int32 limit of 2^31-1");
  }
}

}  // namespace cuspa
