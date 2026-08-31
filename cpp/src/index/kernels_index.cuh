/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <type_traits>

#include "../common/cuda_utils.cuh"

namespace cuspa {

constexpr int REDUCE_BLOCK = 256;
constexpr int FILL_BLOCK   = 128;

// --- Per-polygon AABB reduction --------------------------------------------

template <typename T>
__global__ void poly_aabb_kernel(std::int32_t const* part_offsets,
                                 std::int32_t const* ring_offsets,
                                 T const* xy,
                                 T* out_aabbs /* [P, 4] */) {
  auto const p      = blockIdx.x;
  auto const tid    = threadIdx.x;
  auto const stride = blockDim.x;

  std::int32_t const ring_begin_idx = part_offsets[p];
  std::int32_t const ring_end_idx   = part_offsets[p + 1];
  std::int32_t const pt_begin       = ring_offsets[ring_begin_idx];
  std::int32_t const pt_end         = ring_offsets[ring_end_idx];

  T const pos_inf = std::is_same_v<T, float> ? T(3.4e38) : T(1e308);
  T const neg_inf = -pos_inf;

  T lminx = pos_inf, lminy = pos_inf, lmaxx = neg_inf, lmaxy = neg_inf;
  for (std::int32_t i = pt_begin + tid; i < pt_end; i += stride) {
    T const x = xy[2 * i + 0];
    T const y = xy[2 * i + 1];
    if (x < lminx)
      lminx = x;
    if (y < lminy)
      lminy = y;
    if (x > lmaxx)
      lmaxx = x;
    if (y > lmaxy)
      lmaxy = y;
  }

  __shared__ T s_minx[REDUCE_BLOCK];
  __shared__ T s_miny[REDUCE_BLOCK];
  __shared__ T s_maxx[REDUCE_BLOCK];
  __shared__ T s_maxy[REDUCE_BLOCK];
  s_minx[tid] = lminx;
  s_miny[tid] = lminy;
  s_maxx[tid] = lmaxx;
  s_maxy[tid] = lmaxy;
  __syncthreads();

#pragma unroll
  for (int off = REDUCE_BLOCK / 2; off > 0; off /= 2) {
    if (tid < off) {
      if (s_minx[tid + off] < s_minx[tid])
        s_minx[tid] = s_minx[tid + off];
      if (s_miny[tid + off] < s_miny[tid])
        s_miny[tid] = s_miny[tid + off];
      if (s_maxx[tid + off] > s_maxx[tid])
        s_maxx[tid] = s_maxx[tid + off];
      if (s_maxy[tid + off] > s_maxy[tid])
        s_maxy[tid] = s_maxy[tid + off];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_aabbs[4 * p + 0] = s_minx[0];
    out_aabbs[4 * p + 1] = s_miny[0];
    out_aabbs[4 * p + 2] = s_maxx[0];
    out_aabbs[4 * p + 3] = s_maxy[0];
  }
}

// --- Grid-build kernels -----------------------------------------------------

template <typename T>
__device__ inline int floor_to_int(T value) {
  if constexpr (std::is_same_v<T, float>)
    return __float2int_rd(value);
  else
    return __double2int_rd(value);
}

// Number of grid cells each polygon AABB intersects.
template <typename T>
__global__ void poly_cell_count_kernel(T const* aabbs,
                                       std::int32_t num_polygons,
                                       T origin_x,
                                       T origin_y,
                                       T inv_cell_x,
                                       T inv_cell_y,
                                       std::int32_t nx,
                                       std::int32_t ny,
                                       std::int32_t* counts) {
  auto const p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= num_polygons)
    return;

  T minx = aabbs[4 * p + 0], miny = aabbs[4 * p + 1];
  T maxx = aabbs[4 * p + 2], maxy = aabbs[4 * p + 3];

  int gx0 = floor_to_int((minx - origin_x) * inv_cell_x);
  int gy0 = floor_to_int((miny - origin_y) * inv_cell_y);
  int gx1 = floor_to_int((maxx - origin_x) * inv_cell_x);
  int gy1 = floor_to_int((maxy - origin_y) * inv_cell_y);

  if (gx0 < 0)
    gx0 = 0;
  if (gy0 < 0)
    gy0 = 0;
  if (gx1 >= nx)
    gx1 = nx - 1;
  if (gy1 >= ny)
    gy1 = ny - 1;

  std::int32_t n = 0;
  if (gx1 >= gx0 && gy1 >= gy0) {
    n = (gx1 - gx0 + 1) * (gy1 - gy0 + 1);
  }
  counts[p] = n;
}

// Emit (grid_cell, polygon_idx) pairs. pair_offsets is the exclusive scan of
// counts, so polygon p writes into [pair_offsets[p], pair_offsets[p+1]).
template <typename T>
__global__ void poly_emit_pairs_kernel(T const* aabbs,
                                       std::int32_t num_polygons,
                                       T origin_x,
                                       T origin_y,
                                       T inv_cell_x,
                                       T inv_cell_y,
                                       std::int32_t nx,
                                       std::int32_t ny,
                                       std::int32_t const* pair_offsets,
                                       std::int32_t* pair_cell,
                                       std::int32_t* pair_poly) {
  auto const p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= num_polygons)
    return;

  T minx = aabbs[4 * p + 0], miny = aabbs[4 * p + 1];
  T maxx = aabbs[4 * p + 2], maxy = aabbs[4 * p + 3];

  int gx0 = floor_to_int((minx - origin_x) * inv_cell_x);
  int gy0 = floor_to_int((miny - origin_y) * inv_cell_y);
  int gx1 = floor_to_int((maxx - origin_x) * inv_cell_x);
  int gy1 = floor_to_int((maxy - origin_y) * inv_cell_y);

  if (gx0 < 0)
    gx0 = 0;
  if (gy0 < 0)
    gy0 = 0;
  if (gx1 >= nx)
    gx1 = nx - 1;
  if (gy1 >= ny)
    gy1 = ny - 1;
  if (gx1 < gx0 || gy1 < gy0)
    return;

  std::int32_t base = pair_offsets[p];
  std::int32_t k    = 0;
  for (int gy = gy0; gy <= gy1; ++gy) {
    for (int gx = gx0; gx <= gx1; ++gx) {
      pair_cell[base + k] = gy * nx + gx;
      pair_poly[base + k] = static_cast<std::int32_t>(p);
      ++k;
    }
  }
}

// Given sorted cell keys, produce grid_offsets[ncells+1] via atomic histogram
// + exclusive scan. Runs on sorted_cell[K].
__global__ void grid_hist_kernel(std::int32_t const* sorted_cell,
                                 std::int64_t K,
                                 std::int32_t ncells,
                                 std::int32_t* counts) {
  auto const i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= K)
    return;
  auto const c = sorted_cell[i];
  if (c >= 0 && c < ncells)
    atomicAdd(&counts[c], 1);
}

// --- Host-side build machinery ---------------------------------------------

template <typename T>
void launch_poly_aabbs(std::int32_t const* part_offsets,
                       std::int32_t const* ring_offsets,
                       T const* xy,
                       std::int32_t num_polygons,
                       T* out_aabbs,
                       cudaStream_t stream) {
  if (num_polygons == 0)
    return;
  poly_aabb_kernel<T>
      <<<num_polygons, REDUCE_BLOCK, 0, stream>>>(part_offsets, ring_offsets, xy, out_aabbs);
  CS_CHECK(cudaGetLastError());
}

template <typename T>
std::int64_t run_compute_index_size(T const* aabbs,
                                    std::int32_t num_polygons,
                                    T origin_x,
                                    T origin_y,
                                    T inv_cell_x,
                                    T inv_cell_y,
                                    std::int32_t nx,
                                    std::int32_t ny,
                                    cudaStream_t stream) {
  DeviceScratch scratch{stream};

  // counts[P]
  DeviceBuffer<std::int32_t> d_counts{static_cast<std::size_t>(num_polygons), stream};

  auto const grid = static_cast<unsigned>((num_polygons + FILL_BLOCK - 1) / FILL_BLOCK);
  poly_cell_count_kernel<T><<<grid, FILL_BLOCK, 0, stream>>>(
      aabbs, num_polygons, origin_x, origin_y, inv_cell_x, inv_cell_y, nx, ny, d_counts.ptr);
  CS_CHECK(cudaGetLastError());

  // Sum into int64 so an oversized index is detected before int32 offsets are
  // allocated by the Python layer.
  DeviceBuffer<std::int64_t> d_total{1, stream};
  std::size_t temp_bytes = 0;
  CS_CHECK(
      cub::DeviceReduce::Sum(nullptr, temp_bytes, d_counts.ptr, d_total.ptr, num_polygons, stream));
  scratch.resize(temp_bytes);
  CS_CHECK(cub::DeviceReduce::Sum(
      scratch.ptr, temp_bytes, d_counts.ptr, d_total.ptr, num_polygons, stream));

  std::int64_t total = 0;
  CS_CHECK(
      cudaMemcpyAsync(&total, d_total.ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream));
  CS_CHECK(cudaStreamSynchronize(stream));
  return total;
}

template <typename T>
void run_build_index(T const* aabbs,
                     std::int32_t num_polygons,
                     T origin_x,
                     T origin_y,
                     T inv_cell_x,
                     T inv_cell_y,
                     std::int32_t nx,
                     std::int32_t ny,
                     std::int32_t K,
                     std::int32_t* out_poly_ids,
                     std::int32_t* out_grid_offsets, /* length ncells+1 */
                     cudaStream_t stream) {
  DeviceScratch scratch{stream};
  std::int32_t const ncells = nx * ny;

  // 1. counts + exclusive scan → pair_offsets[P+1]
  DeviceBuffer<std::int32_t> d_counts{static_cast<std::size_t>(num_polygons), stream};
  DeviceBuffer<std::int32_t> d_pair_offs{static_cast<std::size_t>(num_polygons) + 1, stream};

  auto const pg = static_cast<unsigned>((num_polygons + FILL_BLOCK - 1) / FILL_BLOCK);
  poly_cell_count_kernel<T><<<pg, FILL_BLOCK, 0, stream>>>(
      aabbs, num_polygons, origin_x, origin_y, inv_cell_x, inv_cell_y, nx, ny, d_counts.ptr);
  CS_CHECK(cudaGetLastError());

  std::size_t scan_bytes = 0;
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr, scan_bytes, d_counts.ptr, d_pair_offs.ptr, num_polygons, stream));
  scratch.resize(scan_bytes);
  // Scan the P counts, then write the final pair_offs[P] = K explicitly.
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      scratch.ptr, scan_bytes, d_counts.ptr, d_pair_offs.ptr, num_polygons, stream));
  CS_CHECK(cudaMemcpyAsync(
      d_pair_offs.ptr + num_polygons, &K, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));

  // 2. Emit (cell, poly) pairs into unsorted buffers.
  DeviceBuffer<std::int32_t> d_pair_cell{static_cast<std::size_t>(K), stream};
  DeviceBuffer<std::int32_t> d_pair_poly{static_cast<std::size_t>(K), stream};

  poly_emit_pairs_kernel<T><<<pg, FILL_BLOCK, 0, stream>>>(aabbs,
                                                           num_polygons,
                                                           origin_x,
                                                           origin_y,
                                                           inv_cell_x,
                                                           inv_cell_y,
                                                           nx,
                                                           ny,
                                                           d_pair_offs.ptr,
                                                           d_pair_cell.ptr,
                                                           d_pair_poly.ptr);
  CS_CHECK(cudaGetLastError());

  // 3. Radix-sort pairs by cell. Sorted poly values land directly in the
  //    caller's output buffer.
  DeviceBuffer<std::int32_t> d_pair_cell_sorted{static_cast<std::size_t>(K), stream};

  // Compute number of bits needed (ncells-1 high bit) — saves sort time when
  // the grid is small.
  int num_bits = 0;
  for (int v = ncells - 1; v > 0; v >>= 1)
    ++num_bits;
  if (num_bits == 0)
    num_bits = 1;

  std::size_t sort_bytes = 0;
  CS_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                           sort_bytes,
                                           d_pair_cell.ptr,
                                           d_pair_cell_sorted.ptr,
                                           d_pair_poly.ptr,
                                           out_poly_ids,
                                           K,
                                           0,
                                           num_bits,
                                           stream));
  scratch.resize(sort_bytes);
  CS_CHECK(cub::DeviceRadixSort::SortPairs(scratch.ptr,
                                           sort_bytes,
                                           d_pair_cell.ptr,
                                           d_pair_cell_sorted.ptr,
                                           d_pair_poly.ptr,
                                           out_poly_ids,
                                           K,
                                           0,
                                           num_bits,
                                           stream));

  // 4. Histogram sorted cells → out_grid_offsets via exclusive scan.
  CS_CHECK(cudaMemsetAsync(out_grid_offsets, 0, sizeof(std::int32_t) * (ncells + 1), stream));
  auto const hg = static_cast<unsigned>((K + FILL_BLOCK - 1) / FILL_BLOCK);
  if (K > 0) {
    grid_hist_kernel<<<hg, FILL_BLOCK, 0, stream>>>(
        d_pair_cell_sorted.ptr, K, ncells, out_grid_offsets);
    CS_CHECK(cudaGetLastError());
  }
  std::size_t scan2_bytes = 0;
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr, scan2_bytes, out_grid_offsets, out_grid_offsets, ncells + 1, stream));
  scratch.resize(scan2_bytes);
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      scratch.ptr, scan2_bytes, out_grid_offsets, out_grid_offsets, ncells + 1, stream));
}

}  // namespace cuspa
