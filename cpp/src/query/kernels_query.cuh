/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>

#include "../common/cuda_utils.cuh"
#include "../common/geometry.cuh"

namespace cuspa {

constexpr int QUERY_BLOCK = 256;

// --- Query kernel: point → cell_id assignment -------------------------------

template <typename T>
__global__ void assign_points_kernel(T const* points_xy,
                                     std::int64_t num_points,
                                     T origin_x,
                                     T origin_y,
                                     T inv_cell_x,
                                     T inv_cell_y,
                                     std::int32_t nx,
                                     std::int32_t ny,
                                     std::int32_t const* grid_offsets,
                                     std::int32_t const* poly_ids,
                                     T const* poly_aabbs,
                                     std::int32_t const* part_offsets,
                                     std::int32_t const* ring_offsets,
                                     T const* poly_xy,
                                     int edge_inclusive,
                                     std::int32_t* out_cell_id) {
  auto const tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_points)
    return;

  T const px = points_xy[2 * tid + 0];
  T const py = points_xy[2 * tid + 1];

  T const grid_x = (px - origin_x) * inv_cell_x;
  T const grid_y = (py - origin_y) * inv_cell_y;
  if (!(grid_x >= T{0} && grid_x < static_cast<T>(nx) && grid_y >= T{0} &&
        grid_y < static_cast<T>(ny))) {
    out_cell_id[tid] = -1;
    return;
  }
  int const gx = static_cast<int>(grid_x);
  int const gy = static_cast<int>(grid_y);

  std::int32_t const cell_idx = gy * nx + gx;
  std::int32_t const beg      = grid_offsets[cell_idx];
  std::int32_t const end      = grid_offsets[cell_idx + 1];

  for (std::int32_t i = beg; i < end; ++i) {
    std::int32_t const p = poly_ids[i];

    T const minx = poly_aabbs[4 * p + 0];
    T const miny = poly_aabbs[4 * p + 1];
    T const maxx = poly_aabbs[4 * p + 2];
    T const maxy = poly_aabbs[4 * p + 3];
    if (px < minx || px > maxx || py < miny || py > maxy)
      continue;

    std::int32_t const rb = part_offsets[p];
    std::int32_t const re = part_offsets[p + 1];
    if (is_point_in_polygon<T>(px, py, ring_offsets, rb, re - rb, poly_xy, edge_inclusive)) {
      out_cell_id[tid] = p;
      return;
    }
  }
  out_cell_id[tid] = -1;
}

// --- Overlap-pairs kernels: many-to-many (transcript, polygon) hits ---------
//
// Two passes using the same grid index:
//   pass 1: count hits per point  → counts[N]
//                                 → exclusive scan → offsets[N+1], K = offsets[N]
//   pass 2: emit (point, polygon) pairs at offsets[tid]+slot into pairs[K,2]

template <typename T>
__global__ void overlap_count_kernel(T const* points_xy,
                                     std::int64_t num_points,
                                     T origin_x,
                                     T origin_y,
                                     T inv_cell_x,
                                     T inv_cell_y,
                                     std::int32_t nx,
                                     std::int32_t ny,
                                     std::int32_t const* grid_offsets,
                                     std::int32_t const* poly_ids,
                                     T const* poly_aabbs,
                                     std::int32_t const* part_offsets,
                                     std::int32_t const* ring_offsets,
                                     T const* poly_xy,
                                     int edge_inclusive,
                                     std::int32_t* out_counts) {
  auto const tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_points)
    return;

  T const px = points_xy[2 * tid + 0];
  T const py = points_xy[2 * tid + 1];

  T const grid_x = (px - origin_x) * inv_cell_x;
  T const grid_y = (py - origin_y) * inv_cell_y;
  if (!(grid_x >= T{0} && grid_x < static_cast<T>(nx) && grid_y >= T{0} &&
        grid_y < static_cast<T>(ny))) {
    out_counts[tid] = 0;
    return;
  }
  int const gx = static_cast<int>(grid_x);
  int const gy = static_cast<int>(grid_y);

  std::int32_t const cell_idx = gy * nx + gx;
  std::int32_t const beg      = grid_offsets[cell_idx];
  std::int32_t const end      = grid_offsets[cell_idx + 1];

  std::int32_t n_hits = 0;
  for (std::int32_t i = beg; i < end; ++i) {
    std::int32_t const p = poly_ids[i];
    T const minx         = poly_aabbs[4 * p + 0];
    T const miny         = poly_aabbs[4 * p + 1];
    T const maxx         = poly_aabbs[4 * p + 2];
    T const maxy         = poly_aabbs[4 * p + 3];
    if (px < minx || px > maxx || py < miny || py > maxy)
      continue;

    std::int32_t const rb = part_offsets[p];
    std::int32_t const re = part_offsets[p + 1];
    if (is_point_in_polygon<T>(px, py, ring_offsets, rb, re - rb, poly_xy, edge_inclusive)) {
      ++n_hits;
    }
  }
  out_counts[tid] = n_hits;
}

template <typename T>
__global__ void overlap_emit_kernel(T const* points_xy,
                                    std::int64_t num_points,
                                    T origin_x,
                                    T origin_y,
                                    T inv_cell_x,
                                    T inv_cell_y,
                                    std::int32_t nx,
                                    std::int32_t ny,
                                    std::int32_t const* grid_offsets,
                                    std::int32_t const* poly_ids,
                                    T const* poly_aabbs,
                                    std::int32_t const* part_offsets,
                                    std::int32_t const* ring_offsets,
                                    T const* poly_xy,
                                    int edge_inclusive,
                                    std::int32_t const* offsets /* [N+1] */,
                                    std::int32_t* pairs /* [K, 2] flattened */) {
  auto const tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_points)
    return;

  T const px = points_xy[2 * tid + 0];
  T const py = points_xy[2 * tid + 1];

  T const grid_x = (px - origin_x) * inv_cell_x;
  T const grid_y = (py - origin_y) * inv_cell_y;
  if (!(grid_x >= T{0} && grid_x < static_cast<T>(nx) && grid_y >= T{0} &&
        grid_y < static_cast<T>(ny)))
    return;
  int const gx = static_cast<int>(grid_x);
  int const gy = static_cast<int>(grid_y);

  std::int32_t const cell_idx = gy * nx + gx;
  std::int32_t const beg      = grid_offsets[cell_idx];
  std::int32_t const end      = grid_offsets[cell_idx + 1];

  std::int32_t slot = offsets[tid];

  for (std::int32_t i = beg; i < end; ++i) {
    std::int32_t const p = poly_ids[i];
    T const minx         = poly_aabbs[4 * p + 0];
    T const miny         = poly_aabbs[4 * p + 1];
    T const maxx         = poly_aabbs[4 * p + 2];
    T const maxy         = poly_aabbs[4 * p + 3];
    if (px < minx || px > maxx || py < miny || py > maxy)
      continue;

    std::int32_t const rb = part_offsets[p];
    std::int32_t const re = part_offsets[p + 1];
    if (is_point_in_polygon<T>(px, py, ring_offsets, rb, re - rb, poly_xy, edge_inclusive)) {
      pairs[2 * slot + 0] = static_cast<std::int32_t>(tid);
      pairs[2 * slot + 1] = p;
      ++slot;
    }
  }
}

template <typename T>
void run_assign_points(T const* points_xy,
                       std::int64_t num_points,
                       T origin_x,
                       T origin_y,
                       T inv_cell_x,
                       T inv_cell_y,
                       std::int32_t nx,
                       std::int32_t ny,
                       std::int32_t const* grid_offsets,
                       std::int32_t const* poly_ids,
                       T const* poly_aabbs,
                       std::int32_t const* part_offsets,
                       std::int32_t const* ring_offsets,
                       T const* poly_xy,
                       int edge_inclusive,
                       std::int32_t* out_cell_id,
                       cudaStream_t stream) {
  if (num_points == 0)
    return;
  auto const grid = static_cast<unsigned>((num_points + QUERY_BLOCK - 1) / QUERY_BLOCK);
  assign_points_kernel<T><<<grid, QUERY_BLOCK, 0, stream>>>(points_xy,
                                                            num_points,
                                                            origin_x,
                                                            origin_y,
                                                            inv_cell_x,
                                                            inv_cell_y,
                                                            nx,
                                                            ny,
                                                            grid_offsets,
                                                            poly_ids,
                                                            poly_aabbs,
                                                            part_offsets,
                                                            ring_offsets,
                                                            poly_xy,
                                                            edge_inclusive,
                                                            out_cell_id);
  CS_CHECK(cudaGetLastError());
}

// Counts-and-scans: writes counts[N] via the pass-1 kernel, then runs an
// exclusive-sum over (N+1) elements yielding offsets[N+1] (offsets[N] = K).
// Returns K to the host.
template <typename T>
std::int64_t run_overlap_count_and_scan(T const* points_xy,
                                        std::int64_t num_points,
                                        T origin_x,
                                        T origin_y,
                                        T inv_cell_x,
                                        T inv_cell_y,
                                        std::int32_t nx,
                                        std::int32_t ny,
                                        std::int32_t const* grid_offsets,
                                        std::int32_t const* poly_ids,
                                        T const* poly_aabbs,
                                        std::int32_t const* part_offsets,
                                        std::int32_t const* ring_offsets,
                                        T const* poly_xy,
                                        int edge_inclusive,
                                        std::int32_t* out_offsets /* [N+1] */,
                                        cudaStream_t stream) {
  if (num_points == 0) {
    CS_CHECK(cudaMemsetAsync(out_offsets, 0, sizeof(std::int32_t), stream));
    return 0;
  }
  DeviceScratch scratch{stream};

  DeviceBuffer<std::int32_t> d_counts{static_cast<std::size_t>(num_points), stream};

  auto const grid = static_cast<unsigned>((num_points + QUERY_BLOCK - 1) / QUERY_BLOCK);
  overlap_count_kernel<T><<<grid, QUERY_BLOCK, 0, stream>>>(points_xy,
                                                            num_points,
                                                            origin_x,
                                                            origin_y,
                                                            inv_cell_x,
                                                            inv_cell_y,
                                                            nx,
                                                            ny,
                                                            grid_offsets,
                                                            poly_ids,
                                                            poly_aabbs,
                                                            part_offsets,
                                                            ring_offsets,
                                                            poly_xy,
                                                            edge_inclusive,
                                                            d_counts.ptr);
  CS_CHECK(cudaGetLastError());

  // Accumulate K in int64 and reject values that do not fit int32 outputs.
  DeviceBuffer<std::int64_t> d_total{1, stream};
  std::size_t reduce_bytes = 0;
  CS_CHECK(
      cub::DeviceReduce::Sum(nullptr, reduce_bytes, d_counts.ptr, d_total.ptr, num_points, stream));
  scratch.resize(reduce_bytes);
  CS_CHECK(cub::DeviceReduce::Sum(
      scratch.ptr, reduce_bytes, d_counts.ptr, d_total.ptr, num_points, stream));

  std::int64_t K_host = 0;
  CS_CHECK(
      cudaMemcpyAsync(&K_host, d_total.ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream));
  CS_CHECK(cudaStreamSynchronize(stream));
  if (K_host > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("overlap_pairs result exceeds the int32 pair limit");
  }

  std::size_t scan_bytes = 0;
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr, scan_bytes, d_counts.ptr, out_offsets, num_points, stream));
  scratch.resize(scan_bytes);
  CS_CHECK(cub::DeviceScan::ExclusiveSum(
      scratch.ptr, scan_bytes, d_counts.ptr, out_offsets, num_points, stream));
  auto const K_i32 = static_cast<std::int32_t>(K_host);
  CS_CHECK(cudaMemcpyAsync(
      out_offsets + num_points, &K_i32, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
  return K_host;
}

template <typename T>
void run_overlap_emit(T const* points_xy,
                      std::int64_t num_points,
                      T origin_x,
                      T origin_y,
                      T inv_cell_x,
                      T inv_cell_y,
                      std::int32_t nx,
                      std::int32_t ny,
                      std::int32_t const* grid_offsets,
                      std::int32_t const* poly_ids,
                      T const* poly_aabbs,
                      std::int32_t const* part_offsets,
                      std::int32_t const* ring_offsets,
                      T const* poly_xy,
                      int edge_inclusive,
                      std::int32_t const* offsets,
                      std::int32_t* pairs_out,
                      cudaStream_t stream) {
  if (num_points == 0)
    return;
  auto const grid = static_cast<unsigned>((num_points + QUERY_BLOCK - 1) / QUERY_BLOCK);
  overlap_emit_kernel<T><<<grid, QUERY_BLOCK, 0, stream>>>(points_xy,
                                                           num_points,
                                                           origin_x,
                                                           origin_y,
                                                           inv_cell_x,
                                                           inv_cell_y,
                                                           nx,
                                                           ny,
                                                           grid_offsets,
                                                           poly_ids,
                                                           poly_aabbs,
                                                           part_offsets,
                                                           ring_offsets,
                                                           poly_xy,
                                                           edge_inclusive,
                                                           offsets,
                                                           pairs_out);
  CS_CHECK(cudaGetLastError());
}

}  // namespace cuspa
