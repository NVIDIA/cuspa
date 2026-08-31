/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include "../common/ndarray_utils.cuh"
#include "../ops.hpp"
#include "kernels_query.cuh"

namespace nb = nanobind;

namespace cuspa {

// Shared precondition check used by assign_points / overlap_pairs entries.
template <typename Device>
static void _validate_query_inputs(DeviceArray<Device> const& points_xy,
                                   double origin_x,
                                   double origin_y,
                                   double cell_size_x,
                                   double cell_size_y,
                                   std::int32_t nx,
                                   std::int32_t ny,
                                   DeviceArray<Device> const& grid_offsets,
                                   DeviceArray<Device> const& poly_ids,
                                   DeviceArray<Device> const& poly_aabbs,
                                   DeviceArray<Device> const& part_offsets,
                                   DeviceArray<Device> const& ring_offsets,
                                   DeviceArray<Device> const& poly_points_xy) {
  _require_2xy(points_xy, "points_xy");
  _require_2xy(poly_points_xy, "poly_points_xy");
  _require_1d(grid_offsets, "grid_offsets");
  _require_1d(poly_ids, "poly_ids");
  _require_1d(part_offsets, "part_offsets");
  _require_1d(ring_offsets, "ring_offsets");
  _require_int32(grid_offsets, "grid_offsets");
  _require_int32(poly_ids, "poly_ids");
  _require_int32(part_offsets, "part_offsets");
  _require_int32(ring_offsets, "ring_offsets");
  if (poly_aabbs.ndim() != 2 || poly_aabbs.shape(1) != 4)
    throw std::invalid_argument("poly_aabbs must have shape (P, 4)");
  _require_grid_params(origin_x, origin_y, cell_size_x, cell_size_y, nx, ny);
  auto const ncells = _checked_grid_size(nx, ny);
  if (static_cast<std::int64_t>(grid_offsets.shape(0)) != ncells + 1)
    throw std::invalid_argument("grid_offsets size must be nx*ny + 1");
  if (part_offsets.shape(0) < 2 || ring_offsets.shape(0) < 2)
    throw std::invalid_argument("offset arrays must each contain at least two entries");
  auto const P = part_offsets.shape(0) - 1;
  if (poly_aabbs.shape(0) != P)
    throw std::invalid_argument("poly_aabbs row count must equal the polygon count");
  if (points_xy.dtype() != poly_points_xy.dtype() || points_xy.dtype() != poly_aabbs.dtype())
    throw std::invalid_argument("points, poly_points, and poly_aabbs must share a dtype");
}

template <typename Device>
void assign_points_entry(DeviceArray<Device> points_xy,
                         double origin_x,
                         double origin_y,
                         double cell_size_x,
                         double cell_size_y,
                         std::int32_t nx,
                         std::int32_t ny,
                         DeviceArray<Device> grid_offsets,
                         DeviceArray<Device> poly_ids,
                         DeviceArray<Device> poly_aabbs,
                         DeviceArray<Device> part_offsets,
                         DeviceArray<Device> ring_offsets,
                         DeviceArray<Device> poly_points_xy,
                         DeviceArray<Device> out_cell_id,
                         int edge_inclusive,
                         std::uintptr_t stream_handle) {
  _validate_query_inputs(points_xy,
                         origin_x,
                         origin_y,
                         cell_size_x,
                         cell_size_y,
                         nx,
                         ny,
                         grid_offsets,
                         poly_ids,
                         poly_aabbs,
                         part_offsets,
                         ring_offsets,
                         poly_points_xy);
  _require_1d(out_cell_id, "out_cell_id");
  _require_int32(out_cell_id, "out_cell_id");

  auto const N = static_cast<std::int64_t>(points_xy.shape(0));
  if (static_cast<std::int64_t>(out_cell_id.shape(0)) != N)
    throw std::invalid_argument("out_cell_id size must equal number of points");

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const offs_p = static_cast<std::int32_t const*>(grid_offsets.data());
  auto const poly_p = static_cast<std::int32_t const*>(poly_ids.data());
  auto const part_p = static_cast<std::int32_t const*>(part_offsets.data());
  auto const ring_p = static_cast<std::int32_t const*>(ring_offsets.data());
  auto const out_p  = static_cast<std::int32_t*>(out_cell_id.data());

  if (points_xy.dtype() == nb::dtype<float>()) {
    run_assign_points<float>(static_cast<float const*>(points_xy.data()),
                             N,
                             static_cast<float>(origin_x),
                             static_cast<float>(origin_y),
                             static_cast<float>(1.0 / cell_size_x),
                             static_cast<float>(1.0 / cell_size_y),
                             nx,
                             ny,
                             offs_p,
                             poly_p,
                             static_cast<float const*>(poly_aabbs.data()),
                             part_p,
                             ring_p,
                             static_cast<float const*>(poly_points_xy.data()),
                             edge_inclusive,
                             out_p,
                             stream);
  } else if (points_xy.dtype() == nb::dtype<double>()) {
    run_assign_points<double>(static_cast<double const*>(points_xy.data()),
                              N,
                              origin_x,
                              origin_y,
                              1.0 / cell_size_x,
                              1.0 / cell_size_y,
                              nx,
                              ny,
                              offs_p,
                              poly_p,
                              static_cast<double const*>(poly_aabbs.data()),
                              part_p,
                              ring_p,
                              static_cast<double const*>(poly_points_xy.data()),
                              edge_inclusive,
                              out_p,
                              stream);
  } else {
    throw std::invalid_argument("points dtype must be float32 or float64");
  }
}

// Pass 1 of overlap_pairs: count hits, exclusive-scan, return K.
template <typename Device>
std::int64_t overlap_count_and_scan_entry(DeviceArray<Device> points_xy,
                                          double origin_x,
                                          double origin_y,
                                          double cell_size_x,
                                          double cell_size_y,
                                          std::int32_t nx,
                                          std::int32_t ny,
                                          DeviceArray<Device> grid_offsets,
                                          DeviceArray<Device> poly_ids,
                                          DeviceArray<Device> poly_aabbs,
                                          DeviceArray<Device> part_offsets,
                                          DeviceArray<Device> ring_offsets,
                                          DeviceArray<Device> poly_points_xy,
                                          DeviceArray<Device> out_offsets,
                                          int edge_inclusive,
                                          std::uintptr_t stream_handle) {
  _validate_query_inputs(points_xy,
                         origin_x,
                         origin_y,
                         cell_size_x,
                         cell_size_y,
                         nx,
                         ny,
                         grid_offsets,
                         poly_ids,
                         poly_aabbs,
                         part_offsets,
                         ring_offsets,
                         poly_points_xy);
  _require_1d(out_offsets, "out_offsets");
  _require_int32(out_offsets, "out_offsets");

  auto const N = static_cast<std::int64_t>(points_xy.shape(0));
  if (N > std::numeric_limits<std::int32_t>::max())
    throw std::overflow_error("overlap_pairs supports at most 2^31-1 input points");
  if (static_cast<std::int64_t>(out_offsets.shape(0)) != N + 1)
    throw std::invalid_argument("out_offsets size must be num_points + 1");

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const offs_p = static_cast<std::int32_t const*>(grid_offsets.data());
  auto const poly_p = static_cast<std::int32_t const*>(poly_ids.data());
  auto const part_p = static_cast<std::int32_t const*>(part_offsets.data());
  auto const ring_p = static_cast<std::int32_t const*>(ring_offsets.data());
  auto const oo_p   = static_cast<std::int32_t*>(out_offsets.data());

  if (points_xy.dtype() == nb::dtype<float>()) {
    return run_overlap_count_and_scan<float>(static_cast<float const*>(points_xy.data()),
                                             N,
                                             static_cast<float>(origin_x),
                                             static_cast<float>(origin_y),
                                             static_cast<float>(1.0 / cell_size_x),
                                             static_cast<float>(1.0 / cell_size_y),
                                             nx,
                                             ny,
                                             offs_p,
                                             poly_p,
                                             static_cast<float const*>(poly_aabbs.data()),
                                             part_p,
                                             ring_p,
                                             static_cast<float const*>(poly_points_xy.data()),
                                             edge_inclusive,
                                             oo_p,
                                             stream);
  }
  if (points_xy.dtype() == nb::dtype<double>()) {
    return run_overlap_count_and_scan<double>(static_cast<double const*>(points_xy.data()),
                                              N,
                                              origin_x,
                                              origin_y,
                                              1.0 / cell_size_x,
                                              1.0 / cell_size_y,
                                              nx,
                                              ny,
                                              offs_p,
                                              poly_p,
                                              static_cast<double const*>(poly_aabbs.data()),
                                              part_p,
                                              ring_p,
                                              static_cast<double const*>(poly_points_xy.data()),
                                              edge_inclusive,
                                              oo_p,
                                              stream);
  }
  throw std::invalid_argument("points dtype must be float32 or float64");
}

// Pass 2 of overlap_pairs: fill pairs[K, 2].
template <typename Device>
void overlap_emit_entry(DeviceArray<Device> points_xy,
                        double origin_x,
                        double origin_y,
                        double cell_size_x,
                        double cell_size_y,
                        std::int32_t nx,
                        std::int32_t ny,
                        DeviceArray<Device> grid_offsets,
                        DeviceArray<Device> poly_ids,
                        DeviceArray<Device> poly_aabbs,
                        DeviceArray<Device> part_offsets,
                        DeviceArray<Device> ring_offsets,
                        DeviceArray<Device> poly_points_xy,
                        DeviceArray<Device> offsets,
                        DeviceArray<Device> out_pairs,
                        int edge_inclusive,
                        std::uintptr_t stream_handle) {
  _validate_query_inputs(points_xy,
                         origin_x,
                         origin_y,
                         cell_size_x,
                         cell_size_y,
                         nx,
                         ny,
                         grid_offsets,
                         poly_ids,
                         poly_aabbs,
                         part_offsets,
                         ring_offsets,
                         poly_points_xy);
  _require_1d(offsets, "offsets");
  _require_int32(offsets, "offsets");
  if (out_pairs.ndim() != 2 || out_pairs.shape(1) != 2)
    throw std::invalid_argument("out_pairs must have shape (K, 2)");
  if (out_pairs.dtype() != nb::dtype<std::int32_t>())
    throw std::invalid_argument("out_pairs must be int32");

  auto const N = static_cast<std::int64_t>(points_xy.shape(0));
  if (N > std::numeric_limits<std::int32_t>::max())
    throw std::overflow_error("overlap_pairs supports at most 2^31-1 input points");
  if (static_cast<std::int64_t>(offsets.shape(0)) != N + 1)
    throw std::invalid_argument("offsets size must be num_points + 1");

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const offs_p = static_cast<std::int32_t const*>(grid_offsets.data());
  auto const poly_p = static_cast<std::int32_t const*>(poly_ids.data());
  auto const part_p = static_cast<std::int32_t const*>(part_offsets.data());
  auto const ring_p = static_cast<std::int32_t const*>(ring_offsets.data());
  auto const off_p  = static_cast<std::int32_t const*>(offsets.data());
  auto const out_p  = static_cast<std::int32_t*>(out_pairs.data());

  if (points_xy.dtype() == nb::dtype<float>()) {
    run_overlap_emit<float>(static_cast<float const*>(points_xy.data()),
                            N,
                            static_cast<float>(origin_x),
                            static_cast<float>(origin_y),
                            static_cast<float>(1.0 / cell_size_x),
                            static_cast<float>(1.0 / cell_size_y),
                            nx,
                            ny,
                            offs_p,
                            poly_p,
                            static_cast<float const*>(poly_aabbs.data()),
                            part_p,
                            ring_p,
                            static_cast<float const*>(poly_points_xy.data()),
                            edge_inclusive,
                            off_p,
                            out_p,
                            stream);
  } else if (points_xy.dtype() == nb::dtype<double>()) {
    run_overlap_emit<double>(static_cast<double const*>(points_xy.data()),
                             N,
                             origin_x,
                             origin_y,
                             1.0 / cell_size_x,
                             1.0 / cell_size_y,
                             nx,
                             ny,
                             offs_p,
                             poly_p,
                             static_cast<double const*>(poly_aabbs.data()),
                             part_p,
                             ring_p,
                             static_cast<double const*>(poly_points_xy.data()),
                             edge_inclusive,
                             off_p,
                             out_p,
                             stream);
  } else {
    throw std::invalid_argument("points dtype must be float32 or float64");
  }
}

template <typename Device>
void bind_query_device(nb::module_& m) {
  m.def("assign_points",
        &cuspa::assign_points_entry<Device>,
        nb::arg("points_xy"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("cell_size_x"),
        nb::arg("cell_size_y"),
        nb::arg("nx"),
        nb::arg("ny"),
        nb::arg("grid_offsets"),
        nb::arg("poly_ids"),
        nb::arg("poly_aabbs"),
        nb::arg("part_offsets"),
        nb::arg("ring_offsets"),
        nb::arg("poly_points_xy"),
        nb::arg("out_cell_id"),
        nb::arg("edge_inclusive") = 0,
        nb::arg("stream")         = std::uintptr_t{0});

  m.def("overlap_count_and_scan",
        &cuspa::overlap_count_and_scan_entry<Device>,
        nb::arg("points_xy"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("cell_size_x"),
        nb::arg("cell_size_y"),
        nb::arg("nx"),
        nb::arg("ny"),
        nb::arg("grid_offsets"),
        nb::arg("poly_ids"),
        nb::arg("poly_aabbs"),
        nb::arg("part_offsets"),
        nb::arg("ring_offsets"),
        nb::arg("poly_points_xy"),
        nb::arg("out_offsets"),
        nb::arg("edge_inclusive") = 0,
        nb::arg("stream")         = std::uintptr_t{0},
        "Pass 1 of many-to-many overlap_pairs: count hits per point + "
        "exclusive scan into out_offsets. Returns K = total number of "
        "(point, polygon) hits.");

  m.def("overlap_emit",
        &cuspa::overlap_emit_entry<Device>,
        nb::arg("points_xy"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("cell_size_x"),
        nb::arg("cell_size_y"),
        nb::arg("nx"),
        nb::arg("ny"),
        nb::arg("grid_offsets"),
        nb::arg("poly_ids"),
        nb::arg("poly_aabbs"),
        nb::arg("part_offsets"),
        nb::arg("ring_offsets"),
        nb::arg("poly_points_xy"),
        nb::arg("offsets"),
        nb::arg("out_pairs"),
        nb::arg("edge_inclusive") = 0,
        nb::arg("stream")         = std::uintptr_t{0},
        "Pass 2 of many-to-many overlap_pairs: fill (K, 2) int32 pairs using "
        "offsets from overlap_count_and_scan.");
}

void bind_query_ops(nb::module_& m) {
  bind_query_device<nb::device::cuda>(m);
  bind_query_device<nb::device::cuda_managed>(m);
}

}  // namespace cuspa
