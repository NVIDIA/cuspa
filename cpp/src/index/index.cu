/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include "../common/ndarray_utils.cuh"
#include "../ops.hpp"
#include "kernels_index.cuh"

namespace nb = nanobind;

namespace cuspa {

template <typename Device>
void compute_poly_aabbs_entry(DeviceArray<Device> part_offsets,
                              DeviceArray<Device> ring_offsets,
                              DeviceArray<Device> poly_points_xy,
                              DeviceArray<Device> out_aabbs,
                              std::uintptr_t stream_handle) {
  _require_1d(part_offsets, "part_offsets");
  _require_1d(ring_offsets, "ring_offsets");
  _require_int32(part_offsets, "part_offsets");
  _require_int32(ring_offsets, "ring_offsets");
  _require_2xy(poly_points_xy, "poly_points_xy");
  if (out_aabbs.ndim() != 2 || out_aabbs.shape(1) != 4)
    throw std::invalid_argument("out_aabbs must have shape (P, 4)");
  if (out_aabbs.dtype() != poly_points_xy.dtype())
    throw std::invalid_argument("out_aabbs must match poly_points_xy dtype");

  if (part_offsets.shape(0) < 2 || ring_offsets.shape(0) < 2)
    throw std::invalid_argument("offset arrays must each contain at least two entries");
  auto const P = _checked_int32_size(part_offsets.shape(0) - 1, "polygon count");
  if (P < 1)
    throw std::invalid_argument("need at least 1 polygon");
  if (static_cast<std::int64_t>(out_aabbs.shape(0)) != P)
    throw std::invalid_argument("out_aabbs shape[0] must equal num_polygons");

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const part_p = static_cast<std::int32_t const*>(part_offsets.data());
  auto const ring_p = static_cast<std::int32_t const*>(ring_offsets.data());

  if (poly_points_xy.dtype() == nb::dtype<float>()) {
    launch_poly_aabbs<float>(part_p,
                             ring_p,
                             static_cast<float const*>(poly_points_xy.data()),
                             P,
                             static_cast<float*>(out_aabbs.data()),
                             stream);
  } else if (poly_points_xy.dtype() == nb::dtype<double>()) {
    launch_poly_aabbs<double>(part_p,
                              ring_p,
                              static_cast<double const*>(poly_points_xy.data()),
                              P,
                              static_cast<double*>(out_aabbs.data()),
                              stream);
  } else {
    throw std::invalid_argument("poly_points_xy dtype must be float32 or float64");
  }
}

template <typename Device>
std::int64_t compute_index_size_entry(DeviceArray<Device> aabbs,
                                      double origin_x,
                                      double origin_y,
                                      double cell_size_x,
                                      double cell_size_y,
                                      std::int32_t nx,
                                      std::int32_t ny,
                                      std::uintptr_t stream_handle) {
  if (aabbs.ndim() != 2 || aabbs.shape(1) != 4)
    throw std::invalid_argument("aabbs must have shape (P, 4)");
  _require_grid_params(origin_x, origin_y, cell_size_x, cell_size_y, nx, ny);

  auto const P = _checked_int32_size(aabbs.shape(0), "polygon count");
  if (P < 1)
    return 0;

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);

  if (aabbs.dtype() == nb::dtype<float>()) {
    return run_compute_index_size<float>(static_cast<float const*>(aabbs.data()),
                                         P,
                                         static_cast<float>(origin_x),
                                         static_cast<float>(origin_y),
                                         static_cast<float>(1.0 / cell_size_x),
                                         static_cast<float>(1.0 / cell_size_y),
                                         nx,
                                         ny,
                                         stream);
  }
  if (aabbs.dtype() == nb::dtype<double>()) {
    return run_compute_index_size<double>(static_cast<double const*>(aabbs.data()),
                                          P,
                                          origin_x,
                                          origin_y,
                                          1.0 / cell_size_x,
                                          1.0 / cell_size_y,
                                          nx,
                                          ny,
                                          stream);
  }
  throw std::invalid_argument("aabbs dtype must be float32 or float64");
}

template <typename Device>
void build_index_entry(DeviceArray<Device> aabbs,
                       double origin_x,
                       double origin_y,
                       double cell_size_x,
                       double cell_size_y,
                       std::int32_t nx,
                       std::int32_t ny,
                       DeviceArray<Device> out_poly_ids,
                       DeviceArray<Device> out_grid_offsets,
                       std::uintptr_t stream_handle) {
  if (aabbs.ndim() != 2 || aabbs.shape(1) != 4)
    throw std::invalid_argument("aabbs must have shape (P, 4)");
  _require_1d(out_poly_ids, "out_poly_ids");
  _require_1d(out_grid_offsets, "out_grid_offsets");
  _require_int32(out_poly_ids, "out_poly_ids");
  _require_int32(out_grid_offsets, "out_grid_offsets");
  _require_grid_params(origin_x, origin_y, cell_size_x, cell_size_y, nx, ny);

  auto const P      = _checked_int32_size(aabbs.shape(0), "polygon count");
  auto const ncells = _checked_grid_size(nx, ny);
  if (static_cast<std::int64_t>(out_grid_offsets.shape(0)) != ncells + 1)
    throw std::invalid_argument("out_grid_offsets size must be nx*ny + 1");
  if (out_poly_ids.shape(0) > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error("out_poly_ids exceeds the int32 size limit");

  auto const stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const poly_p = static_cast<std::int32_t*>(out_poly_ids.data());
  auto const offs_p = static_cast<std::int32_t*>(out_grid_offsets.data());
  auto const K      = static_cast<std::int32_t>(out_poly_ids.shape(0));

  if (aabbs.dtype() == nb::dtype<float>()) {
    run_build_index<float>(static_cast<float const*>(aabbs.data()),
                           P,
                           static_cast<float>(origin_x),
                           static_cast<float>(origin_y),
                           static_cast<float>(1.0 / cell_size_x),
                           static_cast<float>(1.0 / cell_size_y),
                           nx,
                           ny,
                           K,
                           poly_p,
                           offs_p,
                           stream);
  } else if (aabbs.dtype() == nb::dtype<double>()) {
    run_build_index<double>(static_cast<double const*>(aabbs.data()),
                            P,
                            origin_x,
                            origin_y,
                            1.0 / cell_size_x,
                            1.0 / cell_size_y,
                            nx,
                            ny,
                            K,
                            poly_p,
                            offs_p,
                            stream);
  } else {
    throw std::invalid_argument("aabbs dtype must be float32 or float64");
  }
}

template <typename Device>
void bind_index_device(nb::module_& m) {
  m.def("compute_poly_aabbs",
        &cuspa::compute_poly_aabbs_entry<Device>,
        nb::arg("part_offsets"),
        nb::arg("ring_offsets"),
        nb::arg("poly_points_xy"),
        nb::arg("out_aabbs"),
        nb::arg("stream") = std::uintptr_t{0});

  m.def("compute_index_size",
        &cuspa::compute_index_size_entry<Device>,
        nb::arg("aabbs"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("cell_size_x"),
        nb::arg("cell_size_y"),
        nb::arg("nx"),
        nb::arg("ny"),
        nb::arg("stream") = std::uintptr_t{0});

  m.def("build_index",
        &cuspa::build_index_entry<Device>,
        nb::arg("aabbs"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("cell_size_x"),
        nb::arg("cell_size_y"),
        nb::arg("nx"),
        nb::arg("ny"),
        nb::arg("out_poly_ids"),
        nb::arg("out_grid_offsets"),
        nb::arg("stream") = std::uintptr_t{0});
}

void bind_index_ops(nb::module_& m) {
  bind_index_device<nb::device::cuda>(m);
  bind_index_device<nb::device::cuda_managed>(m);
}

}  // namespace cuspa
