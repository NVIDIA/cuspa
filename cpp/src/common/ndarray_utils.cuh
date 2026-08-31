/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/ndarray.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace cuspa {

// --- nanobind surface -------------------------------------------------------

template <typename Device>
using DeviceArray = nb::ndarray<Device, nb::c_contig>;

template <typename Device>
static void _require_int32(DeviceArray<Device> const& a, char const* name) {
  if (a.dtype() != nb::dtype<std::int32_t>())
    throw std::invalid_argument(std::string{name} + " must be int32");
}

template <typename Device>
static void _require_2xy(DeviceArray<Device> const& a, char const* name) {
  if (a.ndim() != 2 || a.shape(1) != 2)
    throw std::invalid_argument(std::string{name} + " must have shape (N, 2)");
}

template <typename Device>
static void _require_1d(DeviceArray<Device> const& a, char const* name) {
  if (a.ndim() != 1)
    throw std::invalid_argument(std::string{name} + " must be 1-D");
}

static std::int32_t _checked_int32_size(std::size_t size, char const* name) {
  if (size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error(std::string{name} + " exceeds the int32 size limit");
  return static_cast<std::int32_t>(size);
}

static std::int64_t _checked_grid_size(std::int32_t nx, std::int32_t ny) {
  if (nx <= 0 || ny <= 0)
    throw std::invalid_argument("nx and ny must be positive");
  auto const ncells = static_cast<std::int64_t>(nx) * ny;
  if (ncells > std::numeric_limits<std::int32_t>::max())
    throw std::overflow_error("nx * ny exceeds the int32 size limit");
  return ncells;
}

static void _require_grid_params(double origin_x,
                                 double origin_y,
                                 double cell_size_x,
                                 double cell_size_y,
                                 std::int32_t nx,
                                 std::int32_t ny) {
  if (!std::isfinite(origin_x) || !std::isfinite(origin_y))
    throw std::invalid_argument("grid origin must be finite");
  if (!std::isfinite(cell_size_x) || !std::isfinite(cell_size_y) || cell_size_x <= 0 ||
      cell_size_y <= 0)
    throw std::invalid_argument("grid cell sizes must be finite and positive");
  _checked_grid_size(nx, ny);
}

}  // namespace cuspa
