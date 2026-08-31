/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cuspa {

// --- ULP-based float equality ----------------------------------------------

template <int Size, typename = void>
struct uint_selector;
template <>
struct uint_selector<4> {
  using type = std::uint32_t;
};
template <>
struct uint_selector<8> {
  using type = std::uint64_t;
};

template <typename T>
__host__ __device__ inline bool float_equal(T lhs, T rhs) {
  using Bits                 = typename uint_selector<sizeof(T)>::type;
  constexpr Bits sign_mask   = Bits{1} << (8 * sizeof(Bits) - 1);
  constexpr unsigned max_ulp = 4;

  if (isnan(lhs) || isnan(rhs))
    return false;

  union FB {
    T f;
    Bits b;
  };
  FB l{lhs}, r{rhs};
  auto to_biased = [](Bits v) -> Bits { return (v & sign_mask) ? (~v + 1) : (v | sign_mask); };
  Bits lb        = to_biased(l.b);
  Bits rb        = to_biased(r.b);
  return lb >= rb ? (lb - rb) <= max_ulp : (rb - lb) <= max_ulp;
}

// --- 2D ray-crossing --------------------------------------------------------

// ``edge_inclusive``: 0 = contains (strict, boundary excluded; default —
// matches GEOS `contains`); 1 = intersects (boundary included).
template <typename T>
__device__ inline bool is_point_in_polygon(T px,
                                           T py,
                                           std::int32_t const* ring_offsets,
                                           std::int32_t ring_begin_idx,
                                           std::int32_t polygon_num_rings,
                                           T const* xy,
                                           int edge_inclusive) {
  bool point_is_within = false;

  for (std::int32_t r = 0; r < polygon_num_rings; ++r) {
    std::int32_t const ring_begin = ring_offsets[ring_begin_idx + r];
    std::int32_t const ring_end   = ring_offsets[ring_begin_idx + r + 1];
    if (ring_end - ring_begin < 2)
      continue;

    T bx         = xy[2 * (ring_end - 1) + 0];
    T by         = xy[2 * (ring_end - 1) + 1];
    bool y0_flag = by > py;

    bool point_on_edge = false;
    for (std::int32_t i = ring_begin; i < ring_end; ++i) {
      T const ax = xy[2 * i + 0];
      T const ay = xy[2 * i + 1];

      T const run  = bx - ax;
      T const rise = by - ay;

      if (float_equal<T>(run, T{0}) && float_equal<T>(rise, T{0})) {
        bx = ax;
        by = ay;
        continue;
      }

      T const rise_to_point = py - ay;
      T const run_to_point  = px - ax;

      if (float_equal<T>(run * rise_to_point, run_to_point * rise)) {
        T minx = ax, maxx = bx;
        T miny = ay, maxy = by;
        if (minx > maxx) {
          T tmp = minx;
          minx  = maxx;
          maxx  = tmp;
        }
        if (miny > maxy) {
          T tmp = miny;
          miny  = maxy;
          maxy  = tmp;
        }
        if (minx <= px && px <= maxx && miny <= py && py <= maxy) {
          point_on_edge = true;
          break;
        }
      }

      bool const y1_flag = ay > py;
      if (y1_flag != y0_flag) {
        auto const lhs = (px - ax) * rise;
        auto const rhs = run * rise_to_point;
        if ((lhs < rhs) != y1_flag)
          point_is_within = !point_is_within;
      }

      bx      = ax;
      by      = ay;
      y0_flag = y1_flag;
    }

    if (point_on_edge)
      return edge_inclusive != 0;
  }
  return point_is_within;
}

}  // namespace cuspa
