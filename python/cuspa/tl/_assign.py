# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from .. import _core
from .._types import (
    Polygons,
    _array_device,
    _array_mod,
    _dtype_name,
    _require_array,
    _require_stream,
)

Predicate = Literal["contains", "intersects"]


def _predicate_flag(predicate: Predicate) -> int:
    if predicate == "contains":
        return 0
    if predicate == "intersects":
        return 1
    raise ValueError(f"predicate must be 'contains' or 'intersects', got {predicate!r}")


def _as_2d_xy(points: Any) -> Any:
    shape = tuple(points.shape)
    if len(shape) == 2 and shape[1] == 2:
        return points
    if len(shape) == 1 and shape[0] % 2 == 0:
        return points.reshape(shape[0] // 2, 2)
    raise ValueError(f"expected (N, 2) or (2N,) xy tensor, got shape {shape}")


def _empty_int32_like(ref: Any, shape) -> Any:
    mod = _array_mod(ref)
    if mod == "cupy":
        import cupy as xp  # type: ignore[import-not-found]

        return xp.empty(shape, dtype=xp.int32)
    if mod == "torch":
        import torch  # type: ignore[import-not-found]

        return torch.empty(shape, dtype=torch.int32, device=ref.device)
    raise TypeError(f"Cannot auto-allocate for array of type {type(ref)!r}")


def _validate_query_arrays(
    points_xy: Any, polygons: Polygons, out: Any | None = None
) -> None:
    module = _require_array(
        points_xy,
        "points",
        ndim=2,
        dtype=("float32", "float64"),
    )
    if module != _array_mod(polygons.points_xy):
        raise TypeError("points and polygon buffers must use the same array provider")
    device = _array_device(points_xy)
    if device != _array_device(polygons.points_xy):
        raise TypeError("points and polygon buffers must use the same CUDA device")
    if _dtype_name(points_xy) != _dtype_name(polygons.points_xy):
        raise TypeError("points and polygon coordinates must have the same dtype")
    if out is not None:
        _require_array(out, "out", ndim=1, dtype="int32", module=module, device=device)


def assign_points(
    points: Any,
    polygons: Polygons,
    *,
    predicate: Predicate = "contains",
    out: Any | None = None,
    stream: int = 0,
) -> Any:
    """Assign each point to one polygon.

    Parameters
    ----------
    points
        Device tensor of shape ``(N, 2)`` or flat interleaved ``(2N,)``.
    polygons
        :class:`cuspa.Polygons`. Its ``aabbs`` and ``index`` fields are
        populated on first call and reused on subsequent calls.
    predicate
        ``"contains"`` (default) — strict interior; on-edge points excluded
        (matches GEOS ``contains``). ``"intersects"`` — edge-inclusive.
    out
        Optional pre-allocated ``int32`` device tensor of length ``N``.
    stream
        CUDA stream handle. ``0`` means the default stream.

    Returns
    -------
    int32 tensor of length ``N``. ``out[i]`` is the polygon index containing
    point ``i``, or ``-1`` if no polygon contains it.

    Notes
    -----
    When polygons overlap, the first containing polygon found (in index order
    within each grid cell) wins. Use :func:`overlap_pairs` if you need all
    containing polygons per point.
    """
    points_xy = _as_2d_xy(points)
    poly_xy = _as_2d_xy(polygons.points_xy)
    _require_stream(stream)
    _validate_query_arrays(points_xy, polygons, out)

    n = int(points_xy.shape[0])
    if out is None:
        out = _empty_int32_like(points_xy, n)

    idx = polygons.ensure_index(stream=stream)
    edge_flag = _predicate_flag(predicate)

    _core.assign_points(
        points_xy,
        idx.origin_x,
        idx.origin_y,
        idx.cell_size_x,
        idx.cell_size_y,
        idx.nx,
        idx.ny,
        idx.grid_offsets,
        idx.poly_ids,
        idx.aabbs,
        polygons.part_offsets,
        polygons.ring_offsets,
        poly_xy,
        out,
        edge_flag,
        stream,
    )
    return out


def overlap_pairs(
    points: Any,
    polygons: Polygons,
    *,
    predicate: Predicate = "contains",
    stream: int = 0,
) -> Any:
    """Return all matching ``(point_idx, polygon_idx)`` pairs.

    A point can appear more than once when polygons overlap.

    Parameters
    ----------
    points
        Device tensor of shape ``(N, 2)`` or flat interleaved ``(2N,)``.
    polygons
        :class:`cuspa.Polygons`. The shared uniform-grid index is built on
        first call and reused.
    predicate
        ``"contains"`` (default) or ``"intersects"`` (edge-inclusive).
    stream
        CUDA stream handle.

    Returns
    -------
    int32 tensor of shape ``(K, 2)`` where ``K`` is the total number of
    containments. Column 0 is point index, column 1 is polygon index. Rows
    are ordered by ``point_idx``, then by the polygon's position in the grid
    cell list.
    """
    points_xy = _as_2d_xy(points)
    poly_xy = _as_2d_xy(polygons.points_xy)
    _require_stream(stream)
    _validate_query_arrays(points_xy, polygons)
    n = int(points_xy.shape[0])
    if n > 2**31 - 1:
        raise OverflowError("overlap_pairs supports at most 2^31-1 input points")

    idx = polygons.ensure_index(stream=stream)
    edge_flag = _predicate_flag(predicate)

    offsets = _empty_int32_like(points_xy, n + 1)
    K = int(
        _core.overlap_count_and_scan(
            points_xy,
            idx.origin_x,
            idx.origin_y,
            idx.cell_size_x,
            idx.cell_size_y,
            idx.nx,
            idx.ny,
            idx.grid_offsets,
            idx.poly_ids,
            idx.aabbs,
            polygons.part_offsets,
            polygons.ring_offsets,
            poly_xy,
            offsets,
            edge_flag,
            stream,
        )
    )

    pairs = _empty_int32_like(points_xy, (K, 2))
    if K > 0:
        _core.overlap_emit(
            points_xy,
            idx.origin_x,
            idx.origin_y,
            idx.cell_size_x,
            idx.cell_size_y,
            idx.nx,
            idx.ny,
            idx.grid_offsets,
            idx.poly_ids,
            idx.aabbs,
            polygons.part_offsets,
            polygons.ring_offsets,
            poly_xy,
            offsets,
            pairs,
            edge_flag,
            stream,
        )
    return pairs
