# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .. import _core
from .._memory import _collect_host
from .._types import (
    Polygons,
    _array_device,
    _array_mod,
    _dtype_name,
    _require_array,
    _require_stream,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

Predicate = Literal["contains", "intersects"]
_ASSIGN_WORK_BYTES = 32 << 20


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


def _empty_like(ref: Any, shape, dtype: str) -> Any:
    mod = _array_mod(ref)
    if mod == "cupy":
        import cupy as xp  # type: ignore[import-not-found]

        return xp.empty(shape, dtype=getattr(xp, dtype))
    if mod == "torch":
        import torch  # type: ignore[import-not-found]

        return torch.empty(shape, dtype=getattr(torch, dtype), device=ref.device)
    raise TypeError(f"Cannot auto-allocate for array of type {type(ref)!r}")


def _empty_int32_like(ref: Any, shape) -> Any:
    return _empty_like(ref, shape, "int32")


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


def _workspace_bytes(cp):
    free, _ = cp.cuda.runtime.memGetInfo()
    pool = getattr(cp.cuda.get_allocator(), "__self__", None)
    if isinstance(pool, cp.cuda.MemoryPool):
        free += pool.free_bytes()
        if limit := pool.get_limit():
            free = min(free, limit - pool.used_bytes())
    return min(_ASSIGN_WORK_BYTES, free // 2)


def _oom(error):
    return isinstance(error, MemoryError) or str(error) == "CUDA error: out of memory"


def _assignment_batches(
    points: Any,
    polygons: Polygons,
    *,
    predicate: Predicate = "contains",
    stream: int = 0,
) -> Iterator[np.ndarray]:
    """Yield labels for immediate copying to the host result."""
    import cupy as cp

    from ..io._points import host_points

    _predicate_flag(predicate)
    _require_stream(stream)
    coords = polygons.points_xy
    if not isinstance(coords, cp.ndarray):
        raise TypeError("host point assignment requires CuPy polygons")
    device = coords.device
    with device:
        queue = cp.cuda.ExternalStream(stream) if stream else cp.cuda.Stream.null
        with queue:
            polygons.ensure_index(stream=stream)
        row_bytes = 2 * coords.itemsize + np.dtype(np.int32).itemsize
        rows = _workspace_bytes(cp) // row_bytes
    if rows < 1:
        raise MemoryError("insufficient GPU memory for an assignment batch")
    device_input = isinstance(points, cp.ndarray)
    if device_input or isinstance(points, np.ndarray):
        rows = min(rows, max(1, len(_as_2d_xy(points))))
    with device, queue:
        while True:
            try:
                xy, labels = (
                    None if device_input else cp.empty((rows, 2), coords.dtype),
                    cp.empty(rows, cp.int32),
                )
                break
            except MemoryError:
                if rows == 1:
                    raise
                rows = max(1, rows // 2)
    if device_input:
        _validate_query_arrays(points, polygons)
        for start in range(0, len(points), rows):
            block = points[start : start + rows]
            with device, queue:
                result = labels[: len(block)]
                assign_points(
                    block, polygons, predicate=predicate, out=result, stream=stream
                )
                host = result.get()
            yield host
        return
    with device, queue:
        try:
            storage = cp.cuda.alloc_pinned_memory(2 * xy.nbytes + labels.nbytes)
        except (MemoryError, cp.cuda.runtime.CUDARuntimeError):
            storage = np.empty(2 * xy.nbytes + labels.nbytes, np.uint8)
        inputs = np.ndarray((2, rows, 2), coords.dtype, buffer=storage)
        output = np.ndarray(rows, np.int32, buffer=storage, offset=inputs.nbytes)
        pageable = isinstance(storage, np.ndarray)
    batches = host_points(points, rows=rows, dtype=coords.dtype, buffers=cycle(inputs))
    block = next(batches, None)
    while block is not None:
        n = len(block)
        try:
            with device, queue:
                xy[:n].set(block, stream=queue)
                assign_points(
                    xy[:n], polygons, predicate=predicate, out=labels[:n], stream=stream
                )
                labels[:n].get(out=output[:n], stream=queue, blocking=pageable)
            # Pack the next host slot while this upload, assignment and download run.
            block = next(batches, None)
        finally:
            with device:
                queue.synchronize()
        yield output[:n]


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
        CuPy polygons also accept NumPy coordinates, a points frame, an Arrow
        dataset, or batches of x/y columns.
    polygons
        :class:`cuspa.Polygons`. Its ``aabbs`` and ``index`` fields are
        populated on first call and reused on subsequent calls.
    predicate
        ``"contains"`` (default) — strict interior; on-edge points excluded
        (matches GEOS ``contains``). ``"intersects"`` — edge-inclusive.
    out
        Optional writable ``int32`` array of length ``N``, including a chunked host array.
    stream
        CUDA stream handle. ``0`` means the default stream.

    Returns
    -------
    array
        One ``int32`` polygon index per point, or ``-1`` for an unassigned point.
        Returns ``out`` when supplied. Otherwise host inputs return NumPy labels
        and device inputs return device labels. If a CuPy label array exceeds
        available GPU memory, the labels are returned in NumPy.

    Notes
    -----
    When polygons overlap, the first containing polygon found (in index order
    within each grid cell) wins. Use :func:`overlap_pairs` if you need all
    containing polygons per point.
    """
    module = _array_mod(points)
    if module not in ("cupy", "torch") or (
        out is not None and _array_mod(out) not in ("cupy", "torch")
    ):
        if out is not None and (
            _array_mod(out) in ("cupy", "torch")
            or getattr(out, "dtype", None) != np.int32
        ):
            raise TypeError("out must be a writable host int32 array")
        if isinstance(points, np.ndarray) or module in ("cupy", "torch"):
            points = _as_2d_xy(points)
            rows = len(points)
        elif hasattr(points, "iloc") and not hasattr(points, "get_partition"):
            rows = len(points)
        else:
            rows = getattr(points, "num_rows", None)
        if isinstance(out, np.ndarray) and not out.flags.writeable:
            raise ValueError("out must be writable")
        if (
            isinstance(points, np.ndarray)
            and isinstance(out, np.ndarray)
            and np.shares_memory(out, points)
        ):
            raise ValueError("out must not overlap input coordinates")
        labels = _assignment_batches(
            points, polygons, predicate=predicate, stream=stream
        )
        return _collect_host(labels, np.int32, size=rows, out=out)
    points_xy = _as_2d_xy(points)
    poly_xy = _as_2d_xy(polygons.points_xy)
    _require_stream(stream)
    _validate_query_arrays(points_xy, polygons, out)

    edge_flag = _predicate_flag(predicate)
    idx = polygons.ensure_index(stream=stream)
    n = int(points_xy.shape[0])
    if out is None:
        try:
            out = _empty_int32_like(points_xy, n)
        except MemoryError:
            if module != "cupy":
                raise
        if out is None:
            labels = _assignment_batches(
                points_xy, polygons, predicate=predicate, stream=stream
            )
            return _collect_host(labels, np.int32, size=n)

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
        Device XY tensor. CuPy polygons also accept NumPy coordinates, a points
        frame, an Arrow dataset, or batches of x/y columns.
    polygons
        :class:`cuspa.Polygons`. The shared uniform-grid index is built on
        first call and reused.
    predicate
        ``"contains"`` (default) or ``"intersects"`` (edge-inclusive).
    stream
        CUDA stream handle.

    Returns
    -------
    array
        All ``(point_idx, polygon_idx)`` pairs, ordered by input point and then
        by the polygon's position in the grid cell list. Host inputs return an
        ``int64`` NumPy array. Device inputs return an ``int32`` device tensor.
        CuPy inputs use the host result when there are more than ``2**31 - 1``
        input points or GPU memory is insufficient for the operation.
    """
    module = _array_mod(points)
    if module == "torch" or (module == "cupy" and len(_as_2d_xy(points)) <= 2**31 - 1):
        try:
            return next(_overlap_pages(points, polygons, predicate, stream, None))
        except (RuntimeError, MemoryError) as exc:
            if module != "cupy" or not _oom(exc):
                raise
    batches = _overlap_batches(points, polygons, predicate=predicate, stream=stream)
    return _collect_host(batches, np.int64, (2,))


def _overlap_pages(points, polygons, predicate, stream, max_pairs):
    points_xy = _as_2d_xy(points)
    poly_xy = _as_2d_xy(polygons.points_xy)
    _require_stream(stream)
    _validate_query_arrays(points_xy, polygons)
    n = int(points_xy.shape[0])
    if n > 2**31 - 1:
        raise OverflowError("overlap_pairs supports at most 2^31-1 input points")

    idx = polygons.ensure_index(stream=stream)
    edge_flag = _predicate_flag(predicate)

    # int64 offsets let the total pair count exceed 2^31-1.
    offsets = _empty_like(points_xy, n + 1, "int64")
    args = (
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
    )
    count = int(_core.overlap_count_and_scan(*args, edge_flag, stream))
    limit = max_pairs or max(1, count)
    first = 0
    while first < max(1, count):
        n_pairs = min(limit, count - first)
        try:
            pairs = _empty_int32_like(points_xy, (n_pairs, 2))
        except MemoryError:
            if max_pairs is None or n_pairs < 2:
                raise
            limit = n_pairs // 2
            continue
        if count:
            _core.overlap_emit(*args, pairs, edge_flag, stream, first)
        yield pairs
        first += max(1, n_pairs)
        del pairs


def _overlap_batches(
    points: Any,
    polygons: Polygons,
    *,
    predicate: Predicate = "contains",
    stream: int = 0,
) -> Iterator[np.ndarray]:
    """Yield NumPy int64 (global point index, polygon index) pairs in input order.

    Accepts host point sources or a CuPy array with CuPy polygons. Input and
    output batches are bounded independently, including multiple hits per point.
    """
    import cupy as cp

    from ..io._points import host_points

    _predicate_flag(predicate)
    _require_stream(stream)
    coords = polygons.points_xy
    if not isinstance(coords, cp.ndarray):
        raise TypeError("streamed overlaps require CuPy polygons")
    device = coords.device
    with device:
        queue = cp.cuda.ExternalStream(stream) if stream else cp.cuda.Stream.null
        with queue:
            polygons.ensure_index(stream=stream)
        budget = _workspace_bytes(cp)
    if budget < 64:
        raise MemoryError("insufficient GPU memory for an overlap batch")
    rows = budget // 64  # XY, counts, scan offsets and scratch in addition to output.
    if isinstance(points, cp.ndarray):
        points = _as_2d_xy(points)
        _validate_query_arrays(points, polygons)
        blocks = (points,)
    else:
        blocks = host_points(points, rows=rows, dtype=coords.dtype)
    base = 0
    for block in blocks:
        start = 0
        while start < len(block):
            n = min(rows, len(block) - start)
            try:
                with device, queue:
                    pages = _overlap_pages(
                        cp.asarray(block[start : start + n]),
                        polygons,
                        predicate,
                        stream,
                        budget // 32,
                    )
                    page = next(pages)
            except (MemoryError, RuntimeError) as exc:
                if n == 1 or not _oom(exc):
                    raise
                rows = n // 2
                continue
            while page is not None:
                with device, queue:
                    pairs = page.get().astype(np.int64)
                    del page
                if len(pairs):
                    pairs[:, 0] += base
                    yield pairs
                with device, queue:
                    page = next(pages, None)
            start += n
            base += n
