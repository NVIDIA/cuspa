# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Polygon buffers and spatial indexes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import _core


def _array_mod(a: Any) -> str:
    return type(a).__module__.split(".", 1)[0]


def _dtype_name(a: Any) -> str:
    return str(a.dtype).rsplit(".", 1)[-1]


def _array_device(a: Any) -> int:
    device = a.device
    if _array_mod(a) == "cupy":
        return int(device.id)
    return int(device.index)


def _is_contiguous(a: Any) -> bool:
    if _array_mod(a) == "torch":
        return bool(a.is_contiguous())
    flags = getattr(a, "flags", None)
    return bool(flags is not None and flags.c_contiguous)


def _scalar(a: Any) -> Any:
    return a.item() if hasattr(a, "item") else a


def _any(a: Any) -> bool:
    value = a.any()
    return bool(_scalar(value))


def _all_finite(a: Any) -> bool:
    mod = _array_mod(a)
    if mod == "cupy":
        import cupy as xp  # type: ignore[import-not-found]
    elif mod == "torch":
        import torch as xp  # type: ignore[import-not-found]
    else:
        return False
    return bool(_scalar(xp.isfinite(a).all()))


def _require_array(
    a: Any,
    name: str,
    *,
    ndim: int,
    dtype: str | tuple[str, ...],
    module: str | None = None,
    device: int | None = None,
) -> str:
    mod = _array_mod(a)
    if mod not in {"cupy", "torch"}:
        raise TypeError(f"{name} must be a CuPy or CUDA PyTorch array, got {type(a)!r}")
    if mod == "torch" and not bool(a.is_cuda):
        raise TypeError(f"{name} must be a CUDA PyTorch array")
    if module is not None and mod != module:
        raise TypeError(f"all polygon buffers must use {module}; {name} uses {mod}")
    if device is not None and _array_device(a) != device:
        raise TypeError(f"all buffers must use CUDA device {device}; {name} does not")
    if getattr(a, "ndim", None) != ndim:
        raise ValueError(
            f"{name} must be {ndim}-D, got shape {getattr(a, 'shape', None)}"
        )
    accepted = (dtype,) if isinstance(dtype, str) else dtype
    if _dtype_name(a) not in accepted:
        choices = " or ".join(accepted)
        raise TypeError(f"{name} must have dtype {choices}, got {a.dtype}")
    if not _is_contiguous(a):
        raise ValueError(f"{name} must be C-contiguous")
    return mod


def _require_stream(stream: int) -> None:
    if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
        raise ValueError("stream must be a non-negative integer CUDA stream handle")


def _empty_like(ref: Any, shape: tuple[int, ...] | int, dtype: Any = None) -> Any:
    """Allocate an empty device tensor in ``ref``'s framework."""
    mod = _array_mod(ref)
    if mod == "cupy":
        import cupy as xp  # type: ignore[import-not-found]

        return xp.empty(shape, dtype=dtype if dtype is not None else ref.dtype)
    if mod == "torch":
        import torch  # type: ignore[import-not-found]

        if dtype is None:
            dtype = ref.dtype
        elif dtype is int or dtype == "int32":
            dtype = torch.int32
        return torch.empty(shape, dtype=dtype, device=ref.device)
    raise TypeError(
        f"Cannot auto-allocate for array of type {type(ref)!r}; "
        "pass buffers in explicitly."
    )


@dataclass(slots=True)
class SpatialIndex:
    """Uniform-grid spatial index over polygon AABBs.

    ``grid_offsets`` and ``poly_ids`` store a CSR-style mapping from grid cells
    to polygon IDs.
    """

    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nx: int
    ny: int
    grid_offsets: Any  # int32[nx*ny + 1]
    poly_ids: Any  # int32[K]
    aabbs: Any  # float[P, 4]

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny

    def __post_init__(self) -> None:
        if (
            isinstance(self.nx, bool)
            or not isinstance(self.nx, int)
            or self.nx <= 0
            or isinstance(self.ny, bool)
            or not isinstance(self.ny, int)
            or self.ny <= 0
        ):
            raise ValueError("SpatialIndex nx and ny must be positive")
        if self.num_cells > 2**31 - 1:
            raise ValueError("SpatialIndex nx * ny must fit in int32")
        if not math.isfinite(self.origin_x) or not math.isfinite(self.origin_y):
            raise ValueError("SpatialIndex origin must be finite")
        if not math.isfinite(self.cell_size_x) or self.cell_size_x <= 0:
            raise ValueError("SpatialIndex cell_size_x must be finite and positive")
        if not math.isfinite(self.cell_size_y) or self.cell_size_y <= 0:
            raise ValueError("SpatialIndex cell_size_y must be finite and positive")
        module = _require_array(
            self.grid_offsets, "grid_offsets", ndim=1, dtype="int32"
        )
        device = _array_device(self.grid_offsets)
        _require_array(
            self.poly_ids,
            "poly_ids",
            ndim=1,
            dtype="int32",
            module=module,
            device=device,
        )
        _require_array(
            self.aabbs,
            "aabbs",
            ndim=2,
            dtype=("float32", "float64"),
            module=module,
            device=device,
        )
        if tuple(self.aabbs.shape)[1:] != (4,):
            raise ValueError("aabbs must have shape (P, 4)")
        if int(self.grid_offsets.shape[0]) != self.num_cells + 1:
            raise ValueError("grid_offsets length must equal nx * ny + 1")
        if int(_scalar(self.grid_offsets[0])) != 0:
            raise ValueError("grid_offsets must start at 0")
        if int(_scalar(self.grid_offsets[-1])) != int(self.poly_ids.shape[0]):
            raise ValueError("grid_offsets[-1] must equal the number of polygon ids")
        if _any(self.grid_offsets[1:] < self.grid_offsets[:-1]):
            raise ValueError("grid_offsets must be non-decreasing")
        if int(self.poly_ids.shape[0]):
            if int(_scalar(self.poly_ids.min())) < 0 or int(
                _scalar(self.poly_ids.max())
            ) >= int(self.aabbs.shape[0]):
                raise ValueError("poly_ids contains an out-of-range polygon index")
        if not _all_finite(self.aabbs):
            raise ValueError("aabbs must contain only finite values")


@dataclass(slots=True)
class Polygons:
    """A batch of polygons in GeoArrow offsets form.

    Parameters
    ----------
    part_offsets
        ``int32`` tensor of length ``num_polygons + 1``. ``part_offsets[i:i+2]``
        is the half-open range of rings that belong to polygon ``i``.
    ring_offsets
        ``int32`` tensor of length ``num_rings + 1``. ``ring_offsets[r:r+2]``
        is the half-open range of points that belong to ring ``r`` (rings
        must be explicitly closed: first point == last point).
    points_xy
        Interleaved coordinate tensor of shape ``(num_points, 2)``
        (float32 or float64).
    aabbs
        Optional precomputed per-polygon AABBs, ``(num_polygons, 4)``.
    index
        Optional precomputed :class:`SpatialIndex`.
    """

    part_offsets: Any
    ring_offsets: Any
    points_xy: Any
    aabbs: Any | None = field(default=None)
    index: SpatialIndex | None = field(default=None)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate polygon buffers."""
        module = _require_array(
            self.part_offsets, "part_offsets", ndim=1, dtype="int32"
        )
        device = _array_device(self.part_offsets)
        _require_array(
            self.ring_offsets,
            "ring_offsets",
            ndim=1,
            dtype="int32",
            module=module,
            device=device,
        )
        _require_array(
            self.points_xy,
            "points_xy",
            ndim=2,
            dtype=("float32", "float64"),
            module=module,
            device=device,
        )
        if tuple(self.points_xy.shape)[1:] != (2,):
            raise ValueError("points_xy must have shape (N, 2)")
        if int(self.part_offsets.shape[0]) < 2:
            raise ValueError("part_offsets must describe at least one polygon")
        if int(self.ring_offsets.shape[0]) < 2:
            raise ValueError("ring_offsets must describe at least one ring")

        num_rings = int(self.ring_offsets.shape[0]) - 1
        num_points = int(self.points_xy.shape[0])
        if int(_scalar(self.part_offsets[0])) != 0:
            raise ValueError("part_offsets must start at 0")
        if int(_scalar(self.part_offsets[-1])) != num_rings:
            raise ValueError("part_offsets[-1] must equal the number of rings")
        if _any(self.part_offsets[1:] <= self.part_offsets[:-1]):
            raise ValueError("each polygon must contain at least one ring")
        if int(_scalar(self.ring_offsets[0])) != 0:
            raise ValueError("ring_offsets must start at 0")
        if int(_scalar(self.ring_offsets[-1])) != num_points:
            raise ValueError("ring_offsets[-1] must equal the number of points")
        if _any((self.ring_offsets[1:] - self.ring_offsets[:-1]) < 3):
            raise ValueError("each ring must contain at least three points")
        if not _all_finite(self.points_xy):
            raise ValueError("points_xy must contain only finite coordinates")

        if self.aabbs is not None:
            _require_array(
                self.aabbs,
                "aabbs",
                ndim=2,
                dtype=_dtype_name(self.points_xy),
                module=module,
                device=device,
            )
            if tuple(self.aabbs.shape) != (self.num_polygons, 4):
                raise ValueError("aabbs must have shape (num_polygons, 4)")
            if not _all_finite(self.aabbs):
                raise ValueError("aabbs must contain only finite values")
            if _any(self.aabbs[:, 0] > self.aabbs[:, 2]) or _any(
                self.aabbs[:, 1] > self.aabbs[:, 3]
            ):
                raise ValueError("aabbs minima must not exceed maxima")

        if self.index is not None:
            if _array_mod(self.index.aabbs) != module:
                raise TypeError("index buffers must use the polygon array provider")
            if _array_device(self.index.aabbs) != device:
                raise TypeError("index buffers must use the polygon CUDA device")
            if tuple(self.index.aabbs.shape) != (self.num_polygons, 4):
                raise ValueError("index aabbs must match the polygon count")

    @property
    def num_polygons(self) -> int:
        return int(self.part_offsets.shape[0]) - 1

    @property
    def num_rings(self) -> int:
        return int(self.ring_offsets.shape[0]) - 1

    @property
    def num_points(self) -> int:
        return int(self.points_xy.shape[0])

    def ensure_aabbs(self, *, stream: int = 0) -> Any:
        _require_stream(stream)
        if self.aabbs is None:
            self.validate()
            buf = _empty_like(self.points_xy, (self.num_polygons, 4))
            _core.compute_poly_aabbs(
                self.part_offsets,
                self.ring_offsets,
                self.points_xy,
                buf,
                stream,
            )
            self.aabbs = buf
        return self.aabbs

    def ensure_index(
        self,
        *,
        nx: int | None = None,
        ny: int | None = None,
        stream: int = 0,
    ) -> SpatialIndex:
        """Build or return a cached uniform-grid index.

        The default grid dimensions are ``ceil(sqrt(num_polygons))``.
        """
        _require_stream(stream)
        P = self.num_polygons

        if self.index is not None and nx is None and ny is None:
            return self.index

        side = max(1, int(math.ceil(math.sqrt(P))))
        if self.index is not None:
            nx = self.index.nx if nx is None else nx
            ny = self.index.ny if ny is None else ny
            if nx == self.index.nx and ny == self.index.ny:
                return self.index
        else:
            nx = side if nx is None else nx
            ny = side if ny is None else ny

        if isinstance(nx, bool) or not isinstance(nx, int) or nx <= 0:
            raise ValueError("nx must be a positive integer")
        if isinstance(ny, bool) or not isinstance(ny, int) or ny <= 0:
            raise ValueError("ny must be a positive integer")
        if nx * ny > 2**31 - 1:
            raise ValueError("nx * ny must fit in int32")

        aabbs = self.ensure_aabbs(stream=stream)
        # Provider reductions may use a different stream than AABB construction.
        if stream:
            _core.synchronize_stream(stream)

        minx = float(aabbs[:, 0].min())
        miny = float(aabbs[:, 1].min())
        maxx = float(aabbs[:, 2].max())
        maxy = float(aabbs[:, 3].max())

        # Avoid zero-width grid cells for degenerate bounds.
        span_x = maxx - minx
        span_y = maxy - miny
        if span_x <= 0:
            span_x = 1.0
        if span_y <= 0:
            span_y = 1.0
        # Inflate the upper bounds so `floor((maxx - origin) / cell)` for a
        # point exactly at maxx still lands in cell nx-1, not nx.
        eps_x = span_x * 1e-6
        eps_y = span_y * 1e-6
        cell_size_x = (span_x + eps_x) / nx
        cell_size_y = (span_y + eps_y) / ny
        origin_x = minx
        origin_y = miny

        K = int(
            _core.compute_index_size(
                aabbs, origin_x, origin_y, cell_size_x, cell_size_y, nx, ny, stream
            )
        )
        if K > 2**31 - 1:
            raise OverflowError(
                "spatial index has more than 2^31-1 polygon-cell pairs; "
                "use a coarser grid"
            )

        poly_ids = _empty_like(self.part_offsets, (K,))
        grid_offsets = _empty_like(self.part_offsets, (nx * ny + 1,))

        _core.build_index(
            aabbs,
            origin_x,
            origin_y,
            cell_size_x,
            cell_size_y,
            nx,
            ny,
            poly_ids,
            grid_offsets,
            stream,
        )
        if stream:
            _core.synchronize_stream(stream)

        self.index = SpatialIndex(
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            nx=nx,
            ny=ny,
            grid_offsets=grid_offsets,
            poly_ids=poly_ids,
            aabbs=aabbs,
        )
        return self.index

    def clear_cache(self) -> None:
        """Discard derived AABBs and the spatial index after buffer mutation."""
        self.aabbs = None
        self.index = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        cached = []
        if self.aabbs is not None:
            cached.append("aabbs")
        if self.index is not None:
            cached.append("index")
        tag = ",".join(cached) or "lazy"
        return (
            f"Polygons(num_polygons={self.num_polygons}, "
            f"num_rings={self.num_rings}, num_points={self.num_points}, "
            f"cached={{{tag}}})"
        )
