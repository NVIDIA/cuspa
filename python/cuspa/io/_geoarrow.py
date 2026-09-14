# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GeoArrow and GeoPandas adapters."""

from __future__ import annotations

from typing import Any

from .._types import Polygons


def _require_pyarrow_cupy() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "cuspa.io.from_geoarrow requires pyarrow. Install pyarrow or pass "
            "raw cuspa.Polygons buffers directly."
        ) from e
    try:
        import cupy as cp
    except ImportError as e:  # pragma: no cover
        raise ImportError("cuspa.io adapters need cupy for GPU transfer.") from e
    return pa, cp


def _as_arrow_table_or_array(obj: Any) -> tuple[Any | None, Any | None]:
    pa, _ = _require_pyarrow_cupy()

    if isinstance(obj, pa.Table):
        return obj, None
    if isinstance(obj, pa.ChunkedArray | pa.Array):
        return None, obj

    # GeoPandas' ArrowTable/GeoArrowArray expose the Arrow PyCapsule protocol.
    try:
        return pa.table(obj), None
    except (TypeError, ValueError):
        try:
            return None, pa.array(obj)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "expected a pyarrow Table/Array/ChunkedArray or an object "
                "implementing the Arrow PyCapsule protocol"
            ) from e


def _field_extension_name(field: Any) -> str | None:
    meta = field.metadata or {}
    value = meta.get(b"ARROW:extension:name")
    return value.decode() if value is not None else None


def _geometry_column(table: Any, geometry: str | None) -> Any:
    if geometry is not None:
        if geometry not in table.column_names:
            raise KeyError(f"geometry column {geometry!r} not found")
        return table[geometry]

    for i, field in enumerate(table.schema):
        name = _field_extension_name(field)
        if name and name.startswith("geoarrow."):
            if name == "geoarrow.wkb":
                raise TypeError(
                    "geometry column is GeoArrow WKB; call "
                    "GeoDataFrame.to_arrow(geometry_encoding='geoarrow') or "
                    "pass geometry_encoding='geoarrow' to from_geopandas"
                )
            return table.column(i)

    if "geometry" in table.column_names:
        return table["geometry"]

    raise KeyError("could not infer geometry column; pass geometry=...")


def _combine(array: Any) -> Any:
    return array.combine_chunks() if hasattr(array, "combine_chunks") else array


def _is_list_array(array: Any) -> bool:
    return hasattr(array, "offsets") and hasattr(array, "values")


def _offsets(array: Any):
    return array.offsets.to_numpy(zero_copy_only=False)


def _normalized_list(array: Any) -> tuple[Any, Any]:
    """Return zero-based offsets and the matching child slice."""
    import numpy as np

    if getattr(array, "null_count", 0):
        raise ValueError("null geometries and null polygon parts are not supported")
    offsets = np.asarray(_offsets(array), dtype=np.int64)
    start = int(offsets[0])
    stop = int(offsets[-1])
    return offsets - start, array.values.slice(start, stop - start)


def _int32_offsets(offsets: Any, name: str) -> Any:
    import numpy as np

    offsets = np.asarray(offsets)
    if offsets.size and int(offsets[-1]) > np.iinfo(np.int32).max:
        raise OverflowError(f"{name} exceeds the int32 offset limit")
    return offsets.astype(np.int32, copy=False)


def _coords_from_vertices(vertices: Any, dtype: Any):
    import numpy as np

    if getattr(vertices, "null_count", 0):
        raise ValueError("null coordinates are not supported")

    if hasattr(vertices.type, "list_size"):
        width = int(vertices.type.list_size)
        if width < 2:
            raise TypeError("GeoArrow coordinate fixed-size-list must have x/y")
        start = int(getattr(vertices, "offset", 0)) * width
        count = len(vertices) * width
        flat = vertices.values.slice(start, count).to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=dtype).reshape(-1, width)[:, :2]

    if str(vertices.type).startswith("struct<"):
        x = vertices.field("x").to_numpy(zero_copy_only=False)
        y = vertices.field("y").to_numpy(zero_copy_only=False)
        return np.stack(
            [
                np.asarray(x, dtype=dtype),
                np.asarray(y, dtype=dtype),
            ],
            axis=1,
        )

    raise TypeError(
        "expected GeoArrow coordinates as fixed_size_list<x,y> or struct<x,y>, "
        f"got {vertices.type}"
    )


def _polygon_array_to_offsets(array: Any, dtype: Any):
    """Return ``(part_offsets, ring_offsets, coords)`` for Polygon/MultiPolygon."""
    array = _combine(array)
    if not _is_list_array(array):
        raise TypeError(
            f"expected GeoArrow Polygon/MultiPolygon list array, got {array.type}"
        )

    geometry_offsets, level1 = _normalized_list(array)
    if not _is_list_array(level1):
        raise TypeError(f"expected GeoArrow polygon ring list level, got {level1.type}")

    # Polygon:      geometry -> rings -> vertices
    # MultiPolygon: geometry -> polygons -> rings -> vertices
    level1_offsets, level2 = _normalized_list(level1)
    if _is_list_array(level2):
        part_offsets = level1_offsets[geometry_offsets]
        ring_offsets, vertices = _normalized_list(level2)
    else:
        part_offsets = geometry_offsets
        ring_offsets = level1_offsets
        vertices = level2

    coords = _coords_from_vertices(vertices, dtype)
    return (
        _int32_offsets(part_offsets, "polygon offsets"),
        _int32_offsets(ring_offsets, "ring offsets"),
        coords,
    )


def from_geoarrow(
    obj: Any,
    *,
    geometry: str | None = None,
    dtype: Any = None,
) -> Polygons:
    """Convert a GeoArrow Polygon/MultiPolygon table or array to ``Polygons``.

    Parameters
    ----------
    obj
        A pyarrow ``Table``/``Array``/``ChunkedArray`` or a GeoPandas
        ``to_arrow(geometry_encoding="geoarrow")`` result. Polygon and
        MultiPolygon arrays are supported. WKB columns are rejected because
        they require geometry parsing.
    geometry
        Geometry column name when ``obj`` is a table. If omitted, the first
        GeoArrow geometry column is used, falling back to ``"geometry"``.
    dtype
        Coordinate dtype for the GPU buffer. Defaults to ``cupy.float32``.
    """
    _, cp = _require_pyarrow_cupy()
    if dtype is None:
        dtype = cp.float32

    table, array = _as_arrow_table_or_array(obj)
    if table is not None:
        array = _geometry_column(table, geometry)

    part_offsets, ring_offsets, coords = _polygon_array_to_offsets(array, dtype)
    return Polygons(
        part_offsets=cp.asarray(part_offsets, dtype=cp.int32),
        ring_offsets=cp.asarray(ring_offsets, dtype=cp.int32),
        points_xy=cp.ascontiguousarray(cp.asarray(coords, dtype=dtype)),
    )


def from_geopandas(
    gdf: Any,
    *,
    geometry: str | None = None,
    dtype: Any = None,
    include_z: bool | None = False,
) -> Polygons:
    """Convert a GeoPandas GeoDataFrame or GeoSeries to ``Polygons``."""
    if not hasattr(gdf, "to_arrow"):
        raise TypeError("expected a GeoPandas object with a to_arrow() method")

    arrow = gdf.to_arrow(
        geometry_encoding="geoarrow",
        interleaved=True,
        include_z=include_z,
    )
    return from_geoarrow(arrow, geometry=geometry, dtype=dtype)
