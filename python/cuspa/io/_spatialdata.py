# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SpatialData ↔ cuspa adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ._geoarrow import from_geopandas

if TYPE_CHECKING:
    from .._types import Polygons


class Transcripts(NamedTuple):
    """Transcript coordinates and gene metadata from SpatialData.

    Attributes
    ----------
    xy
        Device tensor of shape ``(N, 2)``.
    gene_ids
        Optional device ``int32[N]`` of gene indices (from a categorical
        encoding of ``gene_column``), or ``None`` if no gene column.
    gene_names
        List of gene names in category order, so ``gene_names[gene_ids[i]]``
        recovers the original string. ``None`` if no gene column.
    """

    xy: Any
    gene_ids: Any | None
    gene_names: list[str] | None


def _require_cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "cuspa.io adapters need cupy for the GPU transfer. "
            "Install the CuPy package matching your CUDA runtime."
        ) from e
    return cp


def from_spatialdata(sdata: Any, shape_key: str, *, dtype: Any = None) -> Polygons:
    """Convert a SpatialData ShapesModel element to :class:`Polygons`.

    Parameters
    ----------
    sdata
        A ``spatialdata.SpatialData`` object.
    shape_key
        Name of a shapes element. ``sdata[shape_key]`` must be a GeoDataFrame
        whose ``geometry`` column contains Shapely ``Polygon`` geometries.
    dtype
        Floating-point dtype for the coordinate tensor. Defaults to
        ``cupy.float32``.

    Returns
    -------
    :class:`cuspa.Polygons`
    """
    cp = _require_cupy()
    if dtype is None:
        dtype = cp.float32

    gdf = sdata[shape_key]
    if not hasattr(gdf, "geometry"):
        raise TypeError(
            f"sdata[{shape_key!r}] is not a shapes element (expected a GeoDataFrame)"
        )

    return from_geopandas(gdf, dtype=dtype)


def transcripts_from_spatialdata(
    sdata: Any,
    points_key: str,
    *,
    gene_column: str | None = "feature_name",
    dtype: Any = None,
) -> Transcripts:
    """Extract an (xy, gene_ids, gene_names) triple from a SpatialData points element.

    Parameters
    ----------
    sdata
        A ``spatialdata.SpatialData`` object.
    points_key
        Name of a points element. ``sdata[points_key]`` is a
        ``dask.dataframe.DataFrame`` with at least ``x`` and ``y`` columns.
    gene_column
        Column name holding the per-transcript gene label. Values are encoded
        as a categorical and emitted as ``int32`` codes suitable for
        :func:`cuspa.tl.aggregate_to_cells`. Set to ``None`` to skip.
    dtype
        Floating-point dtype for the xy tensor. Defaults to ``cupy.float32``.

    Returns
    -------
    :class:`Transcripts`
    """
    cp = _require_cupy()
    if dtype is None:
        dtype = cp.float32

    pts = sdata[points_key]
    # Dask dataframe → pandas (single materialization).
    if hasattr(pts, "compute"):
        pts = pts.compute()

    if "x" not in pts.columns or "y" not in pts.columns:
        raise ValueError(
            f"sdata[{points_key!r}] must have 'x' and 'y' columns; got {list(pts.columns)}"
        )

    xy_host = pts[["x", "y"]].to_numpy()
    xy = cp.ascontiguousarray(cp.asarray(xy_host, dtype=dtype))

    if gene_column is None or gene_column not in pts.columns:
        return Transcripts(xy=xy, gene_ids=None, gene_names=None)

    col = pts[gene_column]
    # Stable int32 encoding via pandas categorical.
    if str(col.dtype) == "category":
        cat = col
    else:
        cat = col.astype("category")
    codes = cat.cat.codes.to_numpy().astype("int32", copy=False)
    gene_ids = cp.asarray(codes)
    gene_names = [str(x) for x in cat.cat.categories]
    return Transcripts(xy=xy, gene_ids=gene_ids, gene_names=gene_names)
