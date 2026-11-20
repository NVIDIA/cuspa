# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SpatialData ↔ cuspa adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ._geoarrow import from_geopandas
from ._points import _parquet_scan, _partitions, _prefetch_one, _string_gene

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


def _gene_categories(series):
    import pandas as pd

    categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if categorical and getattr(series.cat, "known", True):
        return series.dtype.categories
    seen = {}
    for part in _partitions(series):
        values = part.cat.categories if categorical else part.dropna().unique()
        seen.update(dict.fromkeys(values))
        del part
    # Categorical partitions retain declared order, including unused categories.
    return pd.Index(seen) if categorical else pd.Categorical(list(seen)).categories


def _arrow_transcripts(scan, gene_column, dtype):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    seen = set()
    for batch in scan([gene_column]):
        genes = batch[gene_column]
        if pa.types.is_dictionary(genes.type):
            dictionary = genes.dictionary
            if len(dictionary) <= len(seen) and seen.issuperset(
                dictionary.to_numpy(zero_copy_only=False)
            ):
                continue
            # A dictionary can include unused entries. Confirm which IDs occur.
            values = pc.take(dictionary, pc.unique(genes.indices))
        else:
            values = pc.unique(genes)
        seen.update(values.drop_null().to_numpy(zero_copy_only=False))
    names = sorted(seen)
    vocabulary = pa.array(names, type=pa.string())

    def batches():
        for batch in scan(["x", "y", gene_column]):
            genes = batch[gene_column].dictionary_encode()
            lookup = pc.index_in(genes.dictionary, value_set=vocabulary)
            ids = pc.take(lookup, genes.indices)
            if ids.null_count:
                ids = pc.fill_null(ids, -1)
            xy = np.empty((len(batch), 2), dtype=dtype)
            xy[:, 0], xy[:, 1] = batch["x"], batch["y"]
            yield xy, ids.to_numpy()

    return lambda: _prefetch_one(batches()), names


def _transcript_source(pts, gene_column, dtype):
    import numpy as np

    dtype = np.dtype(np.float32 if dtype is None else dtype)
    if callable(getattr(pts, "scanner", None)):
        import pyarrow.dataset as ds

        from ._points import _dataset_batches, _point_columns

        if not _string_gene(pts.schema, gene_column):
            raise ValueError(
                f"Arrow points must contain string gene column {gene_column!r}"
            )
        source = _point_columns(pts.schema)
        source[gene_column] = gene_column

        def scan(columns):
            projection = {name: ds.field(source[name]) for name in columns}
            return _dataset_batches(pts, projection, 1 << 20, dictionary=gene_column)

        return _arrow_transcripts(scan, gene_column, dtype)

    import pandas as pd

    columns = ["x", "y"]
    if "x" not in pts.columns or "y" not in pts.columns:
        raise ValueError("points must have 'x' and 'y' columns")
    categories = None
    if gene_column is not None and gene_column in pts.columns:
        if isinstance(pts.dtypes[gene_column], pd.StringDtype):
            scan = _parquet_scan(pts, dictionary=gene_column)
            if scan is not None:
                return _arrow_transcripts(
                    lambda cols: scan(cols, 1 << 20), gene_column, dtype
                )
        categories = _gene_categories(pts[gene_column])
        if gene_column not in columns:
            columns.append(gene_column)

    def batches():
        for frame in _partitions(pts[columns]):
            for start in range(0, max(1, len(frame)), 1 << 20):
                part = frame.iloc[start : start + (1 << 20)]
                xy = np.empty((len(part), 2), dtype=dtype)
                xy[:, 0], xy[:, 1] = part["x"], part["y"]
                ids = None
                if categories is not None:
                    gene = pd.Categorical(part[gene_column]).set_categories(categories)
                    ids = gene.codes.astype(np.int32, copy=False)
                yield xy, ids

    return batches, None if categories is None else [str(x) for x in categories]


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
    if hasattr(pts, "get_partition"):
        batches, names = _transcript_source(pts, gene_column, dtype)
        xy = cp.empty((len(pts["x"]), 2), dtype=dtype)
        ids = None if names is None else cp.empty(len(xy), dtype=cp.int32)
        stop = 0
        for coords, codes in batches():
            start, stop = stop, stop + len(coords)
            xy[start:stop].set(coords)
            if ids is not None:
                ids[start:stop].set(codes)
        if stop != len(xy):
            raise ValueError("number of points changed while reading the points frame")
        return Transcripts(xy, ids, names)
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
