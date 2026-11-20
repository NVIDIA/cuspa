# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def _partitions(frame):
    if hasattr(frame, "get_partition"):
        return (frame.get_partition(i).compute() for i in range(frame.npartitions))
    return (frame,)


def _string_gene(schema, name):
    import pyarrow as pa

    if name not in schema.names:
        return False
    field = schema.field(name).type
    if pa.types.is_dictionary(field):
        field = field.value_type
    return pa.types.is_string(field) or pa.types.is_large_string(field)


def _parquet_scan(frame, dictionary=None):
    """Project validated Dask Parquet reads in their original partition order."""
    expr = getattr(frame, "expr", None)
    aliases = {}
    # PointsModel.parse renames the projected x/y columns and assigns the
    # other columns from the same read. Other transforms use Dask.
    if type(expr).__module__ == "dask.dataframe.dask_expr._expr":
        try:
            from dask.dataframe.dask_expr._expr import Assign, ColumnsSetter, Projection
        except ImportError:
            return None

        sources = []
        while type(expr) is Assign:
            if len(expr.operands) != 3:
                return None
            expr, name, series = expr.operands
            if (
                type(series) is not Projection
                or name in ("x", "y")
                or series.operands[1] != name
            ):
                return None
            sources.append(series.operands[0])
        if type(expr) is not ColumnsSetter:
            return None
        projection, renamed = expr.operands
        if type(projection) is not Projection or renamed != ["x", "y"]:
            return None
        expr, physical = projection.operands
        if any(source is not expr for source in sources):
            return None
        if not isinstance(physical, list) or len(physical) != len(renamed):
            return None
        aliases = dict(zip(renamed, physical, strict=True))
    parquet = "dask.dataframe.dask_expr.io.parquet"
    if type(expr).__module__ != parquet or type(expr).__name__ != "ReadParquetFSSpec":
        return None
    options = dict(zip(expr._parameters, expr.operands, strict=False))
    overrides = ("filters", "_partitions", "categories", "index")
    dtypes = [frame.dtypes.get(k) for k in ("x", "y")]
    if (
        options.get("engine") != "pyarrow"
        or options.get("kwargs") not in ({}, {"dtype_backend": None})
        or any(k not in options or options[k] is not None for k in overrides)
        or any(str(t) not in ("float32", "float64") for t in dtypes)
    ):
        return None
    info = getattr(expr, "_dataset_info", None)
    match info, getattr(expr, "_plan", None):
        case {"ds": ds, "partition_names": []}, {"parts": [*parts]} if ds is not None:
            import pyarrow.dataset as arrow

            fmt, fs = ds.format, ds.filesystem
            if dictionary is not None:
                fmt = arrow.ParquetFileFormat(
                    read_options=arrow.ParquetReadOptions(
                        dictionary_columns=[dictionary]
                    )
                )
            sources, fragments = {}, []
            for part in parts:
                match part:
                    case {"piece": (str() as path, groups, None | [])} if (
                        len(part) == 1
                    ):
                        if groups is not None and not isinstance(groups, list):
                            return None
                    case _:
                        return None
                if path not in sources:
                    sources[path] = fmt.make_fragment(path, filesystem=fs)
                    if dictionary and not _string_gene(
                        sources[path].physical_schema, dictionary
                    ):
                        return None
                fragment = sources[path]
                fragments.append(
                    fragment
                    if groups is None
                    else fragment.subset(row_group_ids=groups)
                )

            def scan(columns, rows):
                if aliases:
                    columns = {
                        name: arrow.field(aliases.get(name, name)) for name in columns
                    }
                return _scan_fragments(fragments, columns, rows)

            return scan if fragments else None


def _point_columns(schema):
    result = {}
    for axis in ("x", "y"):
        field = axis if axis in schema.names else f"{axis}_location"
        if field not in schema.names:
            raise ValueError(f"point table has no {axis!r} or {field!r} column")
        result[axis] = field
    return result


def _read_batches(source, columns, rows, **options):
    return source.to_batches(
        columns=columns,
        batch_size=min(rows, 1 << 20),
        batch_readahead=1,
        fragment_readahead=2,
        **options,
    )


def _scan_fragments(fragments, columns, rows, *, dataset=None, dictionary=None):
    from itertools import batched, groupby

    import pyarrow as pa
    import pyarrow.dataset as ds

    # Keep Arrow's compressed prefetch bounded even on a single large Parquet file.
    def pieces():
        for fragment in fragments:
            schema = dataset.schema if dataset is not None else fragment.physical_schema
            row_groups = fragment.row_groups
            if (
                dictionary is not None
                and fragment.filesystem is not None
                and fragment.format.equals(dataset.format)
                and schema.equals(fragment.physical_schema)
            ):
                fmt, name = dictionary
                fragment = fmt.make_fragment(
                    fragment.path,
                    fragment.filesystem,
                    fragment.partition_expression,
                    [r.id for r in row_groups],
                )
                field = schema.field(name)
                schema = schema.set(
                    schema.get_field_index(name),
                    field.with_type(pa.dictionary(pa.int32(), field.type)),
                )
            if len(row_groups) <= 8:
                yield schema, fragment
            else:
                for part in batched(row_groups, 8):
                    yield schema, fragment.subset(row_group_ids=[r.id for r in part])

    for group in batched(pieces(), 12 if len(columns) == 1 else 3):
        for (schema, fmt), same_schema in groupby(
            group, key=lambda p: (p[0], p[1].format)
        ):
            files = [f for _, f in same_schema]
            first = files[0]
            source = ds.FileSystemDataset(
                files,
                schema,
                fmt,
                first.filesystem,
                root_partition=getattr(dataset, "partition_expression", None),
            )
            yield from _read_batches(source, columns, rows)


def _dataset_batches(dataset, columns, rows, *, dictionary=None):
    import pyarrow as pa
    import pyarrow.dataset as ds

    if isinstance(dataset, ds.FileSystemDataset) and isinstance(
        dataset.format, ds.ParquetFileFormat
    ):
        default = ds.ParquetFileFormat()
        plain = dataset.format.equals(default)
        if dictionary is not None:
            try:
                fragments = dataset.get_fragments()
            except ValueError:
                # Arrow applies the original dataset's filter or projection.
                pass
            else:
                encoding = None
                if plain and dictionary in dataset.schema.names:
                    field = dataset.schema.field(dictionary)
                    if pa.types.is_string(field.type) or pa.types.is_large_string(
                        field.type
                    ):
                        fmt = ds.ParquetFileFormat(
                            read_options=ds.ParquetReadOptions(
                                dictionary_columns=[dictionary]
                            )
                        )
                        encoding = fmt, dictionary
                return _scan_fragments(
                    fragments, columns, rows, dataset=dataset, dictionary=encoding
                )
        if plain and "fragment_scan_options" not in getattr(
            dataset, "_scan_options", {}
        ):
            return _read_batches(
                dataset,
                columns,
                rows,
                fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False),
            )
    return _read_batches(dataset, columns, rows)


def _prefetch_one(batches):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing

    with closing(batches), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, batches, None)
        while (batch := future.result()) is not None:
            future = pool.submit(next, batches, None)
            yield batch


def host_points(points, *, rows, dtype, buffers=None):
    """Yield contiguous XY slices in polygon precision, in source order."""
    if hasattr(points, "get_partition"):
        scan = _parquet_scan(points)
        if scan is None:
            batches = _partitions(points[["x", "y"]])
        else:
            batches = scan(["x", "y"], rows)
    elif hasattr(points, "scanner"):
        batches = _dataset_batches(
            points, list(_point_columns(points.schema).values()), rows
        )
    elif isinstance(points, np.ndarray) or hasattr(points, "columns"):
        batches = (points,)
    elif hasattr(points, "to_batches"):
        batches = points.to_batches()
    else:
        batches = points
    for batch in batches:
        yield from _slices(batch, rows, dtype, buffers)
        del batch  # Release the live partition before computing the next.


def _slices(batch, rows, dtype, buffers):
    array = isinstance(batch, np.ndarray)
    if array:
        if batch.ndim == 1 and batch.size % 2 == 0:
            batch = batch.reshape(-1, 2)
        if batch.ndim != 2 or batch.shape[1] != 2:
            raise ValueError("expected (N, 2) coordinates or a flat interleaved array")
        if batch.dtype.kind not in "fiu":
            raise TypeError("point coordinates must be numeric")
    else:
        try:
            columns = (
                _point_columns(batch.schema).values()
                if hasattr(batch, "schema")
                else ("x", "y")
            )
            x, y = (batch[k] for k in columns)
        except (KeyError, TypeError) as exc:
            raise ValueError("point tables must contain 'x' and 'y' columns") from exc
        x, y = getattr(x, "iloc", x), getattr(y, "iloc", y)
    for start in range(0, len(batch), rows):
        n = min(rows, len(batch) - start)
        xy = np.empty((n, 2), dtype=dtype) if buffers is None else next(buffers)[:n]
        if array:
            xy[:] = batch[start : start + n]
        else:
            xy[:, 0], xy[:, 1] = x[start : start + n], y[start : start + n]
        yield xy
