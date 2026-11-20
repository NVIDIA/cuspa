# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import _core
from .._types import (
    Polygons,
    _array_device,
    _array_mod,
    _require_array,
    _require_stream,
)
from ._assign import _as_2d_xy, _empty_int32_like, _oom, assign_points

if TYPE_CHECKING:
    from collections.abc import Iterable


# Largest non-zero count that int32 indptr and indices can address. Beyond
# it, aggregate_to_cells switches both to int64, as scipy does.
_INT32_MAX = 2**31 - 1


@dataclass(slots=True)
class CSRMatrix:
    """CSR matrix backed by CuPy, CUDA PyTorch, or NumPy arrays.

    ``indptr`` and ``indices`` share one dtype: ``int32``, or ``int64`` when the
    matrix has more than ``2**31 - 1`` non-zeros. ``data`` is always ``int32``.
    """

    indptr: Any  # int32 or int64 [num_cells + 1]
    indices: Any  # same dtype as indptr [K]  gene ids in CSR column order
    data: Any  # int32[K]  counts
    shape: tuple[int, int]
    gene_names: list[str] | None = None

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    def to_cupy_csr(self, dtype: Any = "float32") -> Any:
        """Wrap as ``cupyx.scipy.sparse.csr_matrix``.

        ``cupyx.scipy.sparse`` only supports float and complex data. Counts are
        cast to ``dtype``, which defaults to ``float32``.

        ``cupyx.scipy.sparse`` also only supports ``int32`` indices, so this
        raises ``OverflowError`` for a matrix with more than ``2**31 - 1``
        non-zeros. Use :meth:`to_torch_sparse` or the arrays directly instead.
        """
        import cupy as cp  # type: ignore[import-not-found]

        if self.nnz > _INT32_MAX:
            raise OverflowError(
                f"cupyx.scipy.sparse supports at most {_INT32_MAX:,} non-zeros, "
                f"but this matrix has {self.nnz:,}"
            )
        from cupyx.scipy.sparse import csr_matrix  # type: ignore[import-not-found]

        def as_cupy(a: Any) -> Any:
            if _array_mod(a) == "cupy":
                return a
            if _array_mod(a) == "torch":
                return cp.from_dlpack(a)
            return cp.asarray(a)

        # cupyx casts int64 indices to int32; the check above makes that safe.
        data = as_cupy(self.data).astype(dtype, copy=False)
        return csr_matrix(
            (data, as_cupy(self.indices), as_cupy(self.indptr)), shape=self.shape
        )

    def to_torch_sparse(self, dtype: Any = None) -> Any:
        """Wrap as a ``torch.sparse_csr_tensor``."""
        import torch  # type: ignore[import-not-found]

        def as_torch(a: Any) -> Any:
            if _array_mod(a) == "torch":
                return a
            if _array_mod(a) == "cupy":
                return torch.from_dlpack(a)
            return torch.from_numpy(a)

        data = as_torch(self.data)
        if dtype is not None:
            data = data.to(dtype=dtype)
        return torch.sparse_csr_tensor(
            as_torch(self.indptr),
            as_torch(self.indices),
            data,
            size=self.shape,
            device=data.device,
            check_invariants=False,
        )


def _aggregate_one_batch(
    points: Any,
    gene_ids: Any,
    polygons: Polygons,
    num_genes: int,
    *,
    num_cells: int,
    stream: int = 0,
) -> CSRMatrix:
    """Aggregate one device-resident batch into a CSR matrix."""
    points_xy = _as_2d_xy(points)
    if isinstance(points_xy, np.ndarray):
        import cupy as cp

        points_xy = cp.asarray(points_xy, dtype=polygons.points_xy.dtype, order="C")
        gene_ids = cp.asarray(gene_ids, order="C")
    N = int(points_xy.shape[0])

    if N > 2**31 - 1:
        raise OverflowError("aggregate_to_cells supports at most 2^31-1 points")
    if N:
        max_gene = int(gene_ids.max().item())
        if max_gene >= num_genes:
            raise ValueError(
                f"gene_ids contains {max_gene}, but num_genes is {num_genes}"
            )

    cell_ids = assign_points(
        points_xy, polygons, out=_empty_int32_like(points_xy, N), stream=stream
    )

    indptr = _empty_int32_like(cell_ids, num_cells + 1)
    indices_scratch = _empty_int32_like(cell_ids, N)
    data_scratch = _empty_int32_like(cell_ids, N)

    K = int(
        _core.aggregate_to_cells(
            cell_ids,
            gene_ids,
            num_cells,
            indptr,
            indices_scratch,
            data_scratch,
            stream,
        )
    )

    # Copy slices so the N-sized scratch buffers can be released.
    indices = indices_scratch[:K]
    data = data_scratch[:K]
    if hasattr(indices, "copy"):
        indices = indices.copy()
        data = data.copy()
    else:  # torch
        indices = indices.clone()
        data = data.clone()

    return CSRMatrix(
        indptr=indptr,
        indices=indices,
        data=data,
        shape=(num_cells, num_genes),
    )


def _bounded_pairs(batches, coords):
    import cupy as cp

    for points, genes in batches:
        xy = _as_2d_xy(points)
        host = isinstance(xy, np.ndarray)
        if host:
            if xy.dtype.kind not in "fiu":
                raise TypeError("point coordinates must be numeric")
            if not isinstance(genes, np.ndarray) or genes.dtype != np.int32:
                raise TypeError("NumPy points require NumPy int32 gene_ids")
        else:
            if not isinstance(xy, cp.ndarray) or not isinstance(genes, cp.ndarray):
                raise TypeError("expected a pair of NumPy or CuPy arrays")
            _require_array(
                genes,
                "gene_ids",
                ndim=1,
                dtype="int32",
                module="cupy",
                device=_array_device(coords),
            )
            if not xy.flags.c_contiguous:
                raise ValueError("points must be C-contiguous")
        if genes.shape != (len(xy),):
            raise ValueError("gene_ids length must equal number of points")
        free = cp.cuda.runtime.memGetInfo()[0]
        pool = getattr(cp.cuda.get_allocator(), "__self__", None)
        if isinstance(pool, cp.cuda.MemoryPool):
            free += pool.free_bytes()
            if limit := pool.get_limit():
                free = min(free, limit - pool.used_bytes())
        rows = max(1, min(_INT32_MAX, free // 128))
        budget = free if isinstance(pool, cp.cuda.MemoryPool) else 0
        if host:
            rows = min(rows, 1 << 20)
        for start in range(0, max(1, len(xy)), rows):
            yield xy[start : start + rows], genes[start : start + rows], budget


def _host_csr(shape, first, replay):
    """Plan exact row/column positions, then fill their counts in host memory."""
    plan = _core.HostCSRPlan(*shape)
    saved = []
    for cells, genes, values in first:
        plan.mark(cells, genes)
        if replay is None:
            saved.append((cells, genes, values))
    nnz = plan.nnz()
    dtype = np.int64 if nnz > _INT32_MAX else np.int32
    counts = CSRMatrix(
        np.empty(shape[0] + 1, dtype),
        np.empty(nnz, dtype),
        np.zeros(nnz, np.int32),
        shape,
    )
    plan.layout(counts.indptr, counts.indices)
    for cells, genes, values in saved if replay is None else replay():
        plan.add(cells, genes, counts.data, values)
    if replay is not None and nnz and counts.data.min() == 0:
        raise ValueError("observed cell/gene pairs changed between aggregation passes")
    return counts


def _count_records(csr):
    """Download bounded, already coalesced (cell, gene, count) triples."""
    ptr = csr.indptr.get()
    for start in range(0, csr.nnz, 1 << 20):
        end = min(start + (1 << 20), csr.nnz)
        cells = (
            np.searchsorted(ptr, np.arange(start, end, dtype=np.int64), side="right")
            - 1
        )
        yield (
            cells.astype(np.int32),
            csr.indices[start:end].get().astype(np.int32, copy=False),
            csr.data[start:end].get(),
        )


def aggregate_to_cells(
    batches: Iterable[Any],
    polygons: Polygons,
    num_genes: int | None = None,
    *,
    gene_column: str = "feature_name",
    num_cells: int | None = None,
    stream: int = 0,
) -> CSRMatrix:
    """Aggregate transcript batches into one ``(cells × genes)`` CSR.

    Each batch is a ``(points, gene_ids)`` pair of NumPy or CuPy arrays. Batches are
    sized for GPU memory. A points frame supplies ``num_genes`` and ``gene_names``.
    The returned CSR has sorted, duplicate-free ``int32`` data.
    ``indptr`` and ``indices`` are ``int32``, or ``int64`` once the result has
    more than ``2**31 - 1`` non-zeros.

    Parameters
    ----------
    batches
        Iterable of NumPy or CuPy ``(points, int32 gene_ids)`` pairs. The iterable
        is consumed one batch at a time; earlier point and gene-id arrays are
        not retained after their sparse contribution is merged.
        Also accepts a SpatialData or pandas points frame, or an Arrow Dataset
        with x/y (or x_location/y_location) and a string gene column.
    polygons
        Cell polygons shared by every batch.
    num_genes
        Number of gene columns in the output matrix.
    gene_column
        Frame column whose values identify genes.
    num_cells
        Output row count. Defaults to ``polygons.num_polygons``.
    stream
        CUDA stream handle for transfers, kernels, and merges. Defaults to stream 0.

    Returns
    -------
    CSRMatrix
        NumPy batches, points frames and Arrow datasets return a NumPy-backed
        CSR. CuPy batches return a CuPy-backed CSR when the reduction fits in
        GPU memory and a NumPy-backed CSR when it must finish in host memory.

    Raises
    ------
    OverflowError
        If a ``(cell, gene)`` count exceeds ``2**31 - 1``, the limit of the
        ``int32`` ``data`` array.

    Notes
    -----
    Batches are merged with exact integer arithmetic, so counts are never
    rounded.
    """
    import cupy as cp

    coords = polygons.points_xy
    if not isinstance(coords, cp.ndarray):
        raise TypeError("aggregate_to_cells requires CuPy polygons")
    _require_stream(stream)
    names = replay = None
    frame = hasattr(batches, "iloc") or hasattr(batches, "get_partition")
    if frame or callable(getattr(batches, "scanner", None)):
        from ..io._spatialdata import _transcript_source

        if frame and gene_column not in batches.columns:
            raise ValueError(f"points must contain gene column {gene_column!r}")
        replay, names = _transcript_source(batches, gene_column, coords.dtype)
        if num_genes is not None and num_genes != len(names):
            raise ValueError("num_genes differs from the table's gene vocabulary")
        num_genes = len(names)
        source = replay()
    else:
        source = iter(batches)
        # An iterable can yield a fresh iterator over the same one-shot cursor.
        if isinstance(batches, Sequence):
            replay = partial(iter, batches)

    if (
        isinstance(num_genes, bool)
        or not isinstance(num_genes, int)
        or num_genes < 0
        or num_genes > 2**31 - 1
    ):
        raise ValueError("num_genes must be a non-negative int32-sized integer")
    if num_cells is None:
        num_cells = polygons.num_polygons
    if isinstance(num_cells, bool) or not isinstance(num_cells, int):
        raise TypeError("num_cells must be an integer")
    if num_cells < polygons.num_polygons:
        raise ValueError("num_cells cannot be smaller than polygons.num_polygons")
    if num_cells > 2**31 - 1:
        raise ValueError("num_cells must fit in int32")

    def aggregate(points, ids):
        return _aggregate_one_batch(
            points, ids, polygons, num_genes, num_cells=num_cells, stream=stream
        )

    with (
        coords.device,
        cp.cuda.ExternalStream(stream) if stream else cp.cuda.Stream.null,
    ):
        # Keep the first batch for iterables that can only be consumed once.
        first = next(source, None)
        if first is None:
            if names is None:
                raise ValueError("batches must contain at least one transcript batch")
            first = (np.empty((0, 2), coords.dtype), np.empty(0, np.int32))
        bounded = _bounded_pairs(chain((first,), source), coords)
        host_result = isinstance(first[0], np.ndarray)
        del first
        shape = (num_cells, num_genes)

        prior = accumulator = None
        rows = 0
        for batch in chain(bounded, (None,)):
            chunk = None
            try:
                if batch is not None:
                    points, ids, free = batch
                    if accumulator is not None:
                        csr_bytes = 12 * (prior or accumulator).nnz + 16 * (
                            num_cells + 1
                        )
                        capacity = min(1 << 25, _INT32_MAX, (free - csr_bytes) // 64)
                        if rows + len(points) > capacity:
                            prior = _merge_csr(prior, accumulator, stream=stream)
                            accumulator, rows = None, 0
                    chunk = aggregate(points, ids)
                    accumulator = _merge_csr(accumulator, chunk, stream=stream)
                    del chunk
                    rows += len(points)
                    continue
                counts = _merge_csr(prior, accumulator, stream=stream)
            except (MemoryError, RuntimeError) as error:
                if not _oom(error):
                    raise
                break
            if host_result:
                return CSRMatrix(
                    counts.indptr.get(),
                    counts.indices.get(),
                    counts.data.get(),
                    shape,
                    names,
                )
            return counts

        def labels(points, ids):
            if isinstance(points, np.ndarray):
                points = cp.asarray(points, dtype=coords.dtype, order="C")
            cells = assign_points(points, polygons, stream=stream)
            if isinstance(cells, np.ndarray):
                return cells, np.ascontiguousarray(cp.asnumpy(ids)), None
            # Group writes to the same host CSR rows while the labels are on the GPU.
            order = cells.argsort()
            return cells[order].get(), cp.asarray(ids)[order].get(), None

        def parts(bounded, operation):
            for points, ids, _ in bounded:
                try:
                    partial = operation(points, ids)
                except (MemoryError, RuntimeError) as error:
                    if not _oom(error) or len(points) <= 1:
                        raise
                else:
                    yield partial
                    del partial
                    continue
                mid = len(points) // 2
                yield from parts(
                    ((points[:mid], ids[:mid], 0), (points[mid:], ids[mid:], 0)),
                    operation,
                )

        def repeat():
            return parts(_bounded_pairs(replay(), coords), labels)

        # Transfer completed reductions before the next GPU batch. The failing
        # operation did not change its inputs; retry only an unfinished batch.
        if batch is not None and chunk is None:
            bounded = chain((batch,), bounded)
        retained = [p for p in (prior, accumulator, chunk) if p is not None]
        prior = accumulator = chunk = batch = None

        def remaining():
            while retained:
                yield from _count_records(retained.pop())
            if replay is not None:
                yield from parts(bounded, labels)
            else:
                for block in parts(bounded, aggregate):
                    yield from _count_records(block)

        counts = _host_csr(shape, remaining(), repeat if replay else None)
    counts.gene_names = names
    return counts


def _index_dtype(nnz: int, current: Any) -> Any:
    """Index dtype for a merged CSR: int32 while it fits, int64 after."""
    import cupy as cp  # type: ignore[import-not-found]

    if current == cp.int64 or nnz > _INT32_MAX:
        return cp.int64
    return cp.int32


def _merge_csr(a: CSRMatrix | None, b: CSRMatrix, *, stream: int = 0) -> CSRMatrix:
    """Return the exact sum of two canonical CuPy CSR matrices.

    ``a`` is the accumulator, with ``int32`` or ``int64`` indices; ``b`` is a
    batch, with ``int32`` indices. Both must have sorted, duplicate-free indices
    and non-negative ``int32`` counts. The result has the same form, with
    ``int64`` indices once it has more than ``2**31 - 1`` non-zeros. Raises
    ``OverflowError`` if a count exceeds ``2**31 - 1``.
    """
    import cupy as cp  # type: ignore[import-not-found]

    if a is None:
        return b
    if a.shape != b.shape:
        raise ValueError(f"cannot merge CSR shapes {a.shape} and {b.shape}")

    # One bit per B entry plus one popcount per 32 entries; see the kernels.
    num_words = (b.nnz + 31) // 32
    rank_scratch = cp.empty(2 * num_words + 1, dtype=cp.int32)
    nnz = int(
        _core.merge_csr_count(
            a.indptr, a.indices, b.indptr, b.indices, rank_scratch, stream
        )
    )
    index_dtype = _index_dtype(nnz, a.indptr.dtype)

    indptr = cp.empty(a.shape[0] + 1, dtype=index_dtype)
    indices = cp.empty(nnz, dtype=index_dtype)
    data = cp.empty(nnz, dtype=cp.int32)
    _core.merge_csr_fill(
        a.indptr,
        a.indices,
        a.data,
        b.indptr,
        b.indices,
        b.data,
        rank_scratch,
        indptr,
        indices,
        data,
        stream,
    )
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=a.shape)
