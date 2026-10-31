# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import _core
from .._types import (
    Polygons,
    _array_device,
    _array_mod,
    _require_array,
)
from ._assign import _as_2d_xy, assign_points

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


def _empty_int32_like(ref: Any, n: int) -> Any:
    mod = _array_mod(ref)
    if mod == "cupy":
        import cupy as xp  # type: ignore[import-not-found]

        return xp.empty(n, dtype=xp.int32)
    if mod == "torch":
        import torch  # type: ignore[import-not-found]

        return torch.empty(n, dtype=torch.int32, device=ref.device)
    raise TypeError(f"Cannot auto-allocate int32 buffer for {type(ref)!r}")


def _aggregate_one_batch(
    points: Any,
    gene_ids: Any,
    polygons: Polygons,
    num_genes: int,
    *,
    num_cells: int | None = None,
    stream: int = 0,
) -> CSRMatrix:
    """Aggregate one device-resident batch into a CSR matrix."""
    points_xy = _as_2d_xy(points)
    N = int(points_xy.shape[0])

    if (
        isinstance(num_genes, bool)
        or not isinstance(num_genes, int)
        or num_genes <= 0
        or num_genes > 2**31 - 1
    ):
        raise ValueError("num_genes must be a positive int32-sized integer")
    if N > 2**31 - 1:
        raise OverflowError("aggregate_to_cells supports at most 2^31-1 points")
    _require_array(
        gene_ids,
        "gene_ids",
        ndim=1,
        dtype="int32",
        module=_array_mod(points_xy),
        device=_array_device(points_xy),
    )

    if int(gene_ids.shape[0]) != N:
        raise ValueError(
            f"gene_ids length ({int(gene_ids.shape[0])}) must equal number of "
            f"points ({N})"
        )

    if num_cells is None:
        num_cells = polygons.num_polygons
    if isinstance(num_cells, bool) or not isinstance(num_cells, int):
        raise TypeError("num_cells must be an integer")
    if num_cells < polygons.num_polygons:
        raise ValueError("num_cells cannot be smaller than polygons.num_polygons")
    if num_cells > 2**31 - 1:
        raise ValueError("num_cells must fit in int32")
    if N:
        max_gene = int(gene_ids.max().item())
        if max_gene >= num_genes:
            raise ValueError(
                f"gene_ids contains {max_gene}, but num_genes is {num_genes}"
            )

    cell_ids = assign_points(points_xy, polygons, stream=stream)

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


def aggregate_to_cells(
    batches: Iterable[tuple[Any, Any]],
    polygons: Polygons,
    num_genes: int,
    *,
    num_cells: int | None = None,
    stream: int = 0,
) -> CSRMatrix:
    """Aggregate transcript batches into one GPU-resident ``(cells × genes)`` CSR.

    Each batch is a ``(points, gene_ids)`` pair of CuPy arrays. Batches are
    consumed one at a time; this function does not read files or choose a batch
    size. The returned CSR has sorted, duplicate-free ``int32`` data.
    ``indptr`` and ``indices`` are ``int32``, or ``int64`` once the result has
    more than ``2**31 - 1`` non-zeros.

    Parameters
    ----------
    batches
        Iterable of ``(points, gene_ids)`` CuPy CUDA-array pairs. The iterable
        is consumed one batch at a time; earlier point and gene-id arrays are
        not retained after their sparse contribution is merged.
    polygons
        Cell polygons shared by every batch.
    num_genes
        Number of gene columns in the output matrix.
    num_cells
        Output row count. Defaults to ``polygons.num_polygons``.
    stream
        CUDA stream handle for native kernels. ``0`` means the default stream.
        The corresponding CuPy stream must also be current because buffer
        copies and merge buffers use CuPy's current stream.

    Returns
    -------
    CSRMatrix
        Sparse count matrix across all batches, stored on the GPU.

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
    iterator = iter(batches)
    try:
        points, gene_ids = next(iterator)
    except StopIteration as exc:
        raise ValueError("batches must contain at least one transcript batch") from exc

    accumulator = _aggregate_one_batch(
        points,
        gene_ids,
        polygons,
        num_genes,
        num_cells=num_cells,
        stream=stream,
    )
    if _array_mod(accumulator.data) != "cupy":
        raise TypeError("aggregate_to_cells currently requires CuPy batches")

    for points, gene_ids in iterator:
        partial = _aggregate_one_batch(
            points,
            gene_ids,
            polygons,
            num_genes,
            num_cells=num_cells,
            stream=stream,
        )
        if _array_mod(partial.data) != "cupy":
            raise TypeError("all aggregate_to_cells batches must be CuPy arrays")
        updated = _merge_csr(accumulator, partial, stream=stream)
        del accumulator, partial
        accumulator = updated

    return accumulator


def _index_dtype(nnz: int, current: Any) -> Any:
    """Index dtype for a merged CSR: int32 while it fits, int64 after."""
    import cupy as cp  # type: ignore[import-not-found]

    if current == cp.int64 or nnz > _INT32_MAX:
        return cp.int64
    return cp.int32


def _merge_csr(a: CSRMatrix, b: CSRMatrix, *, stream: int = 0) -> CSRMatrix:
    """Return the exact sum of two canonical CuPy CSR matrices.

    ``a`` is the accumulator, with ``int32`` or ``int64`` indices; ``b`` is a
    batch, with ``int32`` indices. Both must have sorted, duplicate-free indices
    and non-negative ``int32`` counts. The result has the same form, with
    ``int64`` indices once it has more than ``2**31 - 1`` non-zeros. Raises
    ``OverflowError`` if a count exceeds ``2**31 - 1``.
    """
    import cupy as cp  # type: ignore[import-not-found]

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
