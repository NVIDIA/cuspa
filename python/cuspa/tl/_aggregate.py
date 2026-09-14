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


# Float32 represents every integer below this threshold exactly. We keep a
# strict boundary so an increment at the limit can never be silently rounded.
_FLOAT32_COUNT_LIMIT = 2**24


@dataclass(slots=True)
class CSRMatrix:
    """CSR matrix backed by CuPy, CUDA PyTorch, or NumPy arrays."""

    indptr: Any  # int32[num_cells + 1]
    indices: Any  # int32[K]  gene ids in CSR column order
    data: Any  # int32[K]  counts
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    def to_cupy_csr(self, dtype: Any = "float32") -> Any:
        """Wrap as ``cupyx.scipy.sparse.csr_matrix``.

        ``cupyx.scipy.sparse`` only supports float and complex data. Counts are
        cast to ``dtype``, which defaults to ``float32``.
        """
        import cupy as cp  # type: ignore[import-not-found]
        from cupyx.scipy.sparse import csr_matrix  # type: ignore[import-not-found]

        def as_cupy(a: Any) -> Any:
            if _array_mod(a) == "cupy":
                return a
            if _array_mod(a) == "torch":
                return cp.from_dlpack(a)
            return cp.asarray(a)

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
        copies and sparse merging use CuPy's current stream.

    Returns
    -------
    CSRMatrix
        Sparse count matrix across all batches, stored on the GPU.

    Notes
    -----
    Batch merging uses ``float32`` sparse matrices. Counts for a
    ``(cell, gene)`` pair must be below ``2**24``.
    """
    import cupy as cp  # type: ignore[import-not-found]

    iterator = iter(batches)
    try:
        points, gene_ids = next(iterator)
    except StopIteration as exc:
        raise ValueError("batches must contain at least one transcript batch") from exc

    first = _aggregate_one_batch(
        points,
        gene_ids,
        polygons,
        num_genes,
        num_cells=num_cells,
        stream=stream,
    )
    if _array_mod(first.data) != "cupy":
        raise TypeError("aggregate_to_cells currently requires CuPy batches")
    accumulator = first.to_cupy_csr(dtype="float32")
    del first

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
        updated = accumulator + partial.to_cupy_csr(dtype="float32")
        del accumulator, partial
        accumulator = updated

    # Sparse addition can leave duplicate entries and non-canonical ordering.
    # Do this once, after the final batch, rather than for every addition.
    accumulator.sum_duplicates()
    accumulator.sort_indices()
    if accumulator.nnz and int(accumulator.data.max()) >= _FLOAT32_COUNT_LIMIT:
        raise OverflowError(
            "float32 batch accumulation requires every (cell, gene) count "
            f"to be below {_FLOAT32_COUNT_LIMIT:,}"
        )

    return CSRMatrix(
        indptr=accumulator.indptr.astype(cp.int32, copy=False),
        indices=accumulator.indices.astype(cp.int32, copy=False),
        data=accumulator.data.astype(cp.int32),
        shape=accumulator.shape,
    )
