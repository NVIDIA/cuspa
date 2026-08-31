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
    _to_nanobind_cuda,
)
from ._assign import _as_2d_xy, assign_points

if TYPE_CHECKING:
    from collections.abc import Iterable


# Float32 represents every integer below this threshold exactly. We keep a
# strict boundary so an increment at the limit can never be silently rounded.
_FLOAT32_COUNT_LIMIT = 2**24


@dataclass(slots=True)
class CSRMatrix:
    """Framework-agnostic CSR container for aggregate results.

    Fields can be CuPy, CUDA PyTorch, or NumPy arrays. Use
    :meth:`to_cupy_csr` or :meth:`to_torch_sparse` to wrap for downstream.
    """

    indptr: Any  # int32[num_cells + 1]
    indices: Any  # int32[K]  gene ids in CSR column order
    data: Any  # int32[K]  counts
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    def to_cupy_csr(self, dtype: Any = "float32") -> Any:
        """Wrap as ``cupyx.scipy.sparse.csr_matrix``.

        ``cupyx.scipy.sparse`` only supports float / complex data, so counts
        are cast to ``dtype`` (default ``float32``) on the way out.
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
        """Wrap the result as a CUDA ``torch.sparse_csr_tensor``."""
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
    """Aggregate one already-resident batch into a CSR.

    This private primitive backs :func:`aggregate_to_cells`. Keeping it
    separate lets the public iterator retain only the current batch plus its
    device-resident accumulator.

    Parameters
    ----------
    points
        Device tensor of shape ``(N, 2)`` (float32 or float64).
    gene_ids
        Device tensor of shape ``(N,)`` (int32). ``gene_ids[i]`` is the gene
        index of transcript ``i``. Negative values are treated as unassigned.
    polygons
        :class:`cuspa.Polygons`. Its spatial index is built on first
        call and reused on subsequent calls.
    num_genes
        Number of gene columns in the output matrix. Determines the column
        count of the CSR.
    num_cells
        Row count of the CSR. Defaults to ``polygons.num_polygons``.
    stream
        CUDA stream handle. ``0`` means the default stream.

    Returns
    -------
    :class:`CSRMatrix` — ``indptr`` (``int32[num_cells + 1]``), ``indices``
    (``int32[K]``, gene IDs), ``data`` (``int32[K]``, counts).
    ``K`` is the number of unique ``(cell, gene)`` pairs found.

    Notes
    -----
    Transient memory is ~40 bytes per transcript during the sort, so a run
    of 1e8 transcripts needs ~4 GB free. At 1e9 you'll want a 40 GB+ GPU.
    """
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

    _nb = _to_nanobind_cuda
    K = int(
        _core.aggregate_to_cells(
            _nb(cell_ids),
            _nb(gene_ids),
            num_cells,
            _nb(indptr),
            _nb(indices_scratch),
            _nb(data_scratch),
            stream,
        )
    )

    # Slice + copy so the N-sized scratch can be released.
    indices = indices_scratch[:K]
    data = data_scratch[:K]
    # cupy/torch views keep the parent alive; copy to drop the surplus.
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

    Each item in ``batches`` is a ``(points, gene_ids)`` pair accepted by
    the private one-batch primitive. The spatial index on ``polygons`` is
    built on the first batch and reused by every later batch. The first batch
    becomes the device-resident CSR reference; later batch CSRs are added to
    it on the GPU. After the final addition, the accumulator is canonicalized
    (duplicate entries coalesced and column indices sorted), still on the
    GPU, so it can be handed directly to CuPy or CUDA PyTorch.

    This function consumes batches supplied by the caller; it does not read
    files or choose a batch size. A source reader should yield one CUDA batch
    at a time to keep source and GPU input memory bounded.

    This bounds GPU memory by the largest batch plus one final sparse output,
    rather than the total number of transcripts. The accumulator uses float32
    temporarily because CuPy sparse matrices do not support integer data. It
    is exact for per-``(cell, gene)`` counts below ``2**24``; larger counts
    raise :class:`OverflowError`. Returned arrays are CUDA int32 arrays. Copy
    them to host only when serializing or otherwise needed.

    Parameters
    ----------
    batches
        Iterable of ``(points, gene_ids)`` CuPy CUDA-array pairs. The iterable
        is consumed one batch at a time; earlier point and gene-id arrays are
        not retained after their sparse contribution is merged.
    polygons
        Cell polygons shared by every batch.
    num_genes
        Number of gene columns in each yielded CSR matrix.
    num_cells
        Row count of each matrix. Defaults to ``polygons.num_polygons``.
    stream
        CUDA stream handle. ``0`` means the default stream.

    Returns
    -------
    CSRMatrix
        The exact, canonical sparse count matrix across all batches, stored
        on the GPU.
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
