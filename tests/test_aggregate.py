# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")
import numpy as np

import cuspa as cs
from cuspa.tl import _aggregate as aggregate_mod
from cuspa.tl._aggregate import CSRMatrix, _merge_csr


def _square(cx: float, cy: float, r: float, dtype=cp.float32):
    xs = cp.array([cx - r, cx + r, cx + r, cx - r, cx - r], dtype=dtype)
    ys = cp.array([cy - r, cy - r, cy + r, cy + r, cy - r], dtype=dtype)
    return cp.stack([xs, ys], axis=1)


def _polys_from_squares(squares):
    poly_pts = cp.concatenate(squares, axis=0)
    ring_offsets = cp.asarray([i * 5 for i in range(len(squares) + 1)], dtype=cp.int32)
    part_offsets = cp.arange(len(squares) + 1, dtype=cp.int32)
    return cs.Polygons(part_offsets, ring_offsets, poly_pts)


def _aggregate(points, gene_ids, polygons, num_genes, **kwargs):
    return cs.tl.aggregate_to_cells([(points, gene_ids)], polygons, num_genes, **kwargs)


def test_aggregate_basic():
    polys = _polys_from_squares([_square(0, 0, 1), _square(10, 10, 1)])
    points = cp.asarray(
        [
            [0.0, 0.0],  # cell 0, gene 0
            [0.5, 0.5],  # cell 0, gene 0
            [-0.5, -0.5],  # cell 0, gene 1
            [10.0, 10.0],  # cell 1, gene 2
            [5.0, 5.0],  # unassigned
        ],
        dtype=cp.float32,
    )
    gene_ids = cp.asarray([0, 0, 1, 2, 0], dtype=cp.int32)

    csr = _aggregate(points, gene_ids, polys, num_genes=3)

    assert csr.shape == (2, 3)
    assert csr.nnz == 3  # (cell0,gene0)=2, (cell0,gene1)=1, (cell1,gene2)=1

    indptr = cp.asnumpy(csr.indptr).tolist()
    indices = cp.asnumpy(csr.indices).tolist()
    data = cp.asnumpy(csr.data).tolist()
    assert indptr == [0, 2, 3]  # cell 0 has 2 unique genes, cell 1 has 1
    assert indices == [0, 1, 2]
    assert data == [2, 1, 1]


def test_aggregate_wraps_to_cupyx_csr():
    polys = _polys_from_squares([_square(0, 0, 1)])
    points = cp.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=cp.float32)
    gene_ids = cp.asarray([3, 3], dtype=cp.int32)
    csr = _aggregate(points, gene_ids, polys, num_genes=5)

    mat = csr.to_cupy_csr()
    assert mat.shape == (1, 5)
    dense = cp.asnumpy(mat.toarray())
    assert dense.tolist() == [[0, 0, 0, 2, 0]]


def test_aggregate_to_cells_accumulates_batches_on_device():
    polys = _polys_from_squares([_square(0, 0, 1), _square(10, 10, 1)])
    points = cp.asarray(
        [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [10.0, 10.0], [5.0, 5.0]],
        dtype=cp.float32,
    )
    gene_ids = cp.asarray([0, 0, 1, 2, 0], dtype=cp.int32)

    batches = [(points[:2], gene_ids[:2]), (points[2:], gene_ids[2:])]
    streamed = cs.tl.aggregate_to_cells(batches, polys, num_genes=3)
    expected = _aggregate(points, gene_ids, polys, num_genes=3)

    assert type(streamed.data).__module__.split(".", 1)[0] == "cupy"
    assert streamed.to_cupy_csr().has_canonical_format
    assert cp.array_equal(
        streamed.to_cupy_csr().toarray(), expected.to_cupy_csr().toarray()
    )


def test_aggregate_matches_scipy_reference():
    from scipy.sparse import csr_matrix as sp_csr

    rng = np.random.default_rng(7)
    P = 400
    side = 20  # 20x20 grid of disjoint squares
    centers = (
        np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1)
        .reshape(-1, 2)[:P]
        .astype(np.float32)
    )
    r = 0.4

    squares = [_square(float(c[0]), float(c[1]), r) for c in centers]
    polys = _polys_from_squares(squares)

    N = 50_000
    G = 200
    pts_host = rng.uniform(-1, side + 1, size=(N, 2)).astype(np.float32)
    genes_host = rng.integers(0, G, size=N).astype(np.int32)

    pts = cp.asarray(pts_host)
    genes = cp.asarray(genes_host)
    gpu_csr = _aggregate(pts, genes, polys, num_genes=G)
    gpu_dense = cp.asnumpy(gpu_csr.to_cupy_csr().toarray())

    # CPU reference: find (cell, gene) by simple AABB (squares are disjoint)
    cell_ids = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        x, y = pts_host[i]
        ix = round(x)
        iy = round(y)
        if 0 <= ix < side and 0 <= iy < side and abs(x - ix) <= r and abs(y - iy) <= r:
            cell_ids[i] = iy * side + ix
    mask = cell_ids >= 0
    ref = sp_csr(
        (np.ones(mask.sum(), dtype=np.int32), (cell_ids[mask], genes_host[mask])),
        shape=(P, G),
    )
    ref_dense = ref.toarray()

    assert np.array_equal(gpu_dense, ref_dense)


def _csr(indptr, indices, data, shape):
    return CSRMatrix(
        indptr=cp.asarray(indptr, dtype=cp.int32),
        indices=cp.asarray(indices, dtype=cp.int32),
        data=cp.asarray(data, dtype=cp.int32),
        shape=shape,
    )


def _random_canonical_csr(rng, shape, density):
    from scipy.sparse import random as sp_random

    mat = sp_random(*shape, density=density, format="csr", random_state=rng)
    mat.data = rng.integers(1, 1000, size=mat.nnz).astype(np.int32)
    mat.sort_indices()
    return mat, _csr(mat.indptr, mat.indices, mat.data, shape)


def _assert_csr_equal(csr, ref, index_dtype=cp.int32):
    assert csr.shape == ref.shape
    assert cp.asnumpy(csr.indptr).tolist() == ref.indptr.tolist()
    assert cp.asnumpy(csr.indices).tolist() == ref.indices.tolist()
    assert cp.asnumpy(csr.data).tolist() == ref.data.tolist()
    assert csr.indptr.dtype == csr.indices.dtype == index_dtype
    assert csr.data.dtype == cp.int32


def test_merge_csr_is_exact_past_float32():
    a = _csr([0, 1], [0], [2**24 - 1], (1, 1))
    b = _csr([0, 1], [0], [2], (1, 1))
    merged = _merge_csr(a, b)
    # float32 cannot represent 2**24 + 1.
    assert cp.asnumpy(merged.data).tolist() == [2**24 + 1]


def test_merge_csr_matches_scipy_with_empty_rows_and_inputs():
    rng = np.random.default_rng(3)
    shape = (300, 50)
    for density_a, density_b in [(0.1, 0.1), (0.0, 0.2), (0.2, 0.0), (0.0, 0.0)]:
        ref_a, a = _random_canonical_csr(rng, shape, density_a)
        ref_b, b = _random_canonical_csr(rng, shape, density_b)
        ref = (ref_a + ref_b).tocsr()
        ref.sort_indices()
        _assert_csr_equal(_merge_csr(a, b), ref)


def test_merge_csr_raises_on_count_overflow():
    a = _csr([0, 1], [0], [2**31 - 1], (1, 1))
    b = _csr([0, 1], [0], [1], (1, 1))
    with pytest.raises(OverflowError, match="2\\^31-1"):
        _merge_csr(a, b)
    # The largest representable count still merges.
    b = _csr([0, 1], [0], [0], (1, 1))
    assert cp.asnumpy(_merge_csr(a, b).data).tolist() == [2**31 - 1]


def test_merge_csr_switches_to_int64_indices(monkeypatch):
    # Lower the int32 limit so small inputs take the int64 paths.
    monkeypatch.setattr(aggregate_mod, "_INT32_MAX", 100)
    rng = np.random.default_rng(5)
    shape = (40, 30)
    ref, acc = _random_canonical_csr(rng, shape, 0.05)
    assert acc.nnz <= 100
    for step in range(4):
        ref_b, b = _random_canonical_csr(rng, shape, 0.1)
        ref = (ref + ref_b).tocsr()
        ref.sort_indices()
        acc = _merge_csr(acc, b)  # int32 -> int64, then int64 -> int64
        assert ref.nnz > 100, step
        _assert_csr_equal(acc, ref, index_dtype=cp.int64)


def test_aggregate_to_cells_returns_int64_indices_past_int32(monkeypatch):
    monkeypatch.setattr(aggregate_mod, "_INT32_MAX", 10)
    polys = _polys_from_squares([_square(float(x), 0, 0.4) for x in range(8)])
    rng = np.random.default_rng(2)
    xs = rng.integers(0, 8, size=500).astype(np.float32)
    genes = rng.integers(0, 6, size=500).astype(np.int32)
    pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    batches = [
        (cp.asarray(pts[i : i + 100]), cp.asarray(genes[i : i + 100]))
        for i in range(0, 500, 100)
    ]
    csr = cs.tl.aggregate_to_cells(batches, polys, num_genes=6)

    from scipy.sparse import csr_matrix as sp_csr

    ref = sp_csr(
        (np.ones(500, dtype=np.int32), (xs.astype(np.int64), genes)), shape=(8, 6)
    )
    ref.sum_duplicates()
    ref.sort_indices()
    _assert_csr_equal(csr, ref, index_dtype=cp.int64)
    with pytest.raises(OverflowError, match="cupyx"):
        csr.to_cupy_csr()


def test_merge_csr_past_int32_nnz_on_device():
    """A real merge past 2^31-1 non-zeros; needs a large GPU, so CI often skips."""
    if cp.cuda.Device().mem_info[0] < 48 * 2**30:
        pytest.skip("needs about 48 GiB of free GPU memory")
    rows, genes = 2**21, 2**10  # rows * genes == 2**31
    nnz_a = rows * genes - 1  # the last (row, gene) is missing
    indptr = cp.minimum(cp.arange(rows + 1, dtype=cp.int64) * genes, nnz_a)
    a = CSRMatrix(
        indptr=indptr.astype(cp.int32),
        indices=cp.tile(cp.arange(genes, dtype=cp.int32), rows)[:nnz_a],
        data=cp.ones(nnz_a, dtype=cp.int32),
        shape=(rows, genes),
    )
    del indptr
    # B adds the missing entry and bumps the first one.
    b_indptr = cp.zeros(rows + 1, dtype=cp.int32)
    b_indptr[1:] = 1
    b_indptr[-1] = 2
    b = _csr(b_indptr, [0, genes - 1], [5, 7], (rows, genes))

    merged = _merge_csr(a, b)
    del a
    assert merged.nnz == 2**31
    assert merged.indptr.dtype == merged.indices.dtype == cp.int64
    assert int(merged.indptr[-1]) == 2**31
    assert int(merged.indptr[-2]) == 2**31 - genes
    assert int(merged.indices[-1]) == genes - 1
    assert cp.asnumpy(merged.data[[0, 1, -2, -1]]).tolist() == [6, 1, 1, 7]
    assert int(merged.data.sum(dtype=cp.int64)) == 2**31 - 1 + 12

    # int64 accumulator + int32 batch.
    again = _merge_csr(merged, b)
    del merged
    assert again.nnz == 2**31
    assert cp.asnumpy(again.data[[0, -1]]).tolist() == [11, 14]


def test_aggregate_to_cells_counts_past_float32_limit():
    polys = _polys_from_squares([_square(0, 0, 1)])
    n = 2**24
    batches = [
        (cp.zeros((n, 2), dtype=cp.float32), cp.zeros(n, dtype=cp.int32)),
        (cp.zeros((1, 2), dtype=cp.float32), cp.zeros(1, dtype=cp.int32)),
    ]
    csr = cs.tl.aggregate_to_cells(batches, polys, num_genes=1)
    assert cp.asnumpy(csr.data).tolist() == [2**24 + 1]


def test_aggregate_to_cells_many_overlapping_batches_match_scipy():
    from scipy.sparse import csr_matrix as sp_csr

    rng = np.random.default_rng(11)
    side = 10
    squares = [
        _square(float(x), float(y), 0.45) for y in range(side) for x in range(side)
    ]
    polys = _polys_from_squares(squares)

    N, G, num_batches = 40_000, 30, 13
    pts_host = rng.uniform(-0.5, side - 0.5, size=(N, 2)).astype(np.float32)
    genes_host = rng.integers(0, G, size=N).astype(np.int32)
    splits = np.sort(rng.choice(np.arange(1, N), size=num_batches - 1, replace=False))
    batches = [
        (cp.asarray(p), cp.asarray(g))
        for p, g in zip(np.split(pts_host, splits), np.split(genes_host, splits))
    ]
    csr = cs.tl.aggregate_to_cells(batches, polys, num_genes=G)

    ix = np.rint(pts_host[:, 0]).astype(np.int64)
    iy = np.rint(pts_host[:, 1]).astype(np.int64)
    inside = (np.abs(pts_host[:, 0] - ix) <= 0.45) & (
        np.abs(pts_host[:, 1] - iy) <= 0.45
    )
    cells = (iy * side + ix)[inside]
    ref = sp_csr(
        (np.ones(cells.size, dtype=np.int32), (cells, genes_host[inside])),
        shape=(side * side, G),
    )
    ref.sum_duplicates()
    ref.sort_indices()
    _assert_csr_equal(csr, ref)


def test_aggregate_all_unassigned_returns_empty():
    polys = _polys_from_squares([_square(0, 0, 0.1)])
    points = cp.asarray([[100.0, 100.0], [200.0, 200.0]], dtype=cp.float32)
    gene_ids = cp.asarray([0, 1], dtype=cp.int32)
    csr = _aggregate(points, gene_ids, polys, num_genes=3)
    assert csr.nnz == 0
    assert cp.asnumpy(csr.indptr).tolist() == [0, 0]


def test_aggregate_rejects_out_of_range_gene_ids():
    polys = _polys_from_squares([_square(0, 0, 1)])
    points = cp.asarray([[0.0, 0.0]], dtype=cp.float32)
    genes = cp.asarray([3], dtype=cp.int32)
    with pytest.raises(ValueError, match="num_genes"):
        _aggregate(points, genes, polys, num_genes=3)


def test_aggregate_rejects_too_few_rows():
    polys = _polys_from_squares([_square(0, 0, 1), _square(5, 5, 1)])
    points = cp.asarray([[0.0, 0.0]], dtype=cp.float32)
    genes = cp.asarray([0], dtype=cp.int32)
    with pytest.raises(ValueError, match="num_cells"):
        _aggregate(points, genes, polys, num_genes=1, num_cells=1)


def test_csr_converts_to_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch is required")

    polys = _polys_from_squares([_square(0, 0, 1)])
    points = cp.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=cp.float32)
    genes = cp.asarray([1, 1], dtype=cp.int32)
    csr = _aggregate(points, genes, polys, num_genes=2)

    dense = csr.to_torch_sparse(dtype=torch.float32).to_dense().cpu().tolist()
    assert dense == [[0.0, 2.0]]
