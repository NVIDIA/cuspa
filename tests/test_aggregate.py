# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")
import numpy as np

import cuspa as cs


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
