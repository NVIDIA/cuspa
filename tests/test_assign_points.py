# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

cp = pytest.importorskip("cupy")
import numpy as np

import cuspa as cs
from cuspa import _core


def _square(cx: float, cy: float, r: float, dtype=cp.float32):
    xs = cp.array([cx - r, cx + r, cx + r, cx - r, cx - r], dtype=dtype)
    ys = cp.array([cy - r, cy - r, cy + r, cy + r, cy - r], dtype=dtype)
    return cp.stack([xs, ys], axis=1)


def _polys_from_squares(squares):
    poly_pts = cp.concatenate(squares, axis=0)
    ring_offsets = cp.asarray([i * 5 for i in range(len(squares) + 1)], dtype=cp.int32)
    part_offsets = cp.arange(len(squares) + 1, dtype=cp.int32)
    return cs.Polygons(part_offsets, ring_offsets, poly_pts)


def test_aabbs_match_numpy():
    polys = _polys_from_squares([_square(0, 0, 1.5), _square(10, 10, 2.0)])
    aabbs = polys.ensure_aabbs()
    expected = cp.asarray(
        [[-1.5, -1.5, 1.5, 1.5], [8.0, 8.0, 12.0, 12.0]], dtype=cp.float32
    )
    assert cp.allclose(aabbs, expected)


def test_index_builds_and_is_cached():
    polys = _polys_from_squares(
        [_square(0, 0, 1), _square(5, 5, 1), _square(10, 10, 1)]
    )
    idx1 = polys.ensure_index()
    idx2 = polys.ensure_index()
    assert idx1 is idx2  # cached
    assert idx1.num_cells == idx1.nx * idx1.ny
    go = cp.asnumpy(idx1.grid_offsets)
    assert np.all(np.diff(go) >= 0)
    assert int(go[-1]) == int(idx1.poly_ids.shape[0])


def test_assign_exact_containment():
    polys = _polys_from_squares([_square(0, 0, 1), _square(10, 10, 1)])
    points = cp.asarray(
        [
            [0.0, 0.0],  # poly 0
            [10.0, 10.0],  # poly 1
            [5.0, 5.0],  # neither
            [0.5, -0.5],  # poly 0
            [-5.0, 0.0],  # neither (inside grid bbox, outside polys)
            [100.0, 0.0],  # outside grid bbox -> -1
        ],
        dtype=cp.float32,
    )
    ids = cs.tl.assign_points(points, polys)
    assert ids.dtype == cp.int32
    assert ids.tolist() == [0, 1, -1, 0, -1, -1]


def test_hole_edge_extension_is_not_treated_as_boundary():
    outer = cp.asarray([[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]], dtype=cp.float32)
    hole = cp.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=cp.float32)
    polys = cs.Polygons(
        cp.asarray([0, 2], dtype=cp.int32),
        cp.asarray([0, 5, 10], dtype=cp.int32),
        cp.concatenate([outer, hole]),
    )

    # This point is inside the shell and collinear with the hole's right edge,
    # but it is not on that finite edge segment.
    point = cp.asarray([[1.0, 3.0]], dtype=cp.float32)
    assert cs.tl.assign_points(point, polys).tolist() == [0]


def test_polygon_metadata_is_validated_before_kernel_launch():
    with pytest.raises(ValueError, match=r"ring_offsets\[-1\]"):
        cs.Polygons(
            cp.asarray([0, 1], dtype=cp.int32),
            cp.asarray([0, 4], dtype=cp.int32),
            cp.zeros((5, 2), dtype=cp.float32),
        )


def test_native_binding_rejects_cpu_memory():
    with pytest.raises(TypeError, match="incompatible function arguments"):
        _core.compute_poly_aabbs(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 5], dtype=np.int32),
            np.zeros((5, 2), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
        )


@pytest.mark.parametrize(
    "name",
    [
        "compute_poly_aabbs",
        "compute_index_size",
        "build_index",
        "assign_points",
        "overlap_count_and_scan",
        "overlap_emit",
        "aggregate_to_cells",
    ],
)
def test_native_binding_exposes_cuda_and_managed_overloads(name):
    signature = getattr(_core, name).__doc__
    assert "device='cuda'" in signature
    assert "device='cuda_managed'" in signature


def test_explicit_grid_shape_rebuilds_cached_index():
    polys = _polys_from_squares([_square(0, 0, 1), _square(10, 10, 1)])
    first = polys.ensure_index(nx=1, ny=1)
    second = polys.ensure_index(nx=2, ny=3)
    assert first is not second
    assert (second.nx, second.ny) == (2, 3)


def test_torch_cuda_arrays_are_supported_directly():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch is required")

    polys = cs.Polygons(
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 5], dtype=torch.int32, device="cuda"),
        torch.tensor(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]],
            dtype=torch.float32,
            device="cuda",
        ),
    )
    points = torch.tensor([[0, 0], [2, 2]], dtype=torch.float32, device="cuda")

    assert cs.tl.assign_points(points, polys).cpu().tolist() == [0, -1]


def test_rmm_managed_cupy_allocator():
    pytest.importorskip("rmm")

    # RMM reinitialization invalidates existing RMM allocations, and CuPy's
    # allocator is process-global. Keep this interoperability check isolated
    # from the rest of the test process.
    script = textwrap.dedent(
        """
        import cupy as cp
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        import cuspa as cs

        rmm.reinitialize(pool_allocator=True, managed_memory=True)
        cp.cuda.set_allocator(rmm_cupy_allocator)

        polygons = cs.Polygons(
            points_xy=cp.asarray(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                dtype=cp.float32,
            ),
            ring_offsets=cp.asarray([0, 5], dtype=cp.int32),
            part_offsets=cp.asarray([0, 1], dtype=cp.int32),
        )
        points = cp.asarray([[0.5, 0.5], [2, 2]], dtype=cp.float32)

        labels = cs.tl.assign_points(points, polygons)
        pairs = cs.tl.overlap_pairs(points, polygons)
        csr = cs.tl.aggregate_to_cells(
            [(points, cp.asarray([1, 0], dtype=cp.int32))],
            polygons,
            num_genes=2,
        )
        cp.cuda.Stream.null.synchronize()
        assert labels.tolist() == [0, -1]
        assert pairs.tolist() == [[0, 0]]
        assert csr.indptr.tolist() == [0, 1]
        assert csr.indices.tolist() == [1]
        assert csr.data.tolist() == [1]
        assert polygons.aabbs.__dlpack_device__()[0] == 13
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_non_default_cuda_stream():
    polys = _polys_from_squares([_square(0, 0, 1)])
    points = cp.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=cp.float32)
    stream = cp.cuda.Stream(non_blocking=True)

    ids = cs.tl.assign_points(points, polys, stream=stream.ptr)
    stream.synchronize()

    assert ids.tolist() == [0, -1]


def test_assign_1000_polygons_random_grid():
    rng = np.random.default_rng(42)
    cx = np.tile(np.arange(30), 30).astype(np.float32)
    cy = np.repeat(np.arange(30), 30).astype(np.float32)
    r = 0.4

    squares = [_square(float(x), float(y), r) for x, y in zip(cx, cy)]
    polys = _polys_from_squares(squares)
    P = polys.num_polygons
    assert P == 900

    n = 50_000
    pts_host = rng.uniform(-2, 32, size=(n, 2)).astype(np.float32)
    pts = cp.asarray(pts_host)
    ids = cp.asnumpy(cs.tl.assign_points(pts, polys))

    # Reference: for each point, find the square whose AABB contains it
    # (squares are disjoint, so AABB == membership).
    expected = np.full(n, -1, dtype=np.int32)
    # Grid cells' integer centers → infer idx from rounding
    for i in range(n):
        x, y = pts_host[i]
        ix = round(x)
        iy = round(y)
        if 0 <= ix < 30 and 0 <= iy < 30 and abs(x - ix) <= r and abs(y - iy) <= r:
            expected[i] = iy * 30 + ix

    assert np.array_equal(ids, expected)


def test_assign_large_scale():
    rng = np.random.default_rng(0)
    P = 5_000
    side = int(np.sqrt(P))
    centers = (
        np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1)
        .reshape(-1, 2)[:P]
        .astype(np.float32)
    )
    centers += rng.uniform(-0.1, 0.1, size=centers.shape).astype(np.float32)
    r = 0.35

    squares = [_square(float(c[0]), float(c[1]), r) for c in centers]
    polys = _polys_from_squares(squares)

    n = 1_000_000
    pts = cp.asarray(rng.uniform(-2, side + 2, size=(n, 2)).astype(np.float32))

    ids = cs.tl.assign_points(pts, polys)
    cp.cuda.Stream.null.synchronize()

    ids_host = cp.asnumpy(ids)
    assert (ids_host >= -1).all()
    assert (ids_host < P).all()
    hit_frac = (ids_host >= 0).mean()
    assert 0.05 < hit_frac < 0.9, f"unexpected hit fraction: {hit_frac}"
