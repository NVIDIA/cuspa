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


# ---------------------------------------------------------------------------
# overlap_pairs: many-to-many
# ---------------------------------------------------------------------------


def test_overlap_pairs_disjoint_polygons_match_assign():
    """When polygons don't overlap, overlap_pairs must agree with assign_points."""
    polys = _polys_from_squares([_square(0, 0, 1), _square(10, 10, 1)])
    points = cp.asarray(
        [
            [0.0, 0.0],  # poly 0
            [10.0, 10.0],  # poly 1
            [5.0, 5.0],  # neither
            [0.5, -0.5],  # poly 0
            [100.0, 0.0],  # outside bbox
        ],
        dtype=cp.float32,
    )
    pairs = cp.asnumpy(cs.tl.overlap_pairs(points, polys))
    expected = np.array([[0, 0], [1, 1], [3, 0]], dtype=np.int32)
    assert np.array_equal(pairs, expected)


def test_overlap_pairs_overlapping_polygons_multi_assign():
    """A point inside two overlapping polygons must produce two pairs."""
    # Two squares sharing overlap at (0,0)..(1,1)
    polys = _polys_from_squares([_square(0.5, 0.5, 1.0), _square(0.5, 0.5, 0.8)])
    pt_inside_both = (0.5, 0.5)
    pt_inside_one = (1.3, 0.5)  # inside poly 0 only
    pt_outside = (5.0, 5.0)
    points = cp.asarray([pt_inside_both, pt_inside_one, pt_outside], dtype=cp.float32)
    pairs = cp.asnumpy(cs.tl.overlap_pairs(points, polys))
    # Point 0 → both polys; point 1 → poly 0 only; point 2 → nothing.
    # Order within a point follows grid-cell insertion order (by poly idx).
    assert pairs.shape == (3, 2)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 0), (0, 1), (1, 0)]


def test_overlap_pairs_empty_when_no_hits():
    polys = _polys_from_squares([_square(0, 0, 0.1)])
    points = cp.asarray([[100.0, 100.0], [200.0, 200.0]], dtype=cp.float32)
    pairs = cs.tl.overlap_pairs(points, polys)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == cp.int32


def test_overlap_pairs_accepts_empty_point_array():
    polys = _polys_from_squares([_square(0, 0, 1)])
    pairs = cs.tl.overlap_pairs(cp.empty((0, 2), dtype=cp.float32), polys)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == cp.int32


def test_overlap_pairs_counts_match_assign_on_random_data():
    """On a non-overlapping polygon grid, |overlap_pairs| == |assigned|."""
    rng = np.random.default_rng(0)
    side = 20
    cx = np.tile(np.arange(side), side).astype(np.float32)
    cy = np.repeat(np.arange(side), side).astype(np.float32)
    r = 0.4

    polys = _polys_from_squares(
        [_square(float(x), float(y), r) for x, y in zip(cx, cy)]
    )
    pts = cp.asarray(rng.uniform(-1, side + 1, size=(10_000, 2)).astype(np.float32))

    ids = cp.asnumpy(cs.tl.assign_points(pts, polys))
    pairs = cp.asnumpy(cs.tl.overlap_pairs(pts, polys))
    assigned = (ids >= 0).sum()
    assert pairs.shape[0] == assigned
    # Every pair's polygon index should equal the assigned id for that point.
    assert np.array_equal(ids[pairs[:, 0]], pairs[:, 1])


# ---------------------------------------------------------------------------
# predicate="intersects": edge-inclusive
# ---------------------------------------------------------------------------


def test_intersects_includes_boundary_points():
    """Points lying exactly on a polygon edge are excluded by 'contains'
    (default) and included by 'intersects'."""
    polys = _polys_from_squares([_square(0, 0, 1)])

    # A point exactly on the right edge: x=1, y=0.
    on_edge = cp.asarray([[1.0, 0.0]], dtype=cp.float32)
    interior = cp.asarray([[0.0, 0.0]], dtype=cp.float32)

    # contains: interior OK, edge rejected.
    ids_c = cp.asnumpy(cs.tl.assign_points(on_edge, polys, predicate="contains"))
    assert ids_c.tolist() == [-1]
    ids_c_interior = cp.asnumpy(
        cs.tl.assign_points(interior, polys, predicate="contains")
    )
    assert ids_c_interior.tolist() == [0]

    # intersects: both included.
    ids_i = cp.asnumpy(cs.tl.assign_points(on_edge, polys, predicate="intersects"))
    assert ids_i.tolist() == [0]


def test_overlap_pairs_intersects_toggles_edge():
    polys = _polys_from_squares([_square(0, 0, 1)])
    on_edge = cp.asarray([[1.0, 0.5]], dtype=cp.float32)

    assert cs.tl.overlap_pairs(on_edge, polys, predicate="contains").shape == (0, 2)

    p = cp.asnumpy(cs.tl.overlap_pairs(on_edge, polys, predicate="intersects"))
    assert p.tolist() == [[0, 0]]


def test_predicate_validation():
    polys = _polys_from_squares([_square(0, 0, 1)])
    pts = cp.asarray([[0.0, 0.0]], dtype=cp.float32)
    with pytest.raises(ValueError, match="predicate"):
        cs.tl.assign_points(pts, polys, predicate="nonsense")
    with pytest.raises(ValueError, match="predicate"):
        cs.tl.overlap_pairs(pts, polys, predicate="nonsense")
