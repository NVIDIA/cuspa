# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")
gpd = pytest.importorskip("geopandas")
pa = pytest.importorskip("pyarrow")

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

import cuspa as cs


def test_from_geopandas_polygon_with_hole():
    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
                holes=[[(1, 1), (1, 2), (2, 2), (1, 1)]],
            )
        ]
    )

    polys = cs.io.from_geopandas(gdf)
    assert polys.num_polygons == 1
    assert polys.num_rings == 2

    points = cp.asarray([[0.5, 0.5], [1.25, 1.25], [5.0, 5.0]], dtype=cp.float32)
    ids = cs.tl.assign_points(points, polys)
    assert ids.tolist() == [0, -1, -1]


def test_from_geopandas_multipolygon_keeps_one_row_per_geometry():
    gdf = gpd.GeoDataFrame(
        geometry=[
            MultiPolygon(
                [
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                    Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
                ]
            )
        ]
    )

    polys = cs.io.from_geopandas(gdf)
    assert polys.num_polygons == 1
    assert polys.num_rings == 2

    points = cp.asarray([[0.5, 0.25], [10.5, 10.25], [5.0, 5.0]], dtype=cp.float32)
    ids = cs.tl.assign_points(points, polys)
    assert ids.tolist() == [0, 0, -1]


def test_from_geoarrow_accepts_geodataframe_arrow_table():
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 10)]),
        ],
    )
    table = pa.table(
        gdf.to_arrow(geometry_encoding="geoarrow", interleaved=True, include_z=False)
    )

    polys = cs.io.from_geoarrow(table)
    assert polys.num_polygons == 2
    assert polys.num_rings == 2
    assert polys.num_points == 8

    aabbs = cp.asnumpy(polys.ensure_aabbs())
    assert np.allclose(aabbs, [[0, 0, 1, 1], [10, 10, 11, 11]])


def test_from_geoarrow_accepts_geoseries_arrow_array():
    geos = gpd.GeoSeries([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])])
    array = pa.array(
        geos.to_arrow(geometry_encoding="geoarrow", interleaved=False, include_z=False)
    )

    polys = cs.io.from_geoarrow(array)
    assert polys.num_polygons == 1
    assert polys.num_rings == 1
    assert polys.num_points == 4


def test_from_geoarrow_normalizes_sliced_array_offsets():
    geos = gpd.GeoSeries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 10)]),
        ]
    )
    array = pa.array(
        geos.to_arrow(geometry_encoding="geoarrow", interleaved=True, include_z=False)
    ).slice(1, 1)

    polys = cs.io.from_geoarrow(array)
    assert polys.num_polygons == 1
    assert polys.num_rings == 1
    assert cp.asnumpy(polys.ensure_aabbs()).tolist() == [[10.0, 10.0, 11.0, 11.0]]


def test_from_geoarrow_rejects_wkb_encoding():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    table = pa.table(gdf.to_arrow())
    with pytest.raises(TypeError, match="GeoArrow WKB"):
        cs.io.from_geoarrow(table)
