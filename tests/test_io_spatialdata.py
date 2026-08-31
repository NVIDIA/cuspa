# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the optional spatialdata adapter."""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")
sd = pytest.importorskip("spatialdata")
shapely = pytest.importorskip("shapely")

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import cuspa as cs


def _square_geom(cx: float, cy: float, r: float) -> Polygon:
    return Polygon(
        [
            (cx - r, cy - r),
            (cx + r, cy - r),
            (cx + r, cy + r),
            (cx - r, cy + r),
            (cx - r, cy - r),
        ]
    )


def _tiny_sdata():
    cells = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[_square_geom(0, 0, 1.0), _square_geom(10, 10, 1.0)],
    )
    shapes = sd.models.ShapesModel.parse(cells)

    tx = pd.DataFrame(
        {
            "x": [0.0, 0.5, -0.2, 10.0, 9.9, 100.0],  # last is unassigned
            "y": [0.0, 0.5, 0.3, 10.0, 9.8, 100.0],
            "feature_name": [
                "GENE_A",
                "GENE_A",
                "GENE_B",
                "GENE_C",
                "GENE_A",
                "GENE_Z",
            ],
        }
    )
    points = sd.models.PointsModel.parse(
        dd.from_pandas(tx, npartitions=1), coordinates={"x": "x", "y": "y"}
    )

    return sd.SpatialData(shapes={"cells": shapes}, points={"transcripts": points})


def test_from_spatialdata_polygons_roundtrip():
    sdata = _tiny_sdata()
    polys = cs.io.from_spatialdata(sdata, "cells")

    assert polys.num_polygons == 2
    assert polys.num_rings == 2
    # Each square ring has 5 coords (closed)
    assert polys.num_points == 10

    # AABBs match what we'd compute by hand
    aabbs = cp.asnumpy(polys.ensure_aabbs())
    assert np.allclose(aabbs, [[-1, -1, 1, 1], [9, 9, 11, 11]])


def test_transcripts_from_spatialdata_categorical_gene_ids():
    sdata = _tiny_sdata()
    tx = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

    assert tx.xy.shape == (6, 2)
    assert tx.gene_ids is not None
    assert tx.gene_names == ["GENE_A", "GENE_B", "GENE_C", "GENE_Z"]

    # Check roundtrip of gene encoding
    gene_ids_host = cp.asnumpy(tx.gene_ids).tolist()
    decoded = [tx.gene_names[i] for i in gene_ids_host]
    assert decoded == ["GENE_A", "GENE_A", "GENE_B", "GENE_C", "GENE_A", "GENE_Z"]


def test_spatialdata_end_to_end_aggregate():
    sdata = _tiny_sdata()
    polys = cs.io.from_spatialdata(sdata, "cells")
    tx = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

    csr = cs.tl.aggregate_to_cells(
        [(tx.xy, tx.gene_ids)], polys, num_genes=len(tx.gene_names)
    )

    # Cell 0 contains 3 transcripts: 2× GENE_A (gene 0), 1× GENE_B (gene 1)
    # Cell 1 contains 2 transcripts: 1× GENE_C (gene 2), 1× GENE_A (gene 0)
    # Last transcript (GENE_Z at 100,100) is unassigned and should be dropped.
    dense = cp.asnumpy(csr.to_cupy_csr().toarray())
    expected = np.array(
        [
            [2, 1, 0, 0],  # cell 0
            [1, 0, 1, 0],  # cell 1
        ],
        dtype=np.float32,
    )
    assert np.array_equal(dense, expected)


def test_from_spatialdata_accepts_multipolygon():
    from shapely.geometry import MultiPolygon

    gdf = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=[MultiPolygon([_square_geom(0, 0, 1), _square_geom(5, 5, 1)])],
    )
    shapes = sd.models.ShapesModel.parse(gdf)
    sdata = sd.SpatialData(shapes={"cells": shapes})
    polys = cs.io.from_spatialdata(sdata, "cells")

    assert polys.num_polygons == 1
    assert polys.num_rings == 2
    points = cp.asarray([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=cp.float32)
    ids = cs.tl.assign_points(points, polys)
    assert ids.tolist() == [0, 0, -1]
