# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from cuspa.io._points import host_points


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


def test_live_parquet_coordinates_follow_frame_order_and_edits(tmp_path):
    for i in (1, 2, 10):
        part = pd.DataFrame({"x": [i + 1e-9, -1.0], "y": [0.0, i]})
        if i == 1:
            part = part.astype("float32")
        part.to_parquet(tmp_path / f"part.{i}.parquet", row_group_size=1)
    points = dd.read_parquet(tmp_path)
    filtered = dd.read_parquet(tmp_path, filters=[("x", ">", 0)])
    split = dd.read_parquet(tmp_path, split_row_groups=True).partitions[1::2]
    edited = points[points.x > 0].assign(y=lambda df: df.x + 4)
    for frame in (points, filtered, split, edited):
        blocks = list(host_points(frame, rows=1, dtype=np.float64))
        assert all(len(block) == 1 for block in blocks)
        expected = frame.compute()[["x", "y"]]
        np.testing.assert_array_equal(np.concatenate(blocks), expected)


def test_pointsmodel_parquet_aliases_and_row_groups(tmp_path):
    from cuspa.io._points import _parquet_scan

    path = tmp_path / "transcripts.parquet"
    data = pd.DataFrame(
        {
            "y_location": np.arange(7, dtype=np.float32),
            "x_location": np.arange(7, dtype=np.float32)[::-1],
            "gene": pd.array(["b", None, "c", "b", "a", "b", "c"], dtype="string"),
            "qv": np.arange(7),
            "is_gene": [True] * 7,
        }
    )
    data.to_parquet(path, row_group_size=2)
    raw = dd.read_parquet(path, split_row_groups=2)
    points = sd.models.PointsModel.parse(
        raw, coordinates={"x": "x_location", "y": "y_location"}, sort=False
    )
    scan = _parquet_scan(points, dictionary="gene")
    assert scan is not None
    for _ in range(2):
        xy, genes = zip(
            *[
                (np.column_stack((b["x"], b["y"])), b["gene"].to_pylist())
                for b in scan(["x", "y", "gene"], 2)
            ],
            strict=True,
        )
        np.testing.assert_array_equal(
            np.concatenate(xy), data[["x_location", "y_location"]]
        )
        assert [g for batch in genes for g in batch] == [
            "b",
            None,
            "c",
            "b",
            "a",
            "b",
            "c",
        ]
    assert _parquet_scan(points.assign(x=points.x + 1), dictionary="gene") is None
    assert (
        _parquet_scan(points.assign(gene=raw.gene.str.upper()), dictionary="gene")
        is None
    )
    coords = sd.models.PointsModel.parse(
        dd.read_parquet(path, columns=["x_location", "y_location"]),
        coordinates={"x": "x_location", "y": "y_location"},
        sort=False,
    )
    assert _parquet_scan(coords) is not None


@pytest.mark.parametrize("all_missing", [False, True])
def test_live_parquet_transcripts_preserve_gene_codes_and_edits(tmp_path, all_missing):
    from cuspa.io._spatialdata import _transcript_source

    for i, genes in [(1, ["z", None]), (2, ["a", "z"]), (10, ["b", "a"])]:
        frame = pd.DataFrame({"x": [float(i), -2.0], "y": [0.0, 1.0]})
        frame["gene"] = pd.array([None, None] if all_missing else genes, dtype="string")
        frame.to_parquet(tmp_path / f"part.{i}.parquet", row_group_size=1)
    points = dd.read_parquet(tmp_path)
    edited = points[points.x > 0].assign(gene=lambda df: df.gene.fillna("missing"))
    for source in (points, edited):
        expected = source.compute()
        genes = expected.gene.astype("category")
        batches, names = _transcript_source(source, "gene", np.float64)
        assert names == list(genes.cat.categories)
        for _ in range(2):
            xy, ids = zip(*batches(), strict=True)
            np.testing.assert_array_equal(np.concatenate(ids), genes.cat.codes)
            np.testing.assert_array_equal(np.concatenate(xy), expected[["x", "y"]])


def test_arrow_vocabulary_contains_only_observed_values():
    import pyarrow as pa

    from cuspa.io._spatialdata import _arrow_transcripts

    # The same dictionary gains an observed gene only in a later batch.
    dictionary = pa.array(["b", "a", "unused"])
    indices = [[0] * 128 + [None], [1, 0], [1]]
    batches = []
    for codes in indices:
        genes = pa.DictionaryArray.from_arrays(pa.array(codes, pa.int32()), dictionary)
        xy = pa.array(np.zeros(len(codes), np.float32))
        batches.append(pa.record_batch([xy, xy, genes], names=["x", "y", "gene"]))

    def scan(columns):
        return (batch.select(columns) for batch in batches)

    source, names = _arrow_transcripts(scan, "gene", np.float32)
    assert names == ["a", "b"]
    for _ in range(2):
        np.testing.assert_array_equal(
            np.concatenate([ids for _, ids in source()]), [1] * 128 + [-1, 0, 1, 0]
        )


def test_from_spatialdata_polygons_roundtrip():
    sdata = _tiny_sdata()
    polys = cs.io.from_spatialdata(sdata, "cells")

    assert polys.num_polygons == 2
    assert polys.num_rings == 2
    assert polys.num_points == 10

    aabbs = cp.asnumpy(polys.ensure_aabbs())
    assert np.allclose(aabbs, [[-1, -1, 1, 1], [9, 9, 11, 11]])


@pytest.mark.parametrize("categorical", [False, True])
def test_transcript_partitions_share_eager_gene_codes(categorical):
    from cuspa.io._spatialdata import _transcript_source

    genes = pd.Series(["z", None, "a"], dtype="string")
    if categorical:
        genes = genes.astype(pd.CategoricalDtype(["z", "unused", "a"]))
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "gene": genes})
    points = dd.from_pandas(frame, npartitions=3)
    if categorical:
        points = points.assign(gene=points.gene.cat.as_unknown())
    blocks, names = _transcript_source(points, "gene", np.float64)
    xy, ids = zip(*blocks(), strict=True)
    expected = genes.astype("category")
    assert names == list(expected.cat.categories)
    np.testing.assert_array_equal(np.concatenate(ids), expected.cat.codes)
    np.testing.assert_array_equal(np.concatenate(xy), frame[["x", "y"]])


def test_transcripts_from_spatialdata_categorical_gene_ids():
    sdata = _tiny_sdata()
    tx = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

    assert tx.xy.shape == (6, 2)
    assert tx.gene_ids is not None
    assert tx.gene_names == ["GENE_A", "GENE_B", "GENE_C", "GENE_Z"]

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
    streamed = cs.tl.aggregate_to_cells(sdata.points["transcripts"], polys)
    assert streamed.gene_names == tx.gene_names
    for field in ("indptr", "indices", "data"):
        cp.testing.assert_array_equal(getattr(streamed, field), getattr(csr, field))

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
