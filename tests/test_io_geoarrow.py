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


def test_batches_dataset_and_scanner_read_projected_xy_in_order():
    import pyarrow.dataset as ds

    table = pa.table({"y_location": [1, 2, 3], "x_location": [0, 7, 3]})
    points = ds.dataset(table).filter(ds.field("x_location") < 5)
    polys = cs.io.from_geopandas(
        gpd.GeoDataFrame(geometry=[Polygon([(-1, 0), (1, 0), (1, 2), (-1, 2)])])
    )
    for source in (points.to_batches(), points, points.scanner()):
        np.testing.assert_array_equal(cs.tl.assign_points(source, polys), [0, -1])


@pytest.mark.parametrize("fmt", ["parquet", "ipc"])
def test_file_dataset_coordinates_preserve_partitions_and_filters(tmp_path, fmt):
    import pyarrow.dataset as ds

    from cuspa.io._points import host_points

    table = pa.table({"x": [2.0, None, 3.0, 1.0, 4.0], "y": [9, 7, 8, 5, 6]})
    ds.write_dataset(
        table, tmp_path, format=fmt, partitioning=["y"], partitioning_flavor="hive"
    )
    dataset = ds.dataset(tmp_path, format=fmt, partitioning="hive")
    for source in (dataset, dataset.filter(ds.field("y") > 6)):
        expected = source.to_table(columns=["x", "y"]).to_pandas().to_numpy()
        got = list(host_points(source, rows=2, dtype=np.float64))
        np.testing.assert_array_equal(np.concatenate(got), expected)


def test_prefetch_closes_source_after_one_pending_batch():
    from threading import Event

    from cuspa.io._points import _prefetch_one

    pending, closed = Event(), Event()
    seen = []

    def batches():
        try:
            for i in range(4):
                seen.append(i)
                if i == 1:
                    pending.set()
                yield i
        finally:
            closed.set()

    reader = _prefetch_one(batches())
    assert next(reader) == 0 and pending.wait(5)
    reader.close()
    assert closed.is_set() and seen == [0, 1]


def test_dictionary_scan_bounds_row_groups_and_preserves_arrow_projection(
    tmp_path, monkeypatch
):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from cuspa.io import _points

    genes = pa.array((["B", None, "A"] * 10), pa.large_string())
    pq.write_table(
        pa.table({"raw_gene": genes}), tmp_path / "data.parquet", row_group_size=1
    )
    src = ds.dataset(tmp_path, format="parquet")
    projection = {"g": ds.field("raw_gene")}
    read = _points._read_batches
    groups = []

    def recorded(group, cols, rows):
        groups.append([len(f.row_groups) for f in group.get_fragments()])
        return read(group, cols, rows)

    monkeypatch.setattr(_points, "_read_batches", recorded)
    actual = list(_points._dataset_batches(src, projection, 4, dictionary="raw_gene"))
    assert groups == [[8, 8, 8, 6]]
    assert all(
        b["g"].type == pa.dictionary(pa.int32(), pa.large_string()) for b in actual
    )
    assert pa.Table.from_batches(actual)["g"].to_pylist() == genes.to_pylist()
    monkeypatch.setattr(_points, "_read_batches", read)
    filtered = src.filter(ds.field("raw_gene") == "B")
    matches = _points._dataset_batches(filtered, projection, 4, dictionary="raw_gene")
    assert pa.Table.from_batches(matches)["g"].to_pylist() == ["B"] * 10

    # A changed logical schema and a custom format retain their declared reader.
    custom = ds.ParquetFileFormat(
        default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False)
    )
    for other in (
        src.replace_schema(src.schema.append(pa.field("missing", pa.int8()))),
        ds.dataset(tmp_path, format=custom),
    ):
        assert all(
            b["g"].type == pa.large_string()
            for b in _points._dataset_batches(
                other, projection, 4, dictionary="raw_gene"
            )
        )

    buf = pa.BufferOutputStream()
    pq.write_table(pa.table({"raw_gene": genes}), buf)
    fmt = ds.ParquetFileFormat()
    frag = fmt.make_fragment(pa.BufferReader(buf.getvalue()))
    memory = ds.FileSystemDataset([frag], frag.physical_schema, fmt)
    batches = _points._dataset_batches(memory, projection, 4, dictionary="raw_gene")
    assert pa.Table.from_batches(batches)["g"].to_pylist() == genes.to_pylist()


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
