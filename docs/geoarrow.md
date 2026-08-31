# GeoArrow Input

cuspa's polygon container mirrors the GeoArrow polygon layout:

```text
part_offsets   polygon -> rings
ring_offsets   ring -> coordinates
points_xy      coordinate buffer, shape (N, 2)
```

## Direct GeoArrow input

```python
polygons = cs.io.from_geoarrow(table)
```

`from_geoarrow` accepts Arrow tables, arrays, or chunked arrays containing
GeoArrow Polygon or MultiPolygon geometry. WKB geometry columns are rejected
because they require geometry parsing.

## GeoPandas input

```python
polygons = cs.io.from_geopandas(gdf)
```

This is a convenience adapter for callers that already have GeoPandas data.
Internally it calls:

```python
gdf.to_arrow(geometry_encoding="geoarrow", interleaved=True)
```

After conversion, cuspa uses Arrow numeric buffers and GPU arrays. It does not
construct Shapely transcript points or call Shapely predicates in the compute
path.

## Supported geometry

The current adapter supports:

- Polygon
- Polygon with holes
- MultiPolygon, preserving one row per input geometry

Input points and polygons must already be in the same coordinate system.
