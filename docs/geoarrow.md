# GeoArrow input

cuspa's polygon container mirrors the GeoArrow polygon layout:

```text
part_offsets   polygon -> rings
ring_offsets   ring -> coordinates
points_xy      coordinate buffer, shape (N, 2)
```

## Direct GeoArrow input

```python
import cuspa as cs

polygons = cs.io.from_geoarrow(table)
```

`from_geoarrow` accepts Arrow tables, arrays, or chunked arrays containing
GeoArrow Polygon or MultiPolygon geometry. WKB geometry columns are rejected
because they require geometry parsing.

## GeoPandas input

```python
polygons = cs.io.from_geopandas(gdf)
```

`from_geopandas` converts the input with:

```python
gdf.to_arrow(geometry_encoding="geoarrow", interleaved=True)
```

## SpatialData input

```python
polygons = cs.io.from_spatialdata(sdata, "cell_boundaries")
```

`from_spatialdata` reads a named SpatialData shapes element through the
GeoPandas adapter.

## Supported geometry

The current adapter supports:

- Polygon
- Polygon with holes
- MultiPolygon, preserving one row per input geometry

Input points and polygons must already be in the same coordinate system.
