# Usage

## Point assignment

```python
import cuspa as cs

polygons = cs.io.from_geoarrow(geoarrow_table)
cell_ids = cs.tl.assign_points(points_xy, polygons)
```

`cell_ids` contains one `int32` label per point in input order. A value of
`-1` means the point was not assigned to a polygon. With CuPy polygons,
you can also pass NumPy coordinates, a SpatialData points frame, an Arrow
dataset, or Arrow record batches directly. These inputs return NumPy labels.
CuPy inputs also return NumPy labels if a label array would exceed available
GPU memory. Use `out=` to supply a label array.

`overlap_pairs(points, polygons)` returns every matching point and polygon.
Device results have two `int32` columns. Host inputs return a NumPy array with
two `int64` columns. CuPy inputs use the same host result above `2**31 - 1`
input points or when GPU memory runs short.

## Transcript aggregation

`aggregate_to_cells` consumes an iterable of `(points_xy, gene_ids)` NumPy or CuPy
batches, a live points frame, or a PyArrow Dataset. For example,
`cs.tl.aggregate_to_cells(dataset, polygons)` reads `x` and `y` (or
`x_location` and `y_location`) and sets `csr.gene_names` from the observed
values of `feature_name`. Use `gene_column=` for another string column. For
in-memory coordinates and gene IDs, pass a single batch.

```python
csr = cs.tl.aggregate_to_cells(
    [(points_xy, gene_ids)],
    polygons,
    num_genes=num_genes,
)
```

cuSpa aggregates batches on the GPU while the counts and workspace fit.
A points frame, Arrow dataset or iterable of NumPy batches returns a NumPy-backed
CSR. CuPy batches return a device CSR while GPU memory permits. When GPU memory
runs short, cuSpa determines the size of the host CSR and fills its arrays on
a second pass. For a one-shot source, it retains the partial counts in host RAM.

Use `csr.to_cupy_csr()` or `csr.to_torch_sparse()` to create a framework-native
sparse matrix.

Counts use exact `int32` arithmetic and raise `OverflowError` past `2**31 - 1`.
`indptr` and `indices` switch to `int64` past `2**31 - 1` nonzeros. For
these matrices, use the raw arrays or `to_torch_sparse()`.

## SpatialData adapter

```python
import cuspa as cs

polygons = cs.io.from_spatialdata(sdata, "cell_boundaries")
csr = cs.tl.aggregate_to_cells(sdata.points["transcripts"], polygons)
```

The resulting `csr.gene_names` gives the column order. For a plain Parquet-backed
frame, cuSpa reads only coordinates and the requested gene column. String gene
labels require an initial gene-only scan to establish the vocabulary.
For a transformed frame, the largest computed source partition affects host
memory use.

## Buffer lifetime and streams

`Polygons` validates its buffers at construction and caches derived AABBs and a
uniform-grid index. Treat the buffers as immutable. After changing them in
place, call `polygons.clear_cache()` before the next query.

The optional `stream=` argument is a raw CUDA stream handle used by the native
kernels. Operations may synchronize it while sizing indexes or results. Keep
inputs alive and synchronize or use the same stream before consuming results
elsewhere.

## Limits

Polygon, ring, and spatial-index offsets use signed 32-bit indices. The polygon
buffers and index must fit on the GPU. Complete host results require enough
host RAM for their arrays and working storage.
