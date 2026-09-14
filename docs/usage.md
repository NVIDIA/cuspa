# Usage

## Point assignment

```python
import cuspa as cs

polygons = cs.io.from_geoarrow(geoarrow_table)
cell_ids = cs.tl.assign_points(points_xy, polygons)
```

`cell_ids` is an `int32` device array. A value of `-1` means the point was not
assigned to a polygon.

## Transcript aggregation

`aggregate_to_cells` consumes an iterable of `(points_xy, gene_ids)` CuPy
batches. It does not read files or choose a batch size. Pass a one-item
iterable for in-memory data:

```python
csr = cs.tl.aggregate_to_cells(
    [(points_xy, gene_ids)],
    polygons,
    num_genes=num_genes,
)
```

Yield one CUDA batch at a time for larger inputs:

```python
import cupy as cp


def gpu_batches(points_host, gene_ids_host, batch_size):
    for start in range(0, len(gene_ids_host), batch_size):
        stop = start + batch_size
        yield (
            cp.asarray(points_host[start:stop], dtype=cp.float32),
            cp.asarray(gene_ids_host[start:stop], dtype=cp.int32),
        )


csr = cs.tl.aggregate_to_cells(
    gpu_batches(points_host, gene_ids_host, batch_size=1_000_000),
    polygons,
    num_genes,
)
```

The returned `CSRMatrix` contains three device arrays:

- `indptr`
- `indices`
- `data`

Its `shape` field is a Python tuple.

Use `csr.to_cupy_csr()` or `csr.to_torch_sparse()` to create a framework-native
sparse matrix without copying through host memory.

Batch aggregation uses `float32` sparse accumulation. Counts for a
`(cell, gene)` pair must be below `2**24` (16,777,216).

### Choosing a batch size

GPU memory must hold the largest input batch, its aggregation workspace, the
polygon index, the accumulated CSR, and sparse-merge temporaries. Batching
limits input memory; the final CSR must still fit on the GPU.

## SpatialData adapter

```python
import cuspa as cs

polygons = cs.io.from_spatialdata(sdata, "cell_boundaries")
transcripts = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

csr = cs.tl.aggregate_to_cells(
    [(transcripts.xy, transcripts.gene_ids)],
    polygons,
    num_genes=len(transcripts.gene_names),
)
```

For large transcript tables, read `x`, `y`, and gene columns in record batches
instead of materializing a point GeoDataFrame.

## Buffer lifetime and streams

`Polygons` validates its buffers at construction and caches derived AABBs and a
uniform-grid index. Treat the buffers as immutable. After changing them in
place, call `polygons.clear_cache()` before the next query.

The optional `stream=` argument is a raw CUDA stream handle used by the native
kernels. Operations may synchronize it while sizing indexes or results. For
`aggregate_to_cells`, also make the corresponding CuPy stream current because
buffer copies and sparse merging use CuPy's current stream. Keep inputs alive
and synchronize or use the same stream before consuming results elsewhere.

## Limits

Polygon, ring, spatial-index, and overlap-pair offsets use signed 32-bit
indices. Spatial indexes and overlap results are checked against the 2^31-1
limit. Each aggregation batch is limited to 2^31-1 transcripts.
