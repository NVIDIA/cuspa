# Usage

## Point assignment

```python
import cuspa as cs

polygons = cs.io.from_geoarrow(geoarrow_table)
cell_ids = cs.tl.assign_points(points_xy, polygons)
```

`cell_ids` is an `int32` device array. A value of `-1` means the point was not
assigned to any polygon.

## Transcript aggregation

`aggregate_to_cells` always consumes an iterable of GPU batches. cuspa owns
the GPU aggregation and sparse merge; the caller owns reading the transcript
source and deciding the batch size. It does not currently open Parquet/Arrow
files or split an input array for you.

For an in-memory dataset, pass a one-item list:

```python
csr = cs.tl.aggregate_to_cells(
    [(points_xy, gene_ids)],
    polygons,
    num_genes=num_genes,
)
```

For a large dataset, create a generator that transfers one source slice at a
time. cuspa pulls one pair at a time and does not retain earlier point or
gene-id batches. This minimal helper illustrates the required shape; replace
the host slices with batches from your Arrow or Parquet reader as appropriate.

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
    gpu_batches(points_host, gene_ids_host, batch_size=100_000_000),
    polygons,
    num_genes,
)
```

Within `aggregate_to_cells`, the polygon index is built once. The first batch
initializes one sparse GPU accumulator; each later batch is aggregated and
added to that accumulator on device. After the final batch, cuspa coalesces
duplicate entries, sorts gene indices within each cell, and returns an `int32`
CSR that remains on the GPU.

Inputs for this operation must currently be CuPy arrays because CuPy performs
the sparse accumulation. The finished result can be used from either CuPy or
CUDA PyTorch without a host transfer.

The result is a lightweight CSR container with device arrays:

- `indptr`
- `indices`
- `data`
- `shape`

Use `csr.to_cupy_csr()` or `csr.to_torch_sparse()` to create a framework-native
sparse matrix without moving it through host memory.

The temporary accumulator uses `float32`, which exactly represents counts
below `2**24` (16,777,216) for each `(cell, gene)` pair. cuspa raises an error
at that conservative limit rather than returning rounded counts.

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

For very large transcript tables, prefer loading `x`/`y`/gene columns as arrays
from Arrow or Parquet rather than materializing a point GeoDataFrame.

### Choosing a batch size

GPU memory is bounded by the largest input batch, its temporary aggregation
workspace, the cell polygon index, and one growing final sparse accumulator.
Batching avoids storing all transcript points at once, but the final CSR still
has to fit on the GPU. For the full ATERA dataset, 100 million transcripts per
batch used 11.4 GB of VRAM and produced a 4.4 GB final CSR.

If `points_host` is itself too large to materialize, use your file reader's
record-batch interface in `gpu_batches` instead. That keeps both host and GPU
point storage bounded; the aggregation call does not otherwise change.

## Buffer lifetime and streams

`Polygons` validates its buffers at construction and caches derived AABBs and a
uniform-grid index. The buffers are expected to remain immutable. If they are
changed in place, call `polygons.clear_cache()` before the next query.

The optional `stream=` argument is a raw CUDA stream handle. Kernel launches
are asynchronous with respect to the host. Keep all inputs alive and order
downstream work on the same stream, or synchronize that stream before consuming
results elsewhere. Index construction performs the minimum host synchronization
needed to size its output buffers.

## Limits

Polygon, ring, spatial-index, overlap-pair, and CSR offsets use signed 32-bit
indices. cuspa raises an error if an index or many-to-many result would exceed
2^31-1 entries. Chunk workloads above this limit.
