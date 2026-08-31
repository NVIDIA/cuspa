# cuspa

An NVIDIA CUDA library for point-in-polygon and transcript-to-cell aggregation
in spatial omics.

[![CI](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml)

cuspa keeps geometry in flat GeoArrow-style buffers and keeps transcript
points as numeric arrays. GeoPandas/SpatialData inputs are supported at the
adapter boundary, but Shapely/GeoPandas are not used in the GPU compute path.

## Requirements

- Python 3.12 or newer
- NVIDIA CUDA Toolkit 12 or 13 and a compatible NVIDIA driver
- a CUDA array provider: CuPy or PyTorch

The core extension uses Python's stable ABI, but CUDA wheels remain specific to
the CUDA runtime and GPU architectures they were built for.

## Install from source

```bash
python -m pip install .
```

For prebuilt wheels, install the distribution matching the CUDA major:

```bash
python -m pip install cuspa-cu12  # CUDA 12
python -m pip install cuspa-cu13  # CUDA 13
```

Both distributions provide the `cuspa` Python package; they must not be
installed together in one environment.

Install the CuPy or PyTorch build matching your CUDA environment separately.
Optional adapters can be installed with the package:

```bash
python -m pip install ".[geoarrow]"
python -m pip install ".[geopandas]"
python -m pip install ".[spatialdata]"
```

Source installs compile for the GPU on the build machine. Release builders
must set an explicit architecture set; see [the install guide](docs/install.md).

## Example

```python
import cuspa as cs

polys = cs.io.from_geoarrow(geoarrow_table)
tx = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

csr = cs.tl.aggregate_to_cells(
    [(tx.xy, tx.gene_ids)],
    polys,
    num_genes=len(tx.gene_names),
)
```

Core APIs:

- `cs.Polygons(part_offsets, ring_offsets, points_xy)`
- `cs.tl.assign_points(points, polygons, predicate="contains")`
- `cs.tl.overlap_pairs(points, polygons, predicate="contains")`
- `cs.tl.aggregate_to_cells(batches, polygons, num_genes)` for bounded-
  memory transcript aggregation; the caller supplies a batch iterator
- `cs.io.from_geoarrow(...)`, `cs.io.from_geopandas(...)`,
  `cs.io.from_spatialdata(...)`

## Docs

Documentation sources live in `docs/` and use NVIDIA's Sphinx theme. The
production `docs.nvidia.com` URL and version switcher will be configured after
the Cuspa docset is provisioned by NVIDIA Docs Platform.

## Security

Do not report vulnerabilities through public GitHub issues. See
[SECURITY.md](SECURITY.md) for NVIDIA's private reporting channels.

## License

Cuspa is licensed under Apache-2.0. See [LICENSE](LICENSE).

## Operational guarantees

Public APIs validate array layout, dtype, polygon offsets, finite coordinates,
and CSR bounds before launching kernels. Derived AABBs and spatial indexes are
cached; call `polygons.clear_cache()` after mutating a polygon buffer. Pair and
CSR offsets are `int32`, so operations fail cleanly if a result would exceed
2^31-1 entries.
