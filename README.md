# cuspa: GPU spatial operations for spatial omics

[![CI](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml)

cuspa assigns transcript coordinates to cell polygons and builds sparse
cells-by-genes count matrices on the GPU. Geometry is stored in flat
GeoArrow-style buffers. GeoPandas and SpatialData inputs are converted before
GPU computation.

## Installation

Install the prebuilt wheel for your CUDA major version:

```bash
[uv] pip install cuspa-cu12  # CUDA 12
[uv] pip install cuspa-cu13  # CUDA 13
```

Both distributions provide the `cuspa` Python package; install only one per
environment. The optional adapters are extras, for example
`[uv] pip install "cuspa-cu12[geopandas]"`; `geoarrow` and `spatialdata` work
the same way.

cuspa requires Python 3.12 or newer, a compatible NVIDIA GPU and driver, a CUDA
12 or CUDA 13 runtime, and CuPy or CUDA-enabled PyTorch. Install the CuPy or
PyTorch package for your CUDA environment separately. Aggregation and I/O
adapters currently require CuPy.

To build from source instead, install the `cuspa` source distribution. This
requires the matching CUDA toolkit and a C++17 compiler:

```bash
[uv] pip install cuspa
```

See the [installation guide](https://nvidia.github.io/cuspa/install.html) for
wheel targets and source-build details.

## Usage

```python
import cuspa as cs

polys = cs.io.from_spatialdata(sdata, "cell_boundaries")
tx = cs.io.transcripts_from_spatialdata(sdata, "transcripts")

csr = cs.tl.aggregate_to_cells(
    [(tx.xy, tx.gene_ids)],
    polys,
    num_genes=len(tx.gene_names),
)
```

## Documentation

See the [documentation](https://nvidia.github.io/cuspa/) for installation,
usage, geometry input, and the API reference.

## Contributing

External contributions are open through pull requests. Report bugs and propose
changes through [GitHub Issues](https://github.com/NVIDIA/cuspa/issues/new/choose).
See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution process. All project
participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Do not report vulnerabilities through public GitHub issues. See
[SECURITY.md](SECURITY.md) for NVIDIA's private reporting channels.

## License

cuspa is licensed under Apache-2.0. See [LICENSE](LICENSE).
