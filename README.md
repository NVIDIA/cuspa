# cuspa: GPU spatial operations for spatial omics

[![CI](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA/cuspa/actions/workflows/ci.yml)

cuspa assigns transcript coordinates to cell polygons and builds sparse
cells-by-genes count matrices on the GPU. Geometry is stored in flat
GeoArrow-style buffers. GeoPandas and SpatialData inputs are converted before
GPU computation.

## Installation

cuspa requires Python 3.12 or newer, a compatible NVIDIA GPU and driver, a CUDA
12 or CUDA 13 runtime, and CuPy or CUDA-enabled PyTorch. Aggregation and I/O
adapters currently require CuPy. Source builds also require the matching CUDA
toolkit and a C++17 compiler.

Install from source with:

```bash
python -m pip install .
```

When published to PyPI, wheels will use separate distribution names for CUDA 12
and CUDA 13:

```bash
python -m pip install cuspa-cu12  # CUDA 12
python -m pip install cuspa-cu13  # CUDA 13
```

Both distributions provide the `cuspa` Python package; they must not be
installed together in one environment.

Install the CuPy or PyTorch package for your CUDA environment separately. From
a source checkout, install optional adapters with:

```bash
python -m pip install ".[geoarrow]"
python -m pip install ".[geopandas]"
python -m pip install ".[spatialdata]"
```

See the [installation guide](docs/install.md) for source-build and wheel
details.

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

See the [documentation](docs/index.md) for installation, usage, geometry input,
and the API reference.

## Security

Do not report vulnerabilities through public GitHub issues. See
[SECURITY.md](SECURITY.md) for NVIDIA's private reporting channels.

## License

cuspa is licensed under Apache-2.0. See [LICENSE](LICENSE).
