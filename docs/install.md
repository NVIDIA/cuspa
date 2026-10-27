# Installation

## Install a wheel (recommended)

Install the prebuilt wheel that matches your CUDA major version:

```bash
[uv] pip install cuspa-cu12  # CUDA 12
[uv] pip install cuspa-cu13  # CUDA 13
```

Install only one. Both distributions provide the `cuspa` Python package. Wheels
target manylinux_2_28 on x86_64 and aarch64. CUDA 12 wheels target SM 75, 80,
86, 89, and 90; CUDA 13 wheels also target SM 100 and 120.

The optional adapters are extras of the same distribution:

```bash
[uv] pip install "cuspa-cu12[geoarrow]"     # or cuspa-cu13
[uv] pip install "cuspa-cu12[geopandas]"
[uv] pip install "cuspa-cu12[spatialdata]"
```

Verify the installation on a GPU host:

```bash
python -c 'import cuspa, cuspa._core; print(cuspa.__version__)'
```

## Requirements

cuspa requires Python 3.12 or newer, a compatible NVIDIA GPU and driver, a CUDA
12 or CUDA 13 runtime, and CuPy or CUDA-enabled PyTorch. Install the CuPy or
PyTorch package matching your CUDA environment separately. Aggregation and I/O
adapters currently require CuPy.

## Build from source

The `cuspa` distribution is the source distribution. Installing it compiles
cuspa for the GPU detected on the build machine, which requires a matching CUDA
toolkit that supports the target GPU and a C++17 compiler.

```bash
[uv] pip install cuspa
```

The optional adapters are extras here too:

```bash
[uv] pip install "cuspa[geoarrow]"
[uv] pip install "cuspa[geopandas]"
[uv] pip install "cuspa[spatialdata]"
```

## Development install

```bash
[uv] pip install -e ".[test]"
```
