# Installation

## Requirements

cuspa requires Python 3.12 or newer, a compatible NVIDIA GPU and driver, a CUDA
12 or CUDA 13 runtime, and CuPy or CUDA-enabled PyTorch. Aggregation and I/O
adapters currently require CuPy.

Source builds also require a matching CUDA toolkit that supports the target GPU
and a C++17 compiler. Install the CuPy or PyTorch package matching your CUDA
environment separately.

## Source install

```bash
python -m pip install .
```

Source installs compile for the GPU detected on the build machine.

## Release wheels

When published to PyPI, install the distribution matching your CUDA major:

```bash
python -m pip install cuspa-cu12  # CUDA 12
python -m pip install cuspa-cu13  # CUDA 13
```

Install only one. Both distributions provide the `cuspa` Python package.
Wheels target manylinux_2_28 on x86_64 and aarch64. CUDA 12 wheels target SM
75, 80, 86, 89, and 90; CUDA 13 wheels also target SM 100 and 120.

Verify the installation on a GPU host:

```bash
python -c 'import cuspa, cuspa._core; print(cuspa.__version__)'
```

## Development install

```bash
python -m pip install -e ".[test]"
```

## Optional adapters

From a source checkout:

```bash
python -m pip install ".[geoarrow]"
python -m pip install ".[geopandas]"
python -m pip install ".[spatialdata]"
```

For a wheel, replace `.` with `cuspa-cu12` or `cuspa-cu13`.
