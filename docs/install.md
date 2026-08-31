# Install

## Runtime requirements

cuspa requires Python 3.12+, a CUDA 12 or CUDA 13 toolkit, a compatible NVIDIA
driver, and either CuPy or CUDA-enabled PyTorch. The CUDA array provider is not
installed automatically because its package must match the deployment CUDA
runtime.

JAX arrays are not currently supported by the public container API because the
spatial index and result buffers are mutable.

## Source install

```bash
python -m pip install .
```

cuspa uses Hatchling with a `scikit-build-core` hook and a CUDA-enabled C++17
toolchain. The core extension links against `libcudart` and uses CUB from the
CUDA Toolkit. By default CMake compiles for the GPU detected on the build
machine.

## Prebuilt wheels

CUDA 12 and CUDA 13 wheels use distinct distribution names:

```bash
python -m pip install cuspa-cu12
python -m pip install cuspa-cu13
```

Choose exactly one. Both distributions install the same `cuspa` import package,
but contain native code built for different CUDA toolkits. Wheel filenames
normalize the distribution hyphen to an underscore, for example
`cuspa_cu12-0.0.1-...whl`.

For a reusable production wheel, set the supported architectures explicitly.
The final architecture should include virtual code so a newer compatible GPU
can JIT the kernel:

```bash
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;90-real;100-real;120' \
  python -m build --wheel
python -m twine check dist/*
```

Choose the architecture list for your fleet; a larger list increases wheel
size and build time. Build separate artifacts for each supported CUDA major.

Verify an installed artifact on a GPU host:

```bash
python -c 'import cuspa, cuspa._core; print(cuspa.__version__)'
python -m pytest -q
```

## Development install

```bash
python -m pip install -e ".[test]"
```

## Optional adapters

```bash
python -m pip install -e ".[geoarrow]"
python -m pip install -e ".[geopandas]"
python -m pip install -e ".[spatialdata]"
```

The core package stays dependency-light. Install CuPy or PyTorch separately,
using the vendor's package for the CUDA runtime deployed on the target host.
