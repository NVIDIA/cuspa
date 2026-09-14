# cuspa: GPU spatial operations for spatial omics

cuspa assigns transcript points to cell polygons and builds sparse
cells-by-genes count matrices on the GPU. Geometry uses flat GeoArrow-style
buffers.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`download;sd-text-primary` Install cuspa
:link: install
:link-type: doc

Install from source. When wheels are published, select one for your CUDA runtime.
:::

:::{grid-item-card} {fas}`play;sd-text-primary` Usage
:link: usage
:link-type: doc

Assign points and aggregate transcripts.
:::

:::{grid-item-card} {fas}`shapes;sd-text-primary` Geometry input
:link: geoarrow
:link-type: doc

Load Polygon and MultiPolygon data from GeoArrow, GeoPandas, or SpatialData.
:::

:::{grid-item-card} {fas}`code;sd-text-primary` API reference
:link: api/index
:link-type: doc

Containers, spatial tools, and I/O adapters.
:::
::::

```{toctree}
:caption: General
:hidden: true
:maxdepth: 1

install
usage
geoarrow
api/index
```
