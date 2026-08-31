# cuspa: GPU spatial operations for spatial omics

cuspa assigns transcript points to cell polygons and builds sparse
cells-by-genes count matrices on the GPU. It keeps geometry in compact,
GeoArrow-style buffers rather than moving it through a CPU geometry engine.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`download;sd-text-primary` Install cuspa
:link: install
:link-type: doc

Build from source or choose the wheel that matches your CUDA runtime.
:::

:::{grid-item-card} {fas}`play;sd-text-primary` Start using cuspa
:link: usage
:link-type: doc

Assign points, aggregate transcripts, and reuse spatial indexes.
:::

:::{grid-item-card} {fas}`shapes;sd-text-primary` Bring your own geometry
:link: geoarrow
:link-type: doc

Load Polygon and MultiPolygon data from GeoArrow, GeoPandas, or SpatialData.
:::

:::{grid-item-card} {fas}`code;sd-text-primary` Browse the API
:link: api/index
:link-type: doc

Explore the public containers, tools, and I/O adapters.
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
