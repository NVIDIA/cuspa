API reference
=============

What to use when
----------------

Spatial operations
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - What it does
   * - :func:`cuspa.tl.assign_points`
     - Assigns each point to one polygon. It returns an ``int32`` polygon ID
       per point, or ``-1`` where no polygon contains the point.
   * - :func:`cuspa.tl.overlap_pairs`
     - Returns every matching point--polygon pair. Use it when polygons can
       overlap and one point may belong to more than one polygon.
   * - :func:`cuspa.tl.aggregate_to_cells`
     - Consumes one or more CuPy transcript batches and counts their gene IDs
       into one canonical, GPU-resident sparse cells-by-genes
       :class:`cuspa.tl.CSRMatrix`. It keeps one sparse accumulator on device;
       the caller supplies the source iterator and batch size.

Geometry and data adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - What it does
   * - :func:`cuspa.io.from_geoarrow`
     - Transfers GeoArrow Polygon or MultiPolygon buffers to a GPU
       :class:`cuspa.Polygons` container.
   * - :func:`cuspa.io.from_geopandas`
     - Converts an existing GeoPandas geometry column through its GeoArrow
       representation; it does not run geometry operations in GeoPandas.
   * - :func:`cuspa.io.from_spatialdata`
     - Reads a named SpatialData shapes element and converts its polygons to
       a GPU :class:`cuspa.Polygons` container.
   * - :func:`cuspa.io.transcripts_from_spatialdata`
     - Reads a named SpatialData points element into GPU xy coordinates and,
       when available, integer gene IDs and gene names for aggregation.

Reusable containers
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - What it does
   * - :meth:`cuspa.Polygons.validate`
     - Checks that polygon buffers have compatible shapes, dtypes, array
       providers, CUDA devices, and offsets before a CUDA kernel is launched.
   * - :meth:`cuspa.Polygons.ensure_aabbs`
     - Computes polygon bounding boxes once and caches them for later queries.
   * - :meth:`cuspa.Polygons.ensure_index`
     - Builds or reuses the uniform-grid spatial index used by point queries.
   * - :meth:`cuspa.Polygons.clear_cache`
     - Drops cached bounding boxes and the spatial index after in-place buffer
       mutation.
   * - :meth:`cuspa.tl.CSRMatrix.to_cupy_csr`
     - Wraps aggregated counts as a ``cupyx.scipy.sparse.csr_matrix``.
   * - :meth:`cuspa.tl.CSRMatrix.to_torch_sparse`
     - Wraps aggregated counts as a CUDA ``torch.sparse_csr_tensor``.

Containers
----------

.. autoclass:: cuspa.Polygons
   :members:

.. autoclass:: cuspa.SpatialIndex
   :members:

Spatial tools
-------------

.. autoclass:: cuspa.tl.CSRMatrix
   :members:

.. autofunction:: cuspa.tl.aggregate_to_cells

.. autofunction:: cuspa.tl.assign_points

.. autofunction:: cuspa.tl.overlap_pairs

I/O adapters
------------

.. autoclass:: cuspa.io.Transcripts
   :members:

.. autofunction:: cuspa.io.from_geoarrow

.. autofunction:: cuspa.io.from_geopandas

.. autofunction:: cuspa.io.from_spatialdata

.. autofunction:: cuspa.io.transcripts_from_spatialdata
