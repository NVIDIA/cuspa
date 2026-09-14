API reference
=============

Import cuspa as:

.. code-block:: python

   import cuspa as cs

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
