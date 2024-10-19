Web sources internal API
========================

These modules are abstracted by :meth:`kadlu.load <kadlu.__init__.load>`, which 
should generally be used instead for fetching and loading from web sources. 
These docs are provided for contributors wishing to add new web data source 
modules that can be accessed by :meth:`kadlu.load <kadlu.__init__.load>`.

:meth:`kadlu.load <kadlu.__init__.load>` works 
by mapping strings to function calls in :mod:`kadlu.geospatial.data_sources.source_map`
as a convenience by passing the source and variable name; new data source 
modules should have a corresponding entry within the fetch and load maps there. 

A general pattern for adding fetch/load modules is that fetching is an implicit 
action of the load function. each fetch and load function should accept a set 
of boundary arguments equal to or subsetting the following function signature, 
such that keyword args can be passed as a dict by fetch_handler:

``(south=-90, west=-180, north=90, east=180, top=0, bottom=5000, start=datetime(2000,1,1), end=datetime(2000,1,1,1))``

|

Fetch Handler
-------------

used for generating fetch requests for web data sources

.. automodule:: kadlu.geospatial.data_sources.fetch_handler
   :members:
   :undoc-members:
   :show-inheritance:

Source Map
----------

used for generating fetch requests for web data sources

.. automodule:: kadlu.geospatial.data_sources.source_map
   :members:
   :undoc-members:
   :show-inheritance:

|

CHS
---

.. automodule:: kadlu.geospatial.data_sources.chs
   :members:
   :undoc-members:
   :show-inheritance:

ERA5
----

.. automodule:: kadlu.geospatial.data_sources.era5
   :members:
   :undoc-members:
   :show-inheritance:

GEBCO
-----

.. automodule:: kadlu.geospatial.data_sources.gebco
   :members:
   :undoc-members:
   :show-inheritance:

HYCOM
-----

.. automodule:: kadlu.geospatial.data_sources.hycom
   :members:
   :undoc-members:
   :show-inheritance:

IFREMER
-------

.. automodule:: kadlu.geospatial.data_sources.ifremer
   :members:
   :undoc-members:
   :show-inheritance:

WWIII
-----

.. automodule:: kadlu.geospatial.data_sources.wwiii
   :members:
   :undoc-members:
   :show-inheritance:

|

Data Utils
----------

.. automodule:: kadlu.geospatial.data_sources.data_util
   :members:
   :undoc-members:
   :show-inheritance:

