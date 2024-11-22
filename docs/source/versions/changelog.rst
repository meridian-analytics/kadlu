Change log
==========

**Version 2.4.3** (Nov 22, 2024)

 * New GFS module for fetching wind and solar data from NOAA GFS forecast

**Version 2.4.2** (Oct 28, 2024)

 * New CMEMS module for fetching data from Copernicus marine repository

**Version 2.4.1** (Sep 26, 2024)

 * Updated ERA5 module to new CDS API

**Version 2.4.0** (Sep 4, 2024)

 * Major overhaul of `geospatial.interpolation` and `geospatial.ocean` modules, including some *non-backward compatible* changes
 * Cleaned up HYCOM module and updated to use GLBy0.08 repository (2019 to present)
 * Cleaned up ERA5 module and improved documentation. 
 * Added support for fetching ERA5 solar irradiance data. 
 * Fixed issue with ERA5 forecast data receiving incorrect timestamps.
 * Added function in ERA5 module for cleaning data cache.
 * Replaced `matplotlib.cm` (deprecated) with `matplotlib.colormaps`

**Version 2.3.9** (June 7, 2024)

 * Improved debug logging
 * Allow use of nearest-neighbor interpolation for 2D data
 * Expose interpolation method in Ocean class constructor
 * Bug fix in method used for computing grid for sound speed profile interpolation

**Version 2.3.8** (May 9, 2024)

* Fixed bug in ocean module related to parsing of input data for Ocean class
* Improved error handling and logging in interpolation module in relation to irregular grids

**Version 2.3.7** (July 14, 2023)

* Fix for ocean module
* Improved logging
* Improved cross-platform support

**Version 2.3.6** (February 22, 2022)

* Update GEBCO bathymetry to latest dataset
* Updated docker image
* Fix deprecation warnings from packages
* Documentation and formatting

**Version 2.3.5** (July 2, 2021)

* Fix bathymetry callback called too many times

**Version 2.3.4** (June 23, 2021)

* Bug fix in parabolic equation module

**Version 2.3.3** (May 28, 2021)

* Update to parabolic equation
* Improved compatability
* Fix error in loading GEBCO bathymetry

**Version 2.3.2** (May 12, 2021)

* Fix bug with setting default ERA5 API token
* Fix problem with downloading bathymetry data on windows
* Remove conda dependencies

**Version 2.3.1** (May 8, 2021)

* Fix behaviour where database_cfg is initialized on import instead of runtime
* Bug fixes and improvements

**Version 2.3.0** (September 14, 2020)

* Refactored data fetching logic
* Add support for GEBCO bathymetry geotiffs
* Remove support for CHS bathymetry
* Update sphinx-build and documentation
* Update notebook tutorials
* Update docstrings
* Update log messages
* Fix off-by-one indexing error when fetching from hycom
* Fix erroneous function calls and fetching wind data in WWIII module
* Improved argument handling in ocean module

**Version 2.2.4** (August 14, 2020)

* Bug fixes

**Version 2.2.3** (August 4, 2020)

* Update to bathymetry file loading

**Version 2.2.2** (August 4, 2020)

* Bug fixes in loading scripts
* Improved testing

**Version 2.2.1** (July 21, 2020)

* Bug fix to loading data from era5 source
* Update to setup script

**Version 2.2.0** (June 26, 2020)

* Added functionality to load data from local files in raster format
* Improved build scripts
* Added default ERA5 token

**Version 2.1.3** (June 15, 2020)

* Include functionality to load from arbitrary netcdf databases for 2D data
* Scripting for automatic loading of GEBCO bathymetric data

**Version 2.1.2** (June 5, 2020)

* Add functionality to output geophony as netcdf

**Version 2.1.1** (May 28, 2020)

* Update kadlu deployment scripting

**Version 2.1.0** (May 27, 2020)

* Update kadlu configuration interface 

**Version 2.0.0** (May 8, 2020)

* Added sound propagation modelling toolset
* Improvements to automated data fetching and storage


**Version 1.0.0** (March 16, 2020)

* First release
