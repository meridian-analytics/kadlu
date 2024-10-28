# Welcome to Kadlu, a Python package for modelling ocean acoustics

Kadlu was developed for the purpose of modelling noise due to waves and rain in shallow coastal 
waters, but contains tools useful for many other underwater acoustics modelling tasks. 

Kadlu is written in Python and utilizes a number of powerful software packages 
including [NumPy](https://numpy.org/), [HDF5](https://www.hdfgroup.org/), 
[NetCDF-4](https://www.unidata.ucar.edu/software/netcdf/), and [SQLite](https://www.sqlite.org/index.html).
It is licensed under the [GNU GPLv3 license](https://www.gnu.org/licenses/) 
and hence freely available for anyone to use and modify. The project is hosted on GitHub at 
[github.com/meridian-analytics/kadlu](https://github.com/meridian-analytics/kadlu).

Kadlu was born out of the [MERIDIAN](http://meridian.cs.dal.ca/) project (2018-2023) at the 
[Institute for Big Data Analytics](https://bigdata.cs.dal.ca/) at Dalhousie University with the 
support and assistance of David Barclay and Calder Robinson, both from the Department of Oceanography 
at Dalhousie University.

Kadlu provides functionalities that automate the process of fetching and interpolating 
environmental data necessary to model ocean ambient noise levels (bathymetry, water temperature 
and salinity, wave height, wind speed, etc.). It also includes various routines that allow 
accurate estimates of noise source levels and transmission losses in realistic ocean environments.
You can find more information about the technical aspects of how sound propagation is modelled in 
Kadlu in [this note](docs/source/_static/kadlu_sound_propagation_note.pdf).

The intended users of Kadlu are researchers and students in underwater acoustics working with ambient noise modeling. 
While Kadlu comes with complete documentation and comprehensive step-by-step tutorials, some familiarity with Python and 
especially the NumPy package would be beneficial. A basic understanding of 
the physical principles of underwater sound propagation would also be an advantage.

For more information, please consult [Kadlu's Documentation Page](https://meridian-analytics.github.io/kadlu/).


## Installation

Kadlu runs on the most recent stable version of Python 3. 

 1. ensure that GCC is installed. On Ubuntu, this can be done by installing the build-essential package if its not installed already
    ```bash
    sudo apt update 
    sudo apt install build-essential
    ```

 2. Install Kadlu using pip
    ```bash
    python -m pip install kadlu
    ```


## Configuration


#### Optionally set the storage directory

Kadlu allows configuration for where data is stored on your machine. By default, a folder 'kadlu_data' will be created in the home directory. To specify a custom location, run the following:

```python
import kadlu
kadlu.storage_cfg(setdir='/specify/desired/path/here/')
```


#### Optionally add an API token for fetching ERA5 data

Kadlu uses ECMWF's ERA5 dataset as one its main sources of environmental data, e.g., for wind, waves and solar irradiance. 
You will need to obtain your own access token to download these data. 
This can be done by registering an account with ECMWF and the Copernicus [Climate Data Store (CDS)](https://cds-beta.climate.copernicus.eu/). 
Once logged in, your access token, comprised of an URL address and a secret key, will be displayed on 
the [Copernicus webpage](https://cds-beta.climate.copernicus.eu/how-to-api). 
Following the instructions, copy the URL and the key to the file `$HOME/.cdsapirc`. 
Finally, you will need to [accept the ERA5 terms of use](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) to activate the token.


## Jupyter notebook tutorials

 1. [The Ocean Module](docs/source/tutorials/ocean_module_tutorial/ocean_module_tutorial.ipynb)

 2. [Fetch and Load Environmental Data](docs/source/tutorials/fetch_load_tutorial/fetch_load_tutorial.ipynb)

 3. [Interpolate Multi-Dimensional Data](docs/source/tutorials/interp_tutorial/interp_tutorial.ipynb)

 4. [Plot and Export Data](docs/source/tutorials/plot_export_tutorial/plot_export_tutorial.ipynb)

 5. [Transmission Loss](docs/source/tutorials/transm_loss_tutorial/transm_loss_tutorial.ipynb)


## Notes for developers

The CI pipeline caches all data pulled from external APIs by the tests. This allows for faster execution of the 
pipeline, but means that the data fetching step is not being tested. Therefore, it is advisable to periodically 
clean the runner caches manually.

To skip execution of the CI pipeline, append `-o ci.skip` to the `git push` command.

Steps for publishing a new release to PyPi:
```
python setup.py sdist bdist_wheel
twine upload dist/kadlu-X.Y.Z.tar.gz
```


## Useful resources

 *  [gsw Python package](https://github.com/TEOS-10/GSW-Python) (Python implementation of the Thermodynamic Equation of Seawater 2010)

