.. _installation_instructions:

Install
=======

Kadlu runs on the most recent stable version of Python 3.

 1. ensure that GCC is installed. On Ubuntu, this can be done by installing the build-essential package if its not installed already
    ```bash
    sudo apt update 
    sudo apt install build-essential
    ```

 2. Install Kadlu from pip
    ```bash
    python -m pip install kadlu
    ```

Configuration
=============

 1. Optionally set the storage directory

    Kadlu allows configuration for where data is stored on your machine. By default, a folder 'kadlu_data' will be created in the home directory. To specify a custom location, run the following: ::

      kadlu.storage_cfg(setdir='/specify/desired/path/here/')

 2. Optionally add an API token for fetching ERA5 data

   Kadlu uses ECMWF's Era5 dataset as one of the data sources for wave height/direction/period and wind speed data.
   By default, an API token is included with kadlu, but if you intend to make frequent use of the Era5 dataset, please consider obtaining your own token.
   This can be obtained by registering an account at [Copernicus API](https://cds.climate.copernicus.eu/api-how-to). Once logged in, your token and URL will be displayed on the aforementioned webpage under heading 'Install the CDS API key'.
   Additionally, you will need to accept the [Copernicus Terms of Use](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products) to activate the token.

   Configure Kadlu to use the token by executing: ::

      kadlu.era5_cfg(key="TOKEN_HERE", url="URL_HERE")

