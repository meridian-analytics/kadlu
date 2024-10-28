import os
import logging


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(format='%(message)s',
                    level=LOGLEVEL,
                    datefmt='%Y-%m-%d %I:%M:%S')

from .index import index, parallelindex

# data utils
from .geospatial.data_sources.data_util import (
    database_cfg,
    dt_2_epoch,
    epoch_2_dt,
    ext,
    index_arr,
    reshape_2D,
    reshape_3D,
    reshape_3D_gridded,
    storage_cfg,
    verbosity,
)
from .geospatial.data_sources.source_map import source_map
from .geospatial.data_sources.source_map import default_val as defaults
from .geospatial.data_sources.load_from_file import load_netcdf, load_raster

# data sources
from .geospatial.data_sources.era5 import Era5 as era5
from .geospatial.data_sources.gebco import Gebco as gebco
from .geospatial.data_sources.hycom import Hycom as hycom
from .geospatial.data_sources.wwiii import Wwiii as wwiii
from .geospatial.data_sources.source_map import load_map

# acoustic and environmental processing
from .geospatial.ocean import Ocean
from .sound.geophony import *
from .sound.parabolic_equation import *
from .sound.sound_speed import *

# file testing for files in kadlu_data/testfiles/
from .tests.geospatial.data_sources.test_files import test_files

# plotting tools
from .plot_util import (
    #        animate,
    #        plot2D,
    plot_transm_loss_horiz,
    plot_transm_loss_vert,
)


def load(source, var, **kwargs):
    """ automated fetching and loading from web sources

        args
            source: string
                view the list of available sources and variables with ``print(kadlu.source_map)``
            var: string
                view the list of available sources and variables with ``print(kadlu.source_map)``
            kwargs: dictionary
                dict containing boundary coordinates. example:
                ``kwargs=dict(south=44.25, west=-64.5, north=44.70, east=-63.33, top=0, bottom=5000, start=datetime(2015, 3, 1), end=datetime(2015, 3, 1, 12))``

        returns: np.ndarray
            numpy arrays containing data ordered by val, lat, lon, [time, depth].
            times are in epoch format
    """

    source, var = source.lower(), var.lower()
    if var == 'bathy' or var == 'depth' or var == 'elevation':
        var = 'bathymetry'

    if var == 'temp': 
        var = 'temperature'

    loadkey = f'{var}_{source}'
    assert loadkey in load_map.keys(), f'error: invalid source or variable. valid options include: \n\n'\
            f'{list(f.rsplit("_", 1)[::-1] for f in load_map.keys())}\n\n'\
            f'for more info, print(kadlu.source_map)'

    return load_map[loadkey](**kwargs)


def load_file(filepath, **kwargs):
    """ loading from local files. currently supports geotiff raster formats as well as certain netcdf databases

        args
            filepath: string
                full path directory and filename of data to load

            kwargs: dictionary
                dict containing coordinate bounds, e.g. north, south,
                east, west, top, bottom, start, end
                note that times are in datetime format

        returns an ND numpy array. arrays ordered by:
            val, lat, lon, [time, depth]
            times are in epoch format
    """
    assert os.path.isfile(filepath), f'error: could not find {filepath}'

    if ext(filepath, ('.nc', )):
        return load_netcdf(filepath, **kwargs)

    elif ext(filepath, (
            '.tif',
            '.tiff',
            '.gtiff',
    )):
        return load_raster(filepath, **kwargs)

    else:
        assert False, f'error {filepath}: unknown file format - currently only .nc and .tif formats are accepted'
