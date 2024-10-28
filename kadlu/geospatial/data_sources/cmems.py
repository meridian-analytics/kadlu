"""
    API for Copernicus Marine Environment Monitoring Service (CMEMS) Global Ocean Physical datasets.

    https://data.marine.copernicus.eu/products

    Currently, implements functions for fetching data from the following datasets,

        *   Global Ocean Physical Analysis and Forecasting Product
            GLOBAL_ANALYSISFORECAST_PHY_001_024
            https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description
            Temporal coverage: Past 5 years + present year up to 10 days into future.
            Spatial resolution: 1/12 degrees
            Temporal resolution: 1 hour
"""

import os
import logging
from os.path import isfile
from glob import glob
from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma
from kadlu import index
from kadlu.geospatial.data_sources.data_util import (
    database_cfg,
    dt_2_epoch,
    logmsg,
    logmsg_nodata,
    storage_cfg,
    str_def,
)
import copernicusmarine 
import netCDF4

"""
    Names of the tables that will be created in the kadlu geospatial.db database for storing CMEMS data.
"""
cmems_tables = {  #kadlu name: CMEMS name
    "water_u": "utotal", 
    "water_v": "vtotal",
}


# subdirectory where downloaded NetCDF files will be stored
CMEMS_SUBDIR = "./cmems"


def data_path():
    """ Returns path to directory where CMEMS NetCDF files are stored """
    return os.path.join(storage_cfg(), CMEMS_SUBDIR)


def filename(var, east, west, south, north, start):
    """ Generates standardized filename for the downloaded NetCDF file """
    fname = f"{var}_{east}E{west}W{south}S{north}N_{start.strftime('%Y%m%d')}.nc"
    return fname


def initdb():
    """ Create tables in kadlu's geospatial.db database for storing CMEMS data"""
    conn, db = database_cfg()
    for var in cmems_tables.keys():
        db.execute(f'CREATE TABLE IF NOT EXISTS {var}'
                   '( val     REAL    NOT NULL, '
                   '  lat     REAL    NOT NULL, '
                   '  lon     REAL    NOT NULL, '
                   '  time    INT     NOT NULL, '
                   '  source  TEXT    NOT NULL) ')
        db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                   f'idx_{var} on {var}(time, lon, lat, val, source)')
    conn.close()


def clear_cache_cmems():
    """ Removes all NetCDF files in the subdirectory `cmems` within the Kadlu storage directory"""
    logger = logging.getLogger("kadlu")

    # path to folder where Kadlu stores data
    dir_path = data_path()

    if not os.path.exists(dir_path):
        warn_msg = f"Failed to clear CMEMS cache. Kadlu data storage directory not found at {dir_path}."
        logger.warning(warn_msg)
        return
    
    # find all ERA5 grib files
    paths = glob(os.path.join(dir_path, "*.nc"))    

    if len(paths) == 0:
        info_msg = f"CMEMS cache is empty."
        logger.info(info_msg)
        return

    # get their size and remove them
    bytes = 0
    for path in paths:
        bytes += os.path.getsize(path)
        os.remove(path)

    info_msg = f"Emptied CMEMS cache (deleted {len(paths)} files, {bytes/1E6:.1f} MB)"
    logger.info(info_msg)


def fetch_cmems(var, *, west, east, south, north, start, **_):
    """ Fetch global CMEMS data for specified variable, geographic region, and time range.

        Downloads 24-hours of global data on the specified day, and saves these data to 
        a *.nc file in the kadlu data storage directory.

        The *.nc file can be deleted manually by calling the `clear_cache_cmems` function 
        to save disk space, if necessary.

        Only data within the specified geographic boundaries (`west`, `east`, `south`, `north`) 
        are inserted into the kadlu geospatial.db database.

        args:
            var: string
                The variable short name of desired wave parameter according to CMEMS docs. 
            west,east,south,north: float
                Geographic boundaries of the data request
            start: datetime.datetime
                UTC date of the data request. 24-hours of data will be fetched.
                
        return:
            True if new data was fetched, else False
    """
    logger = logging.getLogger("kadlu")

    # variable mapping
    if var in cmems_tables:
        var_cmems = cmems_tables[var]
    
    else:
        err_msg = f"Invalid variable `{var}` for data source CMEMS; valid options are: {list(cmems_tables.keys())}"
        raise ValueError(err_msg)

    # time window
    start = datetime(start.year, start.month, start.day)
    end = start + timedelta(days=1)

    # filename
    fname = filename(var_cmems, east, west, south, north, start)

    # full path
    target = os.path.join(data_path(), fname)

    if isfile(target):
        return

    logger.info(f'fetching {target}...')

    # form request
    request = dict(
        dataset_id = 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
        variables = [var_cmems],
        minimum_longitude = west,
        maximum_longitude = east,
        minimum_latitude = south,
        maximum_latitude = north,
        start_datetime = start,
        end_datetime = end,
        minimum_depth = 0,
        maximum_depth = 1,
        output_filename = fname,
        output_directory = data_path(),
        service = "arco-geo-series",
        force_download = True,
    )

    # submit request
    copernicusmarine.subset(**request)

    # open downloaded file
    assert isfile(target)
    nc = netCDF4.Dataset(target)

    # load data into memory (as masked Numpy arrays)
    vals = nc.variables[var_cmems][:,0,:,:]  #(time,depth,lat,lon)
    lats = nc.variables["latitude"][:]  #degrees
    lons = nc.variables["longitude"][:]  #degrees, -180 to +180
    times = nc.variables["time"][:]  #hours since 1950-01-01

    # expand dimesionality
    def expand(a, shape, axis):
        """ Helper function for expanding dimensionality of array """
        dims = list(shape)
        del dims[axis]
        for dim in dims:
            new_shape = a.shape + (dim,)
            a = np.expand_dims(a, axis=-1)  #adds new dimension of shape (1,)
            a = np.broadcast_to(a, new_shape)   #expands the new dimension to the desired size (by replicating the array)

        a = np.moveaxis(a, [0], [axis]) #moves the original dimension to the desired position
        return a

    times = expand(times, vals.shape, 0)
    lats = expand(lats, vals.shape, 1)
    lons = expand(lons, vals.shape, 2)

    # remove masked values
    idx = ma.getmask(vals)
    vals = vals[~idx]
    times = times[~idx]
    lats = lats[~idx]
    lons = lons[~idx]

    vals = vals.flatten()
    times = times.flatten()
    lats = lats.flatten()
    lons = lons.flatten()

    # collect data in a list (values, lats, lons, times, source)
    # OBS: the `time` column in the SQL database has type INT so 
    #   we cannot store half-integer hours. therefore we subtract 
    #   30 minutes at load-time instead to center data within bin
    dt0 = datetime(year=1950,month=1,day=1)
    data = np.array([
        vals, 
        lats, 
        lons, 
        dt_2_epoch([dt0 + timedelta(hours=float(t)) for t in times]),  
        ['cmems' for _ in vals],
    ])

    # SQL table name
    table = var

    # perform the insertion into the database
    initdb()
    conn, db = database_cfg()
    n1 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    db.executemany(
        f"INSERT OR IGNORE INTO {table} "
        f"VALUES (?,?,?,CAST(? AS INT),?)", data.T)
    n2 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    db.execute("COMMIT")
    conn.commit()
    conn.close()

    # log message
    kwargs = dict(
        south = south,
        west = west,
        north = north,
        east = east,
        start = start,
        end = end     
    )
    logmsg('cmems', var, (n1, n2), **kwargs)
    
    return True


def load_cmems(var, *, west, east, south, north, start, end, fetch=True, **_):
    """ Load CMEMS data from local geospatial.db database

        Args:
            var: str
                Variable to be fetched
            west,east,south,north: float
                Geographic boundaries of the data request
            start: datetime.datetime
                UTC start time for the data request.
            end: datetime.datetime
                UTC end time for the data request.
            fetch: bool
                If the data have not already been downloaded and inserted into 
                Kadlu's local geospatial database, fetch data from the Copernicus 
                Climate Data Store (CDS) automatically using the CDS API. Default is True.
                
        Returns:
            values:
                values of the fetched var
            lat:
                y grid coordinates
            lon:
                x grid coordinates
            epoch:
                timestamps in epoch hours since jan 1 2000
    """
    if fetch:
        # Check local database for data.
        # Fetch data from Copernicus API, if missing.
        with index(storagedir=storage_cfg(),
                west=west,
                east=east,
                south=south,
                north=north,
                start=start,
                end=end) as fetchmap:
            fetchmap(callback=fetch_cmems, var=var)

    # connect to local database
    conn, db = database_cfg()

    # table name in local database
    table = var

    # check if the table exists
    rows = db.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchall()
    table_exists = len(rows) > 0

    if table_exists:
        # query local database for data
        sql_query_list = [f"SELECT * FROM {table} WHERE lat >= ?", 'lat <= ?', 'lon >= ?', 'lon <= ?', 'time >= ?', 'time <= ?']
        sql_query = ' AND '.join(sql_query_list) + ' ORDER BY time, lat, lon ASC'
        sql_values = tuple(
            map(str, [south, north, west, east, dt_2_epoch(start), dt_2_epoch(end)])
        )
        db.execute(sql_query, sql_values)
        rowdata = np.array(db.fetchall(), dtype=object).T

    else:
        rowdata = []

    # close database connection
    conn.close()

    # if no data was found, return empty arrays and log info
    if len(rowdata) == 0:
        logmsg_nodata(
            'cmems', var,
            west=west, east=east, south=south, north=north,
            start=start, end=end
        )
        return np.array([[], [], [], []])

    val, lat, lon, epoch, source = rowdata
    data = np.array((val, lat, lon, epoch), dtype=float)

    # OBS: the `time` column in the SQL database has type INT so 
    #   we cannot store half-integer hours. therefore we subtract 
    #   30 minutes at load-time instead to center data within bin
    data[-1] -= 0.5

    return data


class Cmems():
    """ Collection of module functions for fetching and loading.
    
        The functions return (values, lat, lon, epoch) numpy arrays with 
        shape (num_points, 4) where epoch is the number of hours since 2000-01-01.
    """

    def load_water_u(self, **kwargs):
        return load_cmems('water_u', **kwargs)

    def load_water_v(self, **kwargs):
        return load_cmems('water_v', **kwargs)

    def __str__(self):
        info = '\n'.join([
            "Copernicus Marine Environment Monitoring Service (CMEMS)",
            "\thttps://data.marine.copernicus.eu/products",
        ])
        args = "(south, north, west, east, start, end)"
        return str_def(self, info, args)
