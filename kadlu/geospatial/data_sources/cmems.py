"""
    CMEMS = Copernicus Marine Environment Monitoring Service

    https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description

    GLOBAL_ANALYSISFORECAST_PHY_001_024

    Global Ocean Physical Analysis and Forecasting Product

    https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf

    Past 5 years + present year up to 10 days into future.
    1/12 degree resolution
    1-hour time bins.

    docs: https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset


    Notes: there is also the Global Ocean Physics Reanalysis dataset:
    https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
    with temporal coverage from 1993 up to 3 months before present.
"""

import os
import logging
from os.path import isfile
from glob import glob
from datetime import datetime, timedelta
import numpy as np
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
    key: kadlu name
    value: cmems name
"""
cmems_tables = {
    "water_u": "utotal", 
    "water_v": "votal",
}


CMEMS_SUBDIR = "./cmems"


def data_path():
    """ Returns path to directory where CMEMS NetCDF files are stored """
    return os.path.join(storage_cfg(), CMEMS_SUBDIR)


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
        var_name = cmems_tables[var]
    
    else:
        err_msg = f"Invalid variable `{var}` for data source CMEMS; valid options are: {list(cmems_tables.keys())}"
        raise ValueError(err_msg)

    # time window
    start = datetime(start.year, start.month, start.day)
    end = start + timedelta(days=1)

    # filename
    fname = f"{var_name}_{east}E{west}W{south}S{north}N_{start.strftime("%Y%m%d")}.nc"

    # full path
    target = os.path.join(data_path(), fname)

    if isfile(target):
        return


    logger.info(f'fetching {target}...')

    # form request
    request = dict(
        dataset = 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
        variables = [var_name],
        minimum_longitude=east,
        maximum_longitude=west,
        minimum_latitude=south,
        maximum_latitude=north,
        start_datetime=start,
        end_datetime=end,
        minimum_depth=0,
        maximum_depth=1,
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
    values = nc.variables[var_name][:,:,:,:]

    # SQL table name
    table = var

    #latitude  (-90,90)
    #longitude (-180,180)
    #time: units: hours since 1950-01-01
    #utotal: (time,1,lat,lon)


    #TODO: remove masked values from data array
    #  (expand dimensionality of lat,lon,time arrays to match data array, then select non-masked indices similar to era5)

    # load the data from the *.grb2 file and insert it into the database
    grb = pygrib.open(target)
    data = np.array([[], [], [], [], []])
    table = var[4:] if var[0:4] == '10m_' else var

    # process data 'messages' 
    for msg in grb:
        # timestamp
        dt = msg.validDate

        # for forecasts, 'validDate' represents the start time of the forecast (06:00 UTC or 18:00 UTC)
        # so we must add the appropriate number of hourly steps
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=85402030
        #  
        # for accumulated quantities we should subtracting 1/2 hour to get the time at the center of the 
        # forecast bin; however, since our database stores times as INTs (hours since 2000-01-01), we 
        # instead subtract 1/2 hour when the data is loaded from the database
        
        step = msg.step
        ###if msg.stepType == "accum":
        ###    step -= 0.5

        dt += timedelta(seconds = step * 3600)

        debug_msg = f"[ERA5] Processing message with timestamp {dt} (step={step}) ..."
        logger.debug(debug_msg)

        # read grib data (value, lat, lon)
        z, y, x = msg.data()
        if np.ma.is_masked(z):
            z2 = z[~z.mask].data
            y2 = y[~z.mask]
            x2 = x[~z.mask]
        else:  # wind data has no mask
            z2 = z.reshape(-1)
            y2 = y.reshape(-1)
            x2 = x.reshape(-1)

        # ERA5 uses longitude values in the range [0;360] referenced to the Greenwich Prime Meridian
        # convert to longitude to [-180;180]
        x3 = ((x2 + 180) % 360) - 180

        # index coordinates, select query range subset
        xix = np.logical_and(x3 >= west, x3 <= east)
        yix = np.logical_and(y2 >= south, y2 <= north)
        idx = np.logical_and(xix, yix)

        # collect data in a list (values, lats, lons, times, source)
        msg_data = [
            z2[idx], y2[idx], x3[idx], dt_2_epoch([dt for i in z2[idx]]), ['era5' for i in z2[idx]]
        ]

        # aggregate data
        data = np.hstack((data, msg_data))

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
    logmsg('era5', var, (n1, n2), **kwargs)
    
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
    return np.array((val, lat, lon, epoch), dtype=float)


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
