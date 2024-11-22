"""
    API for NOAA's GFS forecast.

    Fetches data on a regular grid with 0.25 degree spatial resolution and 1-hour temporal resolution.

    Data are only available 1 week into the past and 5 days into the future.

    Uses https://github.com/jagoosw/getgfs for fetching the data.
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
import getgfs

"""
    Names of the tables that will be created in the kadlu geospatial.db database for storing GFS data.
"""
gfs_tables = {  #kadlu name: GFS name
    "gfs_wind_u": "ugrd10m", #10 m above ground u-component of wind [m/s] 
    "gfs_wind_v": "vgrd10m", #10 m above ground v-component of wind [m/s]
    "gfs_shortwave_radiation": "dswrfsfc", #surface downward short-wave radiation flux [w/m^2]
    "gfs_longwave_radiation": "dlwrfsfc", #surface downward long-wave radiation flux [w/m^2]
}


def initdb():
    """ Create tables in kadlu's geospatial.db database for storing CMEMS data"""
    conn, db = database_cfg()
    for var in gfs_tables.keys():
        db.execute(f'CREATE TABLE IF NOT EXISTS {var}'
                   '( val     REAL    NOT NULL, '
                   '  lat     REAL    NOT NULL, '
                   '  lon     REAL    NOT NULL, '
                   '  time    INT     NOT NULL, '
                   '  source  TEXT    NOT NULL) ')
        db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                   f'idx_{var} on {var}(time, lon, lat, val, source)')
    conn.close()


def fetch_gfs(var, *, west, east, south, north, start, **_):
    """ Fetch GFS data for specified variable, geographic region, and time range.

        Fetches data on a regular grid with 0.25 degree spatial resolution and 1-hour temporal resolution.

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
    if var in gfs_tables:
        var_gfs = gfs_tables[var]
    
    else:
        err_msg = f"Invalid variable `{var}` for data source GFS; valid options are: {list(gfs_tables.keys())}"
        raise ValueError(err_msg)

    # time window
    start = datetime(start.year, start.month, start.day)
    end = start + timedelta(days=1)

    logger.info(f'fetching GFS data ...')

    # get forecast
    f = getgfs.Forecast("0p25", "1hr")

    # loop over time steps and fetch data
    vals = []
    times = []
    t = start
    while t <= end:
        t_str = t.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            res = f.get([var_gfs], t_str, f"[{south}:{north}]", f"[{west}:{east}]")
            v = res.variables[var_gfs].data
            vals.append(v[0])
            times.append(t)

        except:
            err_msg = f"Unable to fetch GFS data for {t_str} (UTC)"
            logger.error(err_msg)

        t += timedelta(hours=1)

    vals = np.stack(vals) #(time,lat,lon)

    #lat/lon coordinate arrays
    first_lat = int(np.sign(south) * (abs(south) + 0.125) / 0.25) * 0.25
    last_lat = first_lat + 0.25 * (vals.shape[1] - 1)
    lats = np.linspace(first_lat, last_lat, num=vals.shape[1])

    first_lon = int(np.sign(west) * (abs(west) + 0.125) / 0.25) * 0.25
    last_lon = first_lon + 0.25 * (vals.shape[2] - 1)
    lons = np.linspace(first_lon, last_lon, num=vals.shape[2])

    # flatten
    times, lats, lons = np.meshgrid(times, lats, lons, indexing="ij")
    times = times.flatten()
    lats = lats.flatten()
    lons = lons.flatten()
    vals = vals.flatten()

    # collect data in a list (values, lats, lons, times, source)
    data = np.array([
        vals, 
        lats, 
        lons, 
        dt_2_epoch(times),  
        ['gfs' for _ in vals],
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
    logmsg('gfs', var, (n1, n2), **kwargs)
    
    return True


def load_gfs(var, *, west, east, south, north, start, end, fetch=True, **_):
    """ Load GFS data from local geospatial.db database

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
            fetchmap(callback=fetch_gfs, var=var)

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
            'gfs', var,
            west=west, east=east, south=south, north=north,
            start=start, end=end
        )
        return np.array([[], [], [], []])

    val, lat, lon, epoch, source = rowdata
    data = np.array((val, lat, lon, epoch), dtype=float)

    return data


class Gfs():
    """ Collection of module functions for fetching and loading.
    
        The functions return (values, lat, lon, epoch) numpy arrays with 
        shape (num_points, 4) where epoch is the number of hours since 2000-01-01.
    """

    def load_wind_u(self, **kwargs):
        return load_gfs('gfs_wind_u', **kwargs)

    def load_wind_v(self, **kwargs):
        return load_gfs('gfs_wind_v', **kwargs)

    def load_irradiance(self, **kwargs):
        return load_gfs('gfs_shortwave_radiation', **kwargs)

    def __str__(self):
        info = "NOAA GFS Forecast"
        args = "(south, north, west, east, start, end)"
        return str_def(self, info, args)
