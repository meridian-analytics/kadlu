"""
    data source:
        https://www.hycom.org/dataserver/gofs-3pt1/analysis        

    web interface for manual hycom data retrieval:
        https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0.html

    GLBy0.08/expt_93.0 
        Time range: Dec 4th, 2018 to  Present        
        Longitude convention: 0 to +360 degrees
        Depth coordinates are positive below the sea surface
        Times are in epoch hours since 2000-01-01 00:00 UTC

    API limitations:
        * Do not subset more than 1 day at a time using ncss.hycom.org
        * Do not use more than 2 concurrent connections per IP address when downloading data from ncss.hycom.org

"""

import logging
import requests
from functools import reduce
from datetime import datetime, timedelta

import numpy as np

from kadlu import index
from kadlu.geospatial.data_sources.data_util import (
    database_cfg,
    dt_2_epoch,
    fmt_coords,
    fmt_time,
    index_arr,
    logmsg,
    logmsg_nodata,
    storage_cfg,
    str_def,
)

# HYCOM data download URL
hycom_src = "http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

# Specify which variable will be downloaded
hycom_tables = [
    'hycom_salinity', 'hycom_water_temp', 'hycom_water_u', 'hycom_water_v'
]

# Helper function
slices_str = lambda var, slices, steps=(
    1, 1, 1, 1
): f"{var}{''.join(map(lambda tup, step : f'[{tup[0]}:{step}:{tup[1]}]', slices, steps))}"


def initdb():
    """ Create tables in kadlu's geospatial.db database for storing HYCOM data"""
    conn, db = database_cfg()

    for var in hycom_tables:
        db.execute(f'CREATE TABLE IF NOT EXISTS {var}'
                   '( val     REAL NOT NULL,'
                   '  lat     REAL NOT NULL,'
                   '  lon     REAL NOT NULL,'
                   '  time    INT  NOT NULL,'
                   '  depth   INT  NOT NULL,'
                   '  source  TEXT NOT NULL )')

        db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                   f'idx_{var} on {var}(time, lon, lat, depth, val, source)')
    conn.close()


def fetch_grid(**_):
    """ Download HYCOM lat/lon/time/depth arrays for grid indexing.
        
        Times are formatted as epoch hours since 2000-01-01 00:00.

        Returns:
            lat, lon, epoch, depth: numpy array
                The coordinate arrays
    """
    logger = logging.getLogger("kadlu")

    # fetch lat/lon arrays

    logger.info("[HYCOM] Fetching lat and lon indices ...")

    url = f"{hycom_src}.ascii?lat%5B0:1:4250%5D,lon%5B0:1:4499%5D"
    
    logger.debug(f"[HYCOM] data request URL: {url}")
    grid_netcdf = requests.get(url, timeout=450)
    assert (grid_netcdf.status_code == 200)

    meta, data = grid_netcdf.text.split\
    ("---------------------------------------------\n")
    lat_csv, lon_csv = data.split("\n\n")[:-1]
    lat = np.array(lat_csv.split("\n")[1].split(", "), dtype=float)
    lon = np.array(lon_csv.split("\n")[1].split(", "), dtype=float)

    logger.debug(f"[HYCOM] lon: {lon}")

    # fetch time array

    logger.info(f"[HYCOM] Fetching time index ...")

    url = f"{hycom_src}.ascii?time"
    logger.debug(f"[HYCOM] data request URL: {url}")
    time_netcdf = requests.get(url, timeout=450)
    assert (time_netcdf.status_code == 200)
    meta, data = time_netcdf.text.split\
    ("---------------------------------------------\n")
    csv = data.split("\n\n")[:-1][0]
    epoch = np.array(csv.split("\n")[1].split(', ')[1:], dtype=float)

    # fetch depth array

    logger.info(f"[HYCOM] Fetching depth index ...")

    depth = np.array([
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0,
        40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 200.0,
        250.0, 300.0, 350.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
        1250.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0
    ])

    return lat, lon, epoch, depth


def load_grid():
    """ Put spatial grid into memory """
    with index(storagedir=storage_cfg(), bins=False, store=True) as fetchmap:
        return fetchmap(callback=fetch_grid, seed='hycom grids')[0]


class Hycom():
    """ Collection of module functions for fetching and loading HYCOM data.

        Attributes:
            lat, lon: arrays
                Lat/lon coordinates.
            epoch: array
                Time coordinates. Times are formatted as epoch hours since 2000-01-01 00:00
            depth: array
                Depth coordinates.
    """

    def __init__(self):
        self.ygrid, self.xgrid, self.epoch, self.depth = None, None, None, None
        self.logger = logging.getLogger("kadlu")
        self.name = "HYCOM"

    def load_salinity(self, **kwargs):
        return self.load_hycom('salinity', kwargs)

    def load_temp(self, **kwargs):
        return self.load_hycom('water_temp', kwargs)

    def load_water_u(self, **kwargs):
        return self.load_hycom('water_u', kwargs)

    def load_water_v(self, **kwargs):
        return self.load_hycom('water_v', kwargs)

    def callback(self, var, max_attempts=3, **kwargs):
        """ Builds indices for query, fetches data from HYCOM, and inserts into local database.

            Note: Null/NaN values are removed before the data is inserted into the local database.
            Null/NaN values occur when the grid overlaps with land or extends below the seafloor. 

            TODO: Add download progress bar, e.g., using the approach described here:
                https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests

            Args:
                var: string
                    Variable to be fetched. complete list of variables here
                    https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
                max_attempts: int
                    Maximum number of request attempts. Default is 3. Each request has a timeout of 120 s.
                kwargs: dict
                    boundaries as keyword arguments
        """
        # build request indexes
        south = kwargs["south"]
        north = kwargs["north"]
        west = kwargs["west"]
        east = kwargs["east"]
        top = kwargs["top"]
        bottom = kwargs["bottom"]
        start_epoch = dt_2_epoch(kwargs['start'])
        end_epoch = dt_2_epoch(kwargs['end'])

        haystack = np.array([self.epoch, self.depth, self.ygrid, self.xgrid], dtype=object)
        needles1 = np.array([start_epoch, top, south, west])
        needles2 = np.array([end_epoch, bottom, north, east])
        
        slices = list(
            zip(map(index_arr, needles1, haystack),
                map(index_arr, needles2, haystack))
        )
        
        assert reduce(
            np.multiply, map(lambda s: s[1] - s[0] + 1, slices)
        ) > 0, f"0 records available within query boundaries: {kwargs}"

        # generate data download request for HYCOM server
        url = f"{hycom_src}.ascii?{slices_str(var, slices)}"

        self.logger.debug(f"[{self.name}] data request URL: {url}")

        # make several download attempts until successful
        counter = 0
        timeout = 120
        code = None
        while counter < max_attempts:
            counter += 1
            self.logger.debug(f"[{self.name}] Requesting data from Hycom server ... (attempt no. {counter}/{max_attempts})")
            try:
                payload_netcdf = requests.get(url, stream=True, timeout=timeout, verify=True)
                code = payload_netcdf.status_code
                if code == 200:
                    debug_msg = f"[{self.name}] Data request was successful"
                    self.logger.debug(debug_msg)
                    break

                else:
                    debug_msg = f"[{self.name}] Could not access Hycom server; server returned status code {code}"
                    self.logger.debug(debug_msg)

            except requests.exceptions.ReadTimeout:
                debug_msg = f"[{self.name}] Request to Hycom server timed out"
                self.logger.debug(debug_msg)
 
        assert code == 200, f"[{self.name}] Data request unsuccesful. Could not access Hycom server."

        meta, data = payload_netcdf.text.split("---------------------------------------------\n")

        # parse response into numpy array
        arrs = data.split("\n\n")[:-1]
        shape_str, payload = arrs[0].split("\n", 1)
        shape = tuple(
            [int(x) for x in shape_str.split("[", 1)[1][:-1].split("][")])
        cube = np.ndarray(shape, dtype=float)

        for arr in payload.split("\n"):
            ix_str, row_csv = arr.split(", ", 1)
            a, b, c = [int(x) for x in ix_str[1:-1].split("][")]
            cube[a][b][c] = np.array(row_csv.split(", "), dtype=int)

        # build coordinate grid, populate values, adjust scaling, remove nulls
        flatten = reduce(np.multiply, map(lambda s: s[1] - s[0] + 1, slices))
        add_offset = 20 if 'salinity' in var or 'water_temp' in var else 0
        null_value = -10 if 'salinity' in var or 'water_temp' in var else -30

        grid = np.array([
            (None, y, x, t, d, 'hycom')
            for t in self.epoch[slices[0][0]:slices[0][1] + 1]
            for d in self.depth[slices[1][0]:slices[1][1] + 1]
            for y in self.ygrid[slices[2][0]:slices[2][1] + 1]
            for x in self.xgrid[slices[3][0]:slices[3][1] + 1]
        ])
        grid[:, 0] = np.reshape(cube, flatten) * 0.001 + add_offset
        grid = grid[grid[:, 0] != null_value]

        # insert into db
        initdb()
        conn, db = database_cfg()
        n1 = db.execute(f"SELECT COUNT(*) FROM hycom_{var}").fetchall()[0][0]
        db.executemany(
            f"INSERT OR IGNORE INTO hycom_{var} VALUES "
            "(?, ?, ?, CAST(? AS INT), CAST(? AS INT), ?)", grid)
        n2 = db.execute(f"SELECT COUNT(*) FROM hycom_{var}").fetchall()[0][0]
        db.execute("COMMIT")
        conn.commit()
        conn.close()

        logmsg('hycom', var, (n1, n2), **kwargs)
        return

    def fetch_hycom(self, var, kwargs, max_attempts=3):
        """ Fetch data from the HYCOM server.

            Kadlu's `index` class is used for 'binning' the data requests into 
            requests that span 1 degree in lat/lon, 24 hours in time (1 day), 
            and 0-5000 m in depth.

            A data request that spans multiple lat/lon degrees, multiple days, or 
            includes depths greater than 5000m is split into multiple such 'binned' 
            requests.

            Conversely, smaller data requests are 'inflated' to the size of one 'bin'.

            Args:
                var: string
                    Variable to be fetched. complete list of variables here
                    https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
                kwargs: dict
                    boundaries as keyword arguments
                max_attempts: int
                    Maximum number of request attempts. Default is 3. Each request has a timeout of 120 s.
        """
        boundary_keys = ['west', 'east', 'south', 'north', 'top', 'bottom', 'start', 'end']

        kwargs = kwargs.copy()

        # input validation
        assert kwargs['start'] <= kwargs['end']
        assert kwargs['start'] > datetime(2018, 12, 4), f"[{self.name}] Historical data not available before 2018-12-04"
        assert kwargs['end'] < datetime.now(), f"[{self.name}] Forecast data not available"
        assert kwargs['south'] <= kwargs['north']
        assert kwargs['top'] <= kwargs['bottom']

        if self.ygrid is None:
            self.ygrid, self.xgrid, self.epoch, self.depth = load_grid()

        # convert longitudes from [-180;180] to [0;360]
        if kwargs['east'] < 0:
            kwargs['east'] = 360. + kwargs['east']

        if kwargs['west'] < 0:
            kwargs['west'] = 360. + kwargs['west']

        # if query spans meridian, make two seperate fetch requests
        if kwargs['west'] > kwargs['east']:
            self.logger.debug(f'[{self.name}] splitting request')
            argsets = [kwargs.copy(), kwargs.copy()]
            argsets[0]['east'] = self.xgrid[-1]
            argsets[1]['west'] = self.xgrid[0]
        else:
            argsets = [kwargs]

        for argset in argsets:
            args = {k: v for k, v in argset.items() if k in boundary_keys}

            self.logger.info(f'[{self.name}] Fetching {var} in region {fmt_coords(args)} for time period {fmt_time(args)}')

            with index(storagedir=storage_cfg(),
                       bins=True,
                       dx=1,
                       dy=1,
                       dz=5000,
                       dt=timedelta(hours=24),
                       **args) as fetchmap:
                fetchmap(callback=self.callback, var=var, max_attempts=max_attempts)

        return True

    def load_hycom(self, var, kwargs):
        """ Load HYCOM data from local database.

            If data is not present, attempts to fetch it from the HYCOM server.

            Although HYCOM uses the 0 to +360 degree longitude convention, the longitude 
            coordinates returned by this method adhere to the -180 to +180 convention 
            used everywhere else in Kadlu.

            Args:
                var:
                    Variable to be fetched. complete list of variables here
                    https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
                south, north: float
                    ymin, ymax coordinate values. range: -90, 90
                kwargs: dict (boundaries as keyword arguments)
                    west, east: float
                        xmin, xmax coordinate values. range: -180, 180
                    start, end: datetime
                        temporal boundaries in datetime format

            Returns:
                values: array
                    values of the fetched variable
                lat: array
                    y grid coordinates
                lon: array
                    x grid coordinates
                epoch: array
                    timestamps in epoch hours since jan 1 2000
                depth: array
                    measured in meters
        """
        kwargs = kwargs.copy()

        # check if grids are initialized
        if self.ygrid is None:
            self.ygrid, self.xgrid, self.epoch, self.depth = load_grid()

        # convert from [-180;180] to [0;360]
        if kwargs['east'] < 0:
            kwargs['east'] = 360. + kwargs['east']

        if kwargs['west'] < 0:
            kwargs['west'] = 360. + kwargs['west']

        # recursive function call for queries spanning meridian
        if kwargs['west'] > kwargs['east']:
            kwargs1 = kwargs.copy()
            kwargs2 = kwargs.copy()
            kwargs1['west'] = self.xgrid[0]
            kwargs2['east'] = self.xgrid[-1]
            return np.hstack(
                (self.load_hycom(var, kwargs1), self.load_hycom(var, kwargs2)))

        # check for missing data
        self.fetch_hycom(var, kwargs, max_attempts=3)

        # validate and execute query
        assert 8 == sum(
            map(lambda kw: kw in kwargs.keys(), [
                'south', 'north', 'west', 'east', 'start', 'end', 'top',
                'bottom'
            ])), 'malformed query'

        assert kwargs['start'] <= kwargs['end']

        # prepare SQL query
        sql_query = ' AND '.join([
            f"SELECT * FROM hycom_{var} WHERE lat >= ?", 'lat <= ?',
            'lon >= ?', 'lon <= ?', 'time >= ?', 'time <= ?', 'depth >= ?',
            'depth <= ?', "source == 'hycom' "
        ]) + 'ORDER BY time, depth, lat, lon ASC'

        sql_values = [
                kwargs['south'], kwargs['north'], kwargs['west'], kwargs['east'],
                dt_2_epoch(kwargs['start']), dt_2_epoch(kwargs['end']), 
                kwargs['top'], kwargs['bottom']
        ]
        sql_values = tuple(map(str, sql_values))

        # query the local database
        conn, db = database_cfg()
        db.execute(sql_query, sql_values)
        rowdata = np.array(db.fetchall(), dtype=object).T

        conn.close()

        if len(rowdata) == 0:
            logmsg_nodata('hycom', var, **kwargs)
            return np.array([[], [], [], [], []])
        
        # convert longitude from [0;360] to [-180;180]
        lon = rowdata[2]
        lon[lon > 180] = lon[lon > 180] - 360
        rowdata[2] = lon

        return rowdata[0:5].astype(float)

    def load_water_uv(self, **kwargs):
        """ Load water speed, computed as sqrt(vu^2 + vv^2)

            If data is not present, attempts to fetch it from the HYCOM server.

            Args:
                var:
                    Variable to be fetched. complete list of variables here
                    https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
                south, north: float
                    ymin, ymax coordinate values. range: -90, 90
                kwargs: dict (boundaries as keyword arguments)
                    west, east: float
                        xmin, xmax coordinate values. range: -180, 180
                    start, end: datetime
                        temporal boundaries in datetime format

            Returns:
                values: array
                    Water speed values, in m/s
                lat: array
                    y grid coordinates
                lon: array
                    x grid coordinates
                epoch: array
                    timestamps in epoch hours since jan 1 2000
                depth: array
                    measured in meters
        """
        kwargs = kwargs.copy()

        # convert from [-180;180] to [0;360]
        if kwargs['east'] < 0:
            kwargs['east'] = 360. + kwargs['east']

        if kwargs['west'] < 0:
            kwargs['west'] = 360. + kwargs['west']

        self.fetch_hycom('water_u', kwargs, max_attempts=3)
        self.fetch_hycom('water_v', kwargs, max_attempts=3)

        sql_query = ' AND '.join(['SELECT hycom_water_u.val, hycom_water_u.lat, hycom_water_u.lon, hycom_water_u.time, hycom_water_u.depth, hycom_water_v.val FROM hycom_water_u '\
                'INNER JOIN hycom_water_v '\
                'ON hycom_water_u.lat == hycom_water_v.lat',
                'hycom_water_u.lon == hycom_water_v.lon',
                'hycom_water_u.time == hycom_water_v.time '\
                'WHERE hycom_water_u.lat >= ?',
                'hycom_water_u.lat <= ?',
                'hycom_water_u.lon >= ?',
                'hycom_water_u.lon <= ?',
                'hycom_water_u.time >= ?',
                'hycom_water_u.time <= ?',
                'hycom_water_u.depth >= ?',
                'hycom_water_u.depth <= ?',
            ]) + ' ORDER BY hycom_water_u.time, hycom_water_u.lat, hycom_water_u.lon ASC'

        sql_values = [
                kwargs['south'], kwargs['north'], kwargs['west'], kwargs['east'],
                dt_2_epoch(kwargs['start']), dt_2_epoch(kwargs['end']), 
                kwargs['top'], kwargs['bottom']
        ]
        sql_values = tuple(map(str, sql_values))

        # query the local database
        conn, db = database_cfg()
        db.execute(sql_query, sql_values)
        qry = np.array(db.fetchall()).T

        conn.close()

        if len(qry) == 0:
            self.logger.warning(f'[{self.name}] water_uv: no data found in region '
                            f'{fmt_coords(kwargs)}, returning empty arrays')
            return np.array([[], [], [], [], []])

        water_u, lat, lon, epoch, depth, water_v = qry
        val = np.sqrt(np.square(water_u) + np.square(water_v))

        # convert longitude from [0;360] to [-180;180]
        lon[lon > 180] = lon[lon > 180] - 360

        return np.array((val, lat, lon, epoch, depth)).astype(float)

    def __str__(self):
        info = '\n'.join([
            "Native hycom .[ab] data converted to NetCDF at the Naval",
            "Research Laboratory, interpolated to 0.08° grid between",
            "40°S-40°N (0.04° poleward) containing 40 z-levels.",
            "Availability: 2014-July to Present",
            "\thttps://www.hycom.org/data/glby0pt08"
        ])

        args = ("(south, north, west, east, "
                "start, end, top, bottom)")

        return str_def(self, info, args)
