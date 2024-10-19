"""
    Kadlu API for the NOAA WaveWatch III Datastore

    User guides:
        https://github.com/NOAA-EMC/WW3/wiki/WAVEWATCH-III-User-Guide

    Data model description (boundary definitions, map visualizations, etc)
        https://polar.ncep.noaa.gov/waves/implementations.php
"""

import shutil
import logging
import requests
import platform
import warnings
from datetime import datetime, timedelta
from os.path import isfile
from hashlib import md5

import numpy as np
if platform.machine() == 'arm64':
    warnings.warn(
        "WWIII module requires the pygrib package which is not available for this platform"
    )
else:
    import pygrib

from kadlu import index
from kadlu.geospatial.data_sources.data_util import (
    Boundary,
    database_cfg,
    dt_2_epoch,
    logmsg,
    logmsg_nodata,
    storage_cfg,
    str_def,
)

wwiii_tables = ['hs', 'dp', 'tp', 'windU', 'windV']

wwiii_src = "https://data.nodc.noaa.gov/ncep/nww3/"

# region boundaries as defined in WWIII docs:
#    https://polar.ncep.noaa.gov/waves/implementations.php
wwiii_global = Boundary(-90, 90, -180, 180, 'glo_30m')  # global
wwiii_regions = [
    Boundary(15, 47, -99, -60, 'at_4m'),  # atlantic
    Boundary(15, 50, -165, -116, 'wc_4m'),  # US west
    Boundary(48, 74, 140, 180, 'ak_4m'),  # alaska (west)
    Boundary(48, 74, -180, -120, 'ak_4m'),  # alaska (east)
    Boundary(65, 84, -180, 180, 'ao_30m'),  # arctic ocean
    Boundary(-20, 30, 130, 180, 'ep_10m'),  # pacific (west)
    Boundary(-20, 30, -180, -145, 'ep_10m')
]  # pacific (east)


def initdb():
    conn, db = database_cfg()
    for var in wwiii_tables:
        db.execute(f'CREATE TABLE IF NOT EXISTS {var}'
                   '( val     REAL    NOT NULL, '
                   '  lat     REAL    NOT NULL, '
                   '  lon     REAL    NOT NULL, '
                   '  time    INT     NOT NULL, '
                   '  source  TEXT    NOT NULL) ')
        db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                   f'idx_{var} on {var}(time, lon, lat, val, source)')
    conn.close()


def insert(table, agg):
    """ insert parsed data into local database """
    conn, db = database_cfg()
    db.executemany(
        f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,CAST(? AS INT),?)",
        agg.T)
    conn.commit()
    conn.close()


def fetch_wwiii(var, **kwargs):
    """ download wwiii data and return associated filepaths

        args:
            var: string
                the variable name of desired parameter according to WWIII docs
                the complete list of variables can be found at the following
                URL under 'model output'
                https://polar.ncep.noaa.gov/waves/implementations.php
            south, north: float
                ymin, ymax coordinate boundaries (latitude). range: -90, 90
            west, east: float
                xmin, xmax coordinate boundaries (longitude). range: -180, 180
            start: datetime
                the start of the desired time range
            end: datetime
                the end of the desired time range

        return:
            True if new data was fetched, else False
    """
    assert 6 == sum(
        map(lambda kw: kw in kwargs.keys(),
            ['south', 'north', 'west', 'east', 'start', 'end'
             ])), 'malformed query'
    assert var in ('hs', 'tp', 'dp', 'wind'), 'invalid fetch var'
    assert kwargs['end'].month <= kwargs['start'].month+1 and kwargs['start'].year <= kwargs['end'].year+1, \
            f'fetch function can only load one month at a time. got start: {kwargs["start"]} end: {kwargs["end"]}'

    #print("WWIII NOTICE: resolution selection not implemented yet. defaulting to 0.5°")
    regions = ['glo_30m']

    assert regions == ['glo_30m'], 'invalid region string'
    reg = regions[0]
    t = datetime(kwargs['start'].year, kwargs['start'].month, 1)
    fname = f"multi_1.{reg}.{var}.{t.strftime('%Y%m')}.grb2"
    fetchfile = f"{storage_cfg()}{fname}"

    # if file hasnt been downloaded, fetch it
    if not isfile(fetchfile):  # and kwargs['start'].day == 1:
        logging.info(f'WWIII {kwargs["start"].date().isoformat()} {var}: '
                     f'downloading {fname} from NOAA WaveWatch III...')
        if reg == 'glo_30m' and not 'wind' in var and t.year >= 2018:
            fetchurl = f"{wwiii_src}{t.strftime('%Y/%m')}/gribs/{fname}"
        else:
            fetchurl = f"{wwiii_src}{t.strftime('%Y/%m')}/{reg}/{fname}"
        with requests.get(fetchurl, stream=True, timeout=300) as payload:
            assert payload.status_code == 200, f'couldn\'t retrieve file at {fetchurl}'
            with open(fetchfile, 'wb') as f:
                shutil.copyfileobj(payload.raw, f)

    # open the file, parse data, insert values
    grib = pygrib.open(fetchfile)
    assert grib.messages > 0, f'problem opening {fetchfile}'
    null = 0
    agg = np.array([[], [], [], [], []])
    grbvar = grib[1]['name']
    table = f'{var}{grbvar[0]}' if var == 'wind' else var

    initdb()
    conn, db = database_cfg()
    n1 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]

    for msg, num in zip(grib, range(1, grib.messages)):
        if (msg['name'] != grbvar and md5(agg.tobytes()).hexdigest() != md5(
                np.array([[], [], [], [], []]).tobytes()).hexdigest()):
            insert(table, agg)
            table = f'{var}{msg["name"][0]}' if var == 'wind' else var
            agg = np.array([[], [], [], [], []])
            grbvar = msg['name']
            null = 0
        if msg.validDate < kwargs['start']:
            continue
        if msg.validDate >= kwargs['end']:
            continue
        z, y, x = msg.data()
        src = np.array(['wwiii' for each in z[~z.mask].data])
        grid = np.vstack(
            (z[~z.mask].data, y[~z.mask], ((x[~z.mask] + 180) % 360) - 180,
             dt_2_epoch([msg.validDate
                         for each in z[~z.mask].data]), src)).astype(object)
        agg = np.hstack((agg, grid))
        null += sum(sum(z.mask))
    insert(table, agg)

    n2 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    conn.close()

    logmsg('wwiii', var, (n1, n2), **kwargs)

    return True


def load_wwiii(var, kwargs):
    """ return downloaded wwiii data for specified wavevar according to given
        time, lat, lon boundaries

        args:
            var: string
                the variable short name of desired wave parameter according to
                WWIII docs the complete list of variable short names can be
                found here (under 'model output')
                https://polar.ncep.noaa.gov/waves/implementations.php
            south, north: float
                ymin, ymax coordinate boundaries (latitude). range: -90, 90
            west, east: float
                xmin, xmax coordinate boundaries (longitude). range: -180, 180
            start: datetime
                the start of the desired time range
            end: datetime
                the end of the desired time range

        return:
            val, lat, lon, epoch as np arrays of floats
    """
    if 'time' in kwargs.keys() and 'start' not in kwargs.keys():
        kwargs['start'] = kwargs['time']
        del kwargs['time']
    if 'end' not in kwargs.keys():
        kwargs['end'] = kwargs['start'] + timedelta(hours=3)

    assert 6 == sum(
        map(lambda kw: kw in kwargs.keys(),
            ['south', 'north', 'west', 'east', 'start', 'end'
             ])), 'malformed query'
    assert var in ('hs', 'tp', 'dp', 'windU', 'windV'), 'invalid load var'

    with index(dx=180,
               dy=90,
               dz=5000,
               dt=timedelta(days=1),
               storagedir=storage_cfg(),
               **{
                   k: v
                   for k, v in kwargs.items()
                   if k in ['south', 'north', 'west', 'east', 'start', 'end']
               }) as fetchmap:
        fetchvar = var if var not in ('windU', 'windV') else 'wind'
        fetchmap(callback=fetch_wwiii, var=fetchvar)

    conn, db = database_cfg()
    db.execute(
        ' AND '.join([
            f'SELECT * FROM {var} WHERE lat >= ?', 'lat <= ?', 'lon >= ?',
            'lon <= ?', 'time >= ?', 'time <= ? '
        ]) + ' ORDER BY time, lat, lon ASC',
        tuple(
            map(str, [
                kwargs['south'], kwargs['north'], kwargs['west'],
                kwargs['east'],
                dt_2_epoch(kwargs['start']),
                dt_2_epoch(kwargs['end'])
            ])))

    slices = np.array(db.fetchall(), dtype=object).T
    if len(slices) == 0:
        logmsg_nodata('wwiii', 'wind', **kwargs)
        return np.array([[], [], [], []])

    conn.close()
    val, lat, lon, epoch, source = slices
    return np.array((val, lat, lon, epoch), dtype=float)


class Wwiii():
    """ collection of module functions for fetching and loading """

    def load_wavedirection(self, **kwargs):
        return load_wwiii('dp', kwargs)

    def load_waveperiod(self, **kwargs):
        return load_wwiii('tp', kwargs)

    def load_windwaveheight(self, **kwargs):
        return load_wwiii('hs', kwargs)

    def load_wind_u(self, **kwargs):
        return load_wwiii('windU', kwargs)

    def load_wind_v(self, **kwargs):
        return load_wwiii('windV', kwargs)

    def load_wind_uv(self, **kwargs):
        with index(dx=180,
                   dy=90,
                   dz=5000,
                   dt=timedelta(days=1),
                   storagedir=storage_cfg(),
                   **kwargs) as fetchmap:
            fetchmap(callback=fetch_wwiii, var='wind')

        sql = ' AND '.join(['SELECT windU.val, windU.lat, windU.lon, windU.time, windV.val FROM windU '\
                'INNER JOIN windV '\
                'ON windU.lat == windV.lat',
                            'windU.lon == windV.lon',
                            'windU.time == windV.time '\
                                    'WHERE windU.lat >= ?',
                            'windU.lat <= ?',
                            'windU.lon >= ?',
                            'windU.lon <= ?',
                            'windU.time >= ?',
                            'windU.time <= ?']) + ' ORDER BY windU.time, windU.lat, windU.lon ASC'
        conn, db = database_cfg()
        db.execute(
            sql,
            tuple(
                map(str, [
                    kwargs['south'], kwargs['north'], kwargs['west'],
                    kwargs['east'],
                    dt_2_epoch(kwargs['start']),
                    dt_2_epoch(kwargs['end'])
                ])))
        qry = np.array(db.fetchall()).T
        conn.close()
        if len(qry) == 0:
            logmsg_nodata('wwiii', 'wind', **kwargs)

            return np.array([[], [], [], [], []])
        wind_u, lat, lon, epoch, wind_v = qry
        val = np.sqrt(np.square(wind_u) + np.square(wind_v))
        return np.array((val, lat, lon, epoch)).astype(float)

    def __str__(self):
        info = '\n'.join([
            "WAVEWATCH III: a third generation wave height,",
            "water depth, and current hindcasting modeling framework.",
            "Developed for the community at NOAA/NCEP. About WWIII:",
            "\thttps://github.com/NOAA-EMC/WW3/wiki/About-WW3",
            "Model descriptions:",
            "\thttps://polar.ncep.noaa.gov/waves/implementations.php"
        ])
        args = "(south, north, west, east, start, end)"
        return str_def(self, info, args)
