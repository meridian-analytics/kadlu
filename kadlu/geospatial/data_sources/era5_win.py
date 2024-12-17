# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:31:53 2024

@author: kaity
"""

import os
import logging
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta
import cdsapi
import numpy as np
from os.path import isfile, dirname
import xarray as xr  # cfgrib supports xarray
from kadlu.geospatial.data_sources.data_util import (
    database_cfg,
    dt_2_epoch,
    logmsg,
    logmsg_nodata,
    storage_cfg,
)

era5_tables = [
    'significant_height_of_combined_wind_waves_and_swell',
    'mean_wave_direction',
    'mean_wave_period',
    'u_component_of_wind',
    'v_component_of_wind',
    'convective_precipitation',
    'convective_snowfall',
    'normalized_energy_flux_into_ocean',
    'normalized_energy_flux_into_waves',
    'normalized_stress_into_ocean',
    'precipitation_type',
    'surface_solar_radiation_downwards',
]

logging.getLogger('cdsapi').setLevel(logging.WARNING)


def initdb():
    conn, db = database_cfg()
    for var in era5_tables:
        db.execute(f'''
            CREATE TABLE IF NOT EXISTS {var} (
                val REAL NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                time INT NOT NULL,
                source TEXT NOT NULL
            )
        ''')
        db.execute(f'''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{var} 
            ON {var}(time, lon, lat, val, source)
        ''')
    conn.close()


def clear_cache_era5():
    logger = logging.getLogger("kadlu")
    dir_path = Path(storage_cfg())
    if not dir_path.exists():
        logger.warning(f"Kadlu storage directory not found at {dir_path}")
        return

    files = list(dir_path.glob("ERA5_*.grib2"))
    if not files:
        logger.info("ERA5 cache is empty.")
        return

    size = sum(file.stat().st_size for file in files)
    for file in files:
        file.unlink()

    logger.info(f"Emptied ERA5 cache (deleted {len(files)} files, {size / 1E6:.1f} MB)")


def fetch_era5(var, *, west, east, south, north, start, **_):
    logger = logging.getLogger("kadlu")

    try:
        client = cdsapi.Client()
    except Exception as e:
        raise KeyError("You need a CDS API access token.") from e

    t = datetime(start.year, start.month, start.day)
    times = [f"{hour:02d}:00" for hour in range(24)]

    request = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': var,
        'year': t.year,
        'month': t.month,
        'day': t.day,
        'time': times,
        'grid': [0.25, 0.25],
    }

    target = Path(storage_cfg()) / f"ERA5_reanalysis_{var}_{t.strftime('%Y-%m-%d')}.grib2"
    if not target.exists():
        logger.info(f"Fetching {target}...")
        client.retrieve("reanalysis-era5-single-levels", request, str(target))

    assert target.exists()

    ds = xr.open_dataset(str(target), engine="cfgrib")
    data = []

    for dt in ds.time.values:
        timestamp = np.datetime64(dt).astype("datetime64[h]").astype(int)
        for lat, lon, val in zip(ds.latitude.values, ds.longitude.values, ds[var].values.flatten()):
            if west <= lon <= east and south <= lat <= north:
                data.append((val, lat, lon, timestamp, "era5"))

    initdb()
    conn, db = database_cfg()
    table = var
    n1 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    db.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?)", data)
    n2 = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.commit()
    conn.close()

    logmsg("era5", var, (n1, n2), south=south, west=west, north=north, east=east, start=start, end=start + timedelta(days=1))
    return True



def load_era5(var, *, west, east, south, north, start, end, fetch=True, **_):
    if fetch:
        fetch_era5(var, west=west, east=east, south=south, north=north, start=start)

    conn, db = database_cfg()
    table = var
    query = f"""
        SELECT * FROM {table} 
        WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        AND time BETWEEN ? AND ? ORDER BY time, lat, lon
    """
    params = (south, north, west, east, dt_2_epoch(start), dt_2_epoch(end))
    data = np.array(db.execute(query, params).fetchall(), dtype=object).T
    conn.close()

    return data if data.size else None



class Era5():
    """ Collection of module functions for fetching and loading.
    
        The functions return (values, lat, lon, epoch) numpy arrays with 
        shape (num_points, 4) where epoch is the number of hours since 2000-01-01.
    """

    def load_windwaveswellheight(self, **kwargs):
        return load_era5('significant_height_of_combined_wind_waves_and_swell',
                         **kwargs)

    def load_wavedirection(self, **kwargs):
        return load_era5('mean_wave_direction', **kwargs)

    def load_waveperiod(self, **kwargs):
        return load_era5('mean_wave_period', **kwargs)

    def load_precipitation(self, **kwargs):
        return load_era5('convective_precipitation', **kwargs)

    def load_snowfall(self, **kwargs):
        return load_era5('convective_snowfall', **kwargs)

    def load_flux_ocean(self, **kwargs):
        return load_era5('normalized_energy_flux_into_ocean', **kwargs)

    def load_flux_waves(self, **kwargs):
        return load_era5('normalized_energy_flux_into_waves', **kwargs)

    def load_stress_ocean(self, **kwargs):
        return load_era5('normalized_stress_into_ocean', **kwargs)

    def load_precip_type(self, **kwargs):
        return load_era5('precipitation_type', **kwargs)

    def load_wind_u(self, **kwargs):
        return load_era5('10m_u_component_of_wind', **kwargs)

    def load_wind_v(self, **kwargs):
        return load_era5('10m_v_component_of_wind', **kwargs)
    
    def load_insolation(self, **kwargs):
        data = load_era5('surface_solar_radiation_downwards', **kwargs)
        # for accumulated quantities, we subtract 1/2 hour to get the time at the center of the forecast bin
        data[3] -= 0.5
        return data

    def load_irradiance(self, **kwargs):
        data = self.load_insolation(**kwargs)
        data[0] /= 3600
        return data

    def load_wind_uv(self, fetch=True, **kwargs):
        """ Loads wind speed computed as sqrt(wind_u^2 + wind_v^2)"""
        # Check local database for data.
        # Fetch data from CDS API, if missing.
        if fetch:
            with index(storagedir=storage_cfg(),
                    west=kwargs['west'],
                    east=kwargs['east'],
                    south=kwargs['south'],
                    north=kwargs['north'],
                    start=kwargs['start'],
                    end=kwargs['end']) as fetchmap:
                fetchmap(callback=fetch_era5, var='10m_u_component_of_wind')
                fetchmap(callback=fetch_era5, var='10m_v_component_of_wind')

        # establish connection to the geospatial.db database
        conn, db = database_cfg()
        
        # check if the tabls exist
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('u_component_of_wind','v_component_of_wind')"
        rows = db.execute(query).fetchall()
        tables_exist = len(rows) >= 2

        if not tables_exist:
            return np.array(([], [], [], [])).astype(float)

        # form SQL query
        sql = ' AND '.join(['SELECT u_component_of_wind.val, u_component_of_wind.lat, u_component_of_wind.lon, u_component_of_wind.time, v_component_of_wind.val FROM u_component_of_wind '\
                'INNER JOIN v_component_of_wind '\
                'ON u_component_of_wind.lat == v_component_of_wind.lat',
                            'u_component_of_wind.lon == v_component_of_wind.lon',
                            'u_component_of_wind.time == v_component_of_wind.time '\
                                    'WHERE u_component_of_wind.lat >= ?',
                            'u_component_of_wind.lat <= ?',
                            'u_component_of_wind.lon >= ?',
                            'u_component_of_wind.lon <= ?',
                            'u_component_of_wind.time >= ?',
                            'u_component_of_wind.time <= ?']) + ' ORDER BY u_component_of_wind.time, u_component_of_wind.lat, u_component_of_wind.lon ASC'
        
        # perform the query
        db.execute(
            sql,
            tuple(
                map(str, [
                    kwargs['south'], kwargs['north'], kwargs['west'],
                    kwargs['east'],
                    dt_2_epoch(kwargs['start']),
                    dt_2_epoch(kwargs['end'])
                ])))

        wind_u, lat, lon, epoch, wind_v = np.array(db.fetchall()).T

        # compute speed
        val = np.sqrt(np.square(wind_u) + np.square(wind_v))

        conn.close()

        return np.array((val, lat, lon, epoch)).astype(float)

    def __str__(self):
        info = '\n'.join([
            "Era5 Global Dataset from Copernicus Climate Datastore.",
            "Combines model data with observations from across",
            "the world into a globally complete and consistent dataset",
            "\thttps://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels"
        ])
        args = "(south, north, west, east, start, end)"
        return str_def(self, info, args)
