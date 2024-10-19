import os
import logging
import zipfile
import requests
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

import kadlu
from kadlu import index
from kadlu.geospatial.data_sources.data_util import (
    database_cfg,
    logmsg,
    logmsg_nodata,
    storage_cfg,
)

fname = 'GEBCO_2021.nc'
url = 'https://www.bodc.ac.uk/data/open_download/gebco/gebco_2021/zip/'


def initdb():
    conn, db = database_cfg()
    db.execute('CREATE TABLE IF NOT EXISTS gebco'
               '(   val     REAL    NOT NULL,  '
               '    lat     REAL    NOT NULL,  '
               '    lon     REAL    NOT NULL  )')
    db.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_gebco on gebco(val, lat, lon)')
    conn.close()


class Gebco():

    def fetch_bathymetry_grid(self):
        """ download netcdf archive and extract it """

        zipf = os.path.join(kadlu.storage_cfg(), "gebco_2021_netcdf.zip")

        # unzip the file if necessary
        if not os.path.isfile(os.path.join(storage_cfg(), fname)):

            # if there is no zip file, download it
            if not os.path.isfile(zipf):
                logging.info('downloading gebco bathymetry...')
                with requests.get(url, stream=True) as payload_netcdf:
                    assert payload_netcdf.status_code == 200, 'error fetching file'
                    with open(zipf, 'wb') as f:
                        with tqdm(total=4011413504,
                                  desc='gebco_2021_netcdf.zip',
                                  unit='B',
                                  unit_scale=True) as t:
                            for chunk in payload_netcdf.iter_content(
                                    chunk_size=8192):
                                f.write(chunk)
                                _ = t.update(8192)

            # unzip the downloaded file
            with zipfile.ZipFile(zipf, 'r') as zip_ref:
                logging.info('extracting bathymetry data...')
                zip_ref.extractall(path=storage_cfg())
        return

    def fetch_callback(self,
                       south,
                       north,
                       west,
                       east,
                       top=None,
                       bottom=None,
                       start=None,
                       end=None):
        """ build data grid indexes from .nc file and insert into database """

        if not os.path.isfile(os.path.join(storage_cfg(), fname)):
            self.fetch_bathymetry_grid()

        rows = kadlu.load_file(os.path.join(storage_cfg(), fname),
                               south=south,
                               north=north,
                               west=west,
                               east=east).T

        initdb()
        conn, db = database_cfg()
        n1 = db.execute("SELECT COUNT(*) FROM gebco ").fetchall()[0][0]
        db.executemany(
            "INSERT OR IGNORE INTO gebco (val, lat, lon)  VALUES (?,?,?)",
            rows)
        n2 = db.execute("SELECT COUNT(*) FROM gebco ").fetchall()[0][0]

        logmsg('gebco',
               'bathymetry', (n1, n2),
               south=south,
               west=west,
               north=north,
               east=east)
        conn.commit()
        conn.close()
        return

    def load_bathymetry(self, south, north, west, east, **_):
        """ load gebco bathymetry data """

        with index(dx=2,
                   dy=2,
                   dz=99999,
                   dt=timedelta(hours=24),
                   storagedir=storage_cfg(),
                   bins=True,
                   south=south,
                   north=north,
                   west=west,
                   east=east,
                   top=0,
                   bottom=99999,
                   start=datetime(2000, 1, 1),
                   end=datetime(2000, 1, 2)) as fetchmap:
            _ = fetchmap(callback=self.fetch_callback)

        conn, db = kadlu.database_cfg()
        qry = ' AND '.join([
            'SELECT val, lat, lon FROM gebco WHERE lat >= ?', 'lat <= ?',
            'lon >= ?', 'lon <= ?'
        ])
        logging.debug(f'query: {qry}')
        db.execute(qry, (south, north, west, east))
        res = np.array(db.fetchall()).T
        conn.close()
        if len(res) == 0:
            logmsg_nodata('gebco',
                          'bathymetry',
                          south=south,
                          north=north,
                          west=west,
                          east=east)
            return np.array([[], [], []])
        val, lat, lon = res
        return val, lat, lon
