import os
import logging
import zipfile
import requests
from datetime import datetime, timedelta
import time

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

zipfilename = 'gebco_2021_geotiff.zip'
url = 'https://www.bodc.ac.uk/data/open_download/gebco/gebco_2021/geotiff/'


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
        """ download geotiff zip archive and extract it """

        zipf = os.path.join(storage_cfg(), "gebco_2021_geotiff.zip")

        # download the file if necessary
        if not os.path.isfile(zipf):
            print('downloading gebco bathymetry...')
            with requests.get(url, stream=True, timeout=300) as payload:
                assert payload.status_code == 200, 'error fetching file'
                with open(zipf, 'wb') as f:
                    with tqdm(total=4011407575,
                              desc=zipf,
                              unit='B',
                              unit_scale=True) as t:
                        for chunk in payload.iter_content(chunk_size=8192):
                            _ = t.update(f.write(chunk))

            # unzip the downloaded file
            exists = set(sorted(os.listdir(storage_cfg())))
            with zipfile.ZipFile(zipf, 'r') as zip_ref:
                contents = set(zip_ref.namelist())
                members = list(contents - exists)
                print('extracting bathymetry data...')
                zip_ref.extractall(path=storage_cfg(), members=members)

        time.sleep(2)

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
        """ build grid indexes from geotiffs and insert into the database """

        if not os.path.isfile(os.path.join(storage_cfg(), zipfilename)):
            self.fetch_bathymetry_grid()

        filebounds = lambda fpath: {
            f[0]: float(f[1:])
            for f in fpath.split('gebco_2021_', 1)[1].rsplit('.tif', 1)[0].
            split('_')
        }

        rasterfiles = {
            f: filebounds(f)
            for f in {
                k: None
                for k in sorted([
                    f for f in os.listdir(storage_cfg())
                    if f[-4:] == '.tif' and 'gebco_2021' in f
                ])
            }
        }

        initdb()
        for fname, fbounds in rasterfiles.items():
            in_lat_rng = (fbounds['s'] <= south <= fbounds['n']
                          or fbounds['s'] <= north <= fbounds['n'])
            in_lon_rng = (fbounds['w'] <= west <= fbounds['e']
                          or fbounds['w'] <= east <= fbounds['e'])
            if (not in_lat_rng or not in_lon_rng):
                continue
            assert os.path.isfile(os.path.join(storage_cfg(), fname))
            grid = kadlu.load_file(os.path.join(storage_cfg(), fname),
                                   south=south,
                                   north=north,
                                   west=west,
                                   east=east).T

            conn, db = database_cfg()
            n1 = db.execute("SELECT COUNT(*) FROM gebco ").fetchall()[0][0]
            db.executemany(
                "INSERT OR IGNORE INTO gebco (val, lat, lon)  VALUES (?,?,?)",
                grid)
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
        return np.array([val * -1, lat, lon])
