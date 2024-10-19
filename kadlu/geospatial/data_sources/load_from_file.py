import logging
from PIL import Image
from functools import reduce
from xml.etree import ElementTree as ET
import json
from datetime import datetime

import matplotlib
'''
matplotlib.use('tkagg')
'''
import mpl_scatter_density
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from kadlu.geospatial.data_sources.data_util import (
    dt_2_epoch,
    epoch_2_dt,
    index_arr,
)


def load_raster(filepath, plot=False, cmap=None, **kwargs):
    """ load data from raster file

        args:
            filepath: string
                complete filepath descriptor of netcdf file to be read
            plot: boolean
                if True, a plot will be displayed using the qt5agg backend
            cmap: matplotlib colormap
                the full list of available colormaps can be viewed with:
                print(matplotlib.pyplot.colormaps())
                if None is supplied, pyplot will default to
                matplotlib.pyplot.cm.cividis

        returns:
            values: numpy 2D array
            lats:   numpy 1D array
            lons:   numpy 1D array
    """
    if kwargs == {}:
        kwargs.update(dict(south=-90, west=-180, north=90, east=180))

    # load raster
    Image.MAX_IMAGE_PIXELS = 500000000

    #with Image.open(filepath) as im:
    im = Image.open(filepath)
    if 33922 in im.tag.tagdata.keys():
        # GDAL raster format: ModelTiepointTag
        # http://duff.ess.washington.edu/data/raster/drg/docs/geotiff.txt
        i, j, k, x, y, z = im.tag_v2[33922]
        dx, dy, dz = im.tag_v2[33550]  # ModelPixelScaleTag
        meta = im.tag_v2[42112]  # GdalMetadata
        xml = ET.fromstring(meta)
        params = {tag.attrib['name']: tag.text for tag in xml}
        lon = np.arange(x, x + (dx * im.size[0]), dx)
        lat = np.arange(y, y + (dy * im.size[1]), dy)[::-1] - 90
        if np.sum(lat > 91):
            lat -= 90
        rng_lon = (
            index_arr(kwargs['west'], lon),
            index_arr(kwargs['east'], lon),
        )
        rng_lat = (
            abs(index_arr(kwargs['north'], lat[::-1]) - len(lat)),
            abs(index_arr(kwargs['south'], lat[::-1]) - len(lat)),
        )
        logging.debug(
            f'{xml.tag}\nraster coordinate system: {im.tag_v2[34737]}'
            f'\n{json.dumps(params, indent=2, sort_keys=True)}', )
    elif 34264 in im.tag.tagdata.keys():
        # NASA / jet propulsion labs raster format: ModelTransformationTag
        dx, _, _, x, _, dy, _, y, _, _, dz, z, _, _, _, _ = im.tag_v2[34264]
        lon = np.arange(x, x + (dx * im.size[0]), dx)
        lat = np.arange(y, y + (dy * im.size[1]), dy)
        rng_lon = index_arr(kwargs['west'],
                            lon), index_arr(kwargs['east'], lon)
        rng_lat = index_arr(kwargs['south'],
                            -lat), index_arr(kwargs['north'], -lat)

    else:
        assert False, f'error {filepath}: unknown metadata tag encoding'
    assert not (z or dz), f'error {filepath}: 3D rasters not supported yet'

    aoi = im.crop((rng_lon[0], rng_lat[0], rng_lon[1], rng_lat[1])).load()
    grid = np.ndarray((len(range(*rng_lon)), len(range(*rng_lat))))

    n = reduce(np.multiply, (rng_lon[1] - rng_lon[0], rng_lat[1] - rng_lat[0]))
    if n > 10000000:
        logging.info(f'this could take a few moments ({n} points)...')

    for xi in np.arange(rng_lon[0], rng_lon[1]) - rng_lon[0]:
        for yi in np.arange(rng_lat[0], rng_lat[1]) - rng_lat[0]:
            grid[xi][yi] = aoi[int(xi), int(yi)]

    mask = grid == float(im.tag_v2[42113])
    val = np.ma.MaskedArray(grid, mask=mask)

    im.close()

    x1, y1 = np.meshgrid(lon[rng_lon[0]:rng_lon[1]],
                         lat[rng_lat[0]:rng_lat[1]],
                         indexing='ij')
    # plot the data
    if plot:
        fig = plt.figure()
        if (rng_lon[1] - rng_lon[0]) * (rng_lat[1] - rng_lat[0]) >= 100000:
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            plt.axis('scaled')
            _raster = ax.scatter_density(x1, y1, c=val, cmap=cmap)
            plt.tight_layout()
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x1, y1, c=val, cmap=cmap)
            pcm = ax.scatter(x1, y1, marker='s', c=val * -1, cmap=cmap)
            fig.colorbar(pcm, ax=ax)
        plt.show()

    _v = np.array(
        [val[x][y] for x in range(val.shape[0]) for y in range(val.shape[1])],
        dtype=float)
    _x = np.array(
        [x1[x][y] for x in range(x1.shape[0]) for y in range(x1.shape[1])],
        dtype=float)
    _y = np.array(
        [y1[x][y] for x in range(y1.shape[0]) for y in range(y1.shape[1])],
        dtype=float)

    return np.array([_v, _y, _x], dtype=object)


def load_netcdf(filename,
                var=None,
                gridXY=None,
                plot=False,
                cmap=matplotlib.pyplot.cm.bone.reversed(),
                **kwargs):
    """ read environmental data from netcdf and output to gridded numpy array

        args:
            filename: string
                complete filepath descriptor of netcdf file to be read
            var: string (optional)
                the netcdf attribute to be read as the values.
                by default, a guess will be made based on the file metadata
            gridXY: string
                netcdf filename describing grid values if lon/lat are stored as
                grid index
            plot: boolean
                if True, a plot will be displayed using the qt5agg backend
            cmap: matplotlib colormap
                the full list of available colormaps can be viewed with:
                print(matplotlib.pyplot.colormaps())

        returns:
            values: numpy 2D array
            lats:   numpy 1D array
            lons:   numpy 1D array
    """
    if kwargs == {}:
        kwargs.update(
            dict(south=-90,
                 west=-180,
                 north=90,
                 east=180,
                 start=datetime(1, 1, 1),
                 end=datetime.now(),
                 top=0,
                 bottom=9999))
    ncfile = netCDF4.Dataset(filename)

    varmap = dict(
        #MAPSTA='f',  # ifremer tag: appears to be a land mask
        lat='y',
        latitude='y',
        gridY='y',
        lon='x',
        longitude='x',
        gridX='x',
        time='t',
        epoch='t',
        depth='z',
        elevation='z',
    )

    discard = ['crs', 'MAPSTA']

    axes = dict([(varmap[var][0], var) for var in ncfile.variables.keys()
                 if var in varmap.keys()])
    uvars = [
        _ for _ in ncfile.variables.keys()
        if _ not in varmap.keys() and _ not in discard
    ]
    if len(uvars) > 0 or var:
        axes.update({'v': var or uvars[0]})
    if not var:
        var = 'z' if uvars == [] and var is None else uvars[0]

    assert 'x' in axes.keys(), f'missing x axis: {uvars}'
    assert 'y' in axes.keys(), f'missing y axis: {uvars}'
    assert len(uvars) <= 1, f'more than one unknown variable: {uvars}'
    assert sum(
        key in varmap.keys()
        for key in ncfile.variables.keys()) >= len(axes), 'not all vars match'
    assert len(axes) >= len(
        ncfile.variables.keys()) - 1, f'missing axis from: {uvars}'

    logging.info(f'loading data from {ncfile.getncattr("title")}')

    if not gridXY:
        rng_lon = index_arr(kwargs['west'],
                            ncfile[axes['x']][:].data), index_arr(
                                kwargs['east'], ncfile[axes['x']][:].data)
        rng_lat = index_arr(kwargs['south'],
                            ncfile[axes['y']][:].data), index_arr(
                                kwargs['north'], ncfile[axes['y']][:].data)
    else:
        assert False

    out = {} if 'v' not in axes.keys() else dict(
        val=ncfile[axes['v']][:].data[rng_lat[0]:rng_lat[1],
                                      rng_lon[0]:rng_lon[1]])
    out.update(
        dict(lat=ncfile[axes['y']][rng_lat[0]:rng_lat[1]],
             lon=ncfile[axes['x']][rng_lon[0]:rng_lon[1]]))

    # temporal index range
    if 't' in axes.keys():
        if ncfile.variables[
                axes['t']].units == 'days since 1990-01-01T00:00:00Z':
            t0 = datetime(1990, 1, 1)
            rng_t = (index_arr(dt_2_epoch(kwargs['start'], t0),
                               ncfile[axes['t']] * 24),
                     index_arr(dt_2_epoch(kwargs['end'], t0),
                               ncfile[axes['t']] * 24))
            out['time'] = epoch_2_dt(ncfile[axes['t']][rng_t[0]:rng_t[1]] * 24,
                                     t0)
        if ncfile.variables[
                axes['t']].units == 'seconds since 1970-01-01T00:00:00Z':
            t0 = datetime(1970, 1, 1)
            rng_t = (index_arr(dt_2_epoch(kwargs['start'], t0),
                               ncfile[axes['t']] / 60 / 60),
                     index_arr(dt_2_epoch(kwargs['end'], t0),
                               ncfile[axes['t']] / 60 / 60))
            out['time'] = epoch_2_dt(ncfile[axes['t']][rng_t[0]:rng_t[1]],
                                     t0,
                                     unit='seconds')
        else:
            assert False, 'unknown time unit'

    # vertical index range
    if 'z' in axes.keys() and 'v' in axes.keys():
        assert axes['z'] != 'elevation', 'netcdf indexing error'
        rng_z = (index_arr(kwargs['top'], ncfile[axes['z']]),
                 index_arr(kwargs['bottom'], ncfile[axes['z']]))
        out['depth'] = ncfile[axes['z']][rng_z[0]:rng_z[1]]
    elif 'z' in axes.keys() and 'v' not in axes.keys() and len(
            axes.keys()) == 3:
        # when loading bathymetry, z-axis are the intended first column values
        out = dict(val=ncfile[axes['z']][rng_lat[0]:rng_lat[1],
                                         rng_lon[0]:rng_lon[1]],
                   lat=out['lat'].copy(),
                   lon=out['lon'].copy())
        if axes['z'] == 'elevation':
            out['val'] *= -1
    else:
        assert 'v' in axes.keys(), 'something may have gone wrong here...'

    # plot the data
    if plot and len(out.keys()) == 3:
        x1, y1 = np.meshgrid(out['lon'], out['lat'], indexing='xy')
        fig = plt.figure()
        if (rng_lon[1] - rng_lon[0]) * (rng_lat[1] - rng_lat[0]) >= 100000:
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            plt.axis('scaled')
            pcm = ax.scatter_density(x1, y1, c=out['val'], cmap=cmap)
            plt.tight_layout()
        else:
            ax = fig.add_subplot(1, 1, 1)
            pcm = ax.scatter(x1, y1, marker='s', c=out['val'], cmap=cmap)
        fig.colorbar(pcm, ax=ax)
        plt.title('Depth (metres)')
        plt.show()

    lat_m, lon_m = np.meshgrid(out['lat'], out['lon'], indexing='ij')

    grid = np.vstack(
        (out['val'].flatten().astype(float), lat_m.flatten().astype(float),
         lon_m.flatten().astype(float)))

    return grid


'''

kwargs = dict(south = 43.5841631194817, north = 44.01583688051829, west = -59.33904235799604, east = -58.740957642003956, top = 0, bottom = 10000)
filename = '/storage/GEBCO_2020.nc'
var=None
gridXY=None
plot = True
cmap = matplotlib.pyplot.cm.bone.reversed()

'''
