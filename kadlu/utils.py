import numpy as np
import math
from scipy.interpolate import interp1d
from netCDF4 import Dataset
import scipy.io as sio

# Equatorial radius (6,378.1370 km)
# Polar radius (6,356.7523 km)
# The International Union of Geodesy and Geophysics
# (IUGG) defines the mean radius (denoted R1) to be
# 6,371.009 km
R1_IUGG = 6371009

# Degree to radian conversion factor
deg2rad = np.pi / 180.


def as_array(x, dtype=np.float64):
    """ Converts the input to a numpy array.
    
        If the input is None, the return value is also None.

        Args:
            x: int,float,list,tuple
                Value to be converted to a numpy array
            dtype: 
                Data type. Default is np.float64
    
        Returns:
            : numpy array
    """
    if x is None:
        return None

    if np.ndim(x) == 0:
        return np.array([x], dtype=dtype)

    else:
        return np.array(x, dtype=dtype)


def reverse_index_map(x):
    t = np.zeros((np.max(x) + 1,), dtype=int)
    t[x] = np.arange(0, x.size)
    return t


def center_point(lat, lon):
    """ centerpoint of lat/lon bounds """
    if lat is None or lon is None:
        return None
    
    if isinstance(lat, (list, tuple, np.ndarray)):
        return (min(lat) + max(lat)) / 2, (min(lon) + max(lon)) / 2

    return lat, lon


def DLDL_over_DXDY(lat, lat_deriv_order, lon_deriv_order):
    """ Compute factor for transforming partial derivates in
        lat-lon to partial derivates in x-y.

        Args:
            lat: float or array
                Latitude of the positions(s) where the derivatives are to be evaluated
            lat_deriv_order: int
                Order of latitude-derivative
            lon_deriv_order: int
                Order of longitude-derivative

        Returns:
            ratio: float or array
                Factor for transforming partial derivates in lat-lon to partial derivates in x-y
    """
    R = R1_IUGG
    R2 = R * np.cos(lat * deg2rad)

    m = lat_deriv_order
    n = lon_deriv_order

    if m + n == 0:
        return 1

    ratio = 1

    if m > 0:
        ratio *= np.power(1. / R, m)

    if n > 0:
        ratio *= np.power(1. / R2, n)

    return ratio


def LLtoXY(lat,
           lon,
           lat_ref=0,
           lon_ref=0,
           rot=0,
           grid=False,
           squeeze=True):
    """ Transform lat-lon coordinates to xy position coordinates.

        By default, the origin of the xy coordinate system is
        set to 0 deg latitude and 0 deg longitude.

        By default, the x-axis is aligned with the longitude axis
        (west to east) and the y-axis is aligned with the latitude
        axis (south to north).

        Args:
            lat: float or numpy array
                latitude coordinate of a the location(s) of interest in degrees
            lon: float or numpy array
                longitude coordinate of a the location(s) of interest in degrees
            lat_ref: float
                latitude reference coordinate in degrees
            lon_ref: float
                longitude reference coordinate in degrees
            rot: float
                Rotation angle in degrees for the xy coordinate system
                (clockwise rotation).
            squeeze: bool
                If output array has length 1, return float instead of array

        Returns:
            x: float or numpy array
                x position coordinates in meters
            y: float or numpy array
                y positions coordinates in meters
    """
    global R1_IUGG, deg2rad

    if grid:
        lat, lon = np.meshgrid(lat, lon) #, indexing="ij")

    else:
        if np.ndim(lat) == 0: 
            lat = np.array([lat])
        if np.ndim(lon) == 0: 
            lon = np.array([lon])
        if isinstance(lat, list): 
            lat = np.array(lat)
        if isinstance(lon, list): 
            lon = np.array(lon)

        assert lat.shape[0] == lon.shape[
            0], 'lat and lon must have same length'

    R = R1_IUGG
    R2 = R * np.cos(lat_ref * deg2rad)

    x = (lon - lon_ref) * deg2rad * R2
    y = (lat - lat_ref) * deg2rad * R

    if rot != 0:
        s = np.sin(rot * deg2rad)
        c = np.cos(rot * deg2rad)
        rotmat = np.array([[c, -s], [s, c]])
        xy = np.array([x, y])
        xy = np.swapaxes(xy, 0, 1)
        xy = rotmat.dot(xy)
        x = xy[:, 0]
        y = xy[:, 1]

    if len(x) == 1 and squeeze:
        x = float(x)
        y = float(y)

    return x, y


def XYtoLL(
        x,
        y,
        lat_ref=0,
        lon_ref=0,
        rot=0,
        grid=False,
        squeeze=True
    ):
    """ Transform xy position coordinates to lat-lon coordinates.

        By default, the origin of the xy coordinate system is
        set to 0 deg latitude and 0 deg longitude.

        By default, the x-axis is aligned with the longitude axis
        (west to east) and the y-axis is aligned with the latitude
        axis (south to north).

        Args:
            x: float or numpy array
                x coordinate of a the location(s) of interest in meters
            y: float or numpy array
                y coordinate of a the location(s) of interest in meters
            lat_ref: float
                latitude reference coordinate in degrees
            lon_ref: float
                longitude reference coordinate in degrees
            rot: float
                Rotation angle in degrees for the xy coordinate system
                (clockwise rotation).
            squeeze: bool
                If output array has length 1, return float instead of array

        Returns:
            lat: float or numpy array
                latitude coordinates in degrees
            lon: float or numpy array
                longitude coordinates in degrees
    """
    global R1_IUGG, deg2rad

    if grid:
        x, y = np.meshgrid(x, y) #, indexing="ij")

    else:
        if np.ndim(x) == 0: 
            x = np.array([x])
        if np.ndim(y) == 0: 
            y = np.array([y])
        if isinstance(x, list): 
            x = np.array(x)
        if isinstance(y, list): 
            y = np.array(y)

        assert x.shape[0] == y.shape[0], 'x and y must have same length'

    R = R1_IUGG
    R2 = R * np.cos(lat_ref * deg2rad)

    if rot != 0:
        s = np.sin(rot * deg2rad)
        c = np.cos(rot * deg2rad)
        rotmat = np.array([[c, s], [-s, c]])
        xy = np.array([x, y])
        xy = np.swapaxes(xy, 0, 1)
        xy = rotmat.dot(xy)
        x = xy[:, 0]
        y = xy[:, 1]

    lon = lon_ref + x / deg2rad / R2
    lat = lat_ref + y / deg2rad / R

    if np.ndim(lat) == 1 and len(lat) == 1 and squeeze:
        lat = float(lat)

    if np.ndim(lon) == 1 and len(lon) == 1 and squeeze:
        lon = float(lon)

    return lat, lon


def xdist(lon1, lon2, lat):
    R2 = R1_IUGG * np.cos(lat * deg2rad)
    d = np.abs(lon2 - lon1) * deg2rad * R2
    return d


def ydist(lat1, lat2):
    d = np.abs(lat2 - lat1) * deg2rad * R1_IUGG
    return d


def torad(lat, lon):
    """ Convert latitute and longitude values from degrees to radians.

        The method expects the latitude to be in the range (-90,90) and
        the longitude to be in the range (-180,180).

        The output latitude is in the range (0,pi) and the output
        longitude is in the range (-pi,pi).

        Args:
            lat: float or array
                latitude(s) in degrees from -90 to +90.
            lon: float or array
                longitude(s) in degrees from -180 to +180.

        Returns:
            lat_rad: float or array
                latitude(s) in radians from 0 to pi.
            lon_rad: float or array
                longitude(s) in radians from -pi to pi.
    """
    if lat is None:
        lat_rad = None
    else:
        lat_rad = (lat + 90) * deg2rad
    
    if lon is None:
        lon_rad = None
    else:
        lon_rad = lon * deg2rad
    
    return lat_rad, lon_rad


def get_slices(distance, num_slices=1, bins=100, angle=0):
    """ Generate x,y coordinates for equally spaced radial slices
        originating from (0,0).

        Args:
            distance: float
                Length of the radial slice.
            num_slices: int
                Number of slices
            bins: int
                Number of points per slice
            angle: float
                Angle of the first slice relative to the x-axis.

        Returns:
            x,y: list of numpy arrays
                x,y coordinate arrays for each slice
    """
    x, y = list(), list()

    # distance array
    dr = distance / float(bins)
    r = np.arange(bins, dtype=float)
    r *= dr
    r += 0.5 * dr

    # loop over angles
    a = angle
    da = 360. / float(num_slices)
    for _ in range(num_slices):
        x.append(r * np.cos(a * np.pi / 180.))
        y.append(r * np.sin(a * np.pi / 180.))

    if num_slices == 1:
        x = x[0]
        y = y[0]

    return x, y


def interp_grid_1d(y, x=None, num_pts=math.inf, rel_err=None, method='linear'):
    """ Determine the optimal interpolation grid for the
        function y(x).

        The grid will in general not be uniform, as the
        grid points will be more densily clustered in
        regions where y(x) is changing more rapidly.

        Args:
            y: 1d numpy array
                y values
            x: 1d numpy array
                x values. If none are specified, they are
                assumed to be 0,1,2,...
            num_pts: int
                Number of grid points. If rel_err is specified,
                num_pts becomes the maximum possible number of
                grid points.
            rel_err: float
                Maximum deviation between the interpolation and
                y, relative to the range of values spanned by y.
            method: str
                Interpolation method

        Returns:
            a: 1d numpy array
                Indices of the grid points
            e: float
                Maximum relative error
    """
    n = len(y)
    norm = np.max(y) - np.min(y)

    if x is None:
        x = np.arange(n)

    if num_pts == math.inf and rel_err is None:
        num_pts = 101

    a = np.array([0, n - 1])
    num = len(a)

    while num < num_pts:

        f = interp1d(x=x[a], y=y[a], kind=method)

        dev = np.abs(y - f(x))

        a0 = np.argmax(dev)
        dev0 = dev[a0]

        e = dev0 / norm

        if rel_err is not None and e < rel_err:
            break
        else:
            a = np.append(a, a0)

        num = len(a)

    a = np.sort(a)
    return a, e


def load_data_from_file(path, val_name='bathy', lat_name='lat', lon_name='lon', lon_axis=1,\
    south=-90, north=90, west=-180, east=180):
    """ Load geospatial data from a single file.

        Currently supported formats are NetCDF (*.nc) and MatLab (*.mat).

        The data can be cropped by speciyfing south/north/west/east
        boundaries.

        Args:
            path: str
                File path
            val_name: str
                Name of variable/field containing the data values
            lat_name: str
                Name of variable/field containing the latitude values
            lon_name: str
                Name of variable/field containing the longitude values
            lon_axis: int
                Specify if the longitude dimension is the second (1, default)
                or first (0) axis.
            south: float
                Southern boundary of the region of interest.
            north: float
                Northern boundary of the region of interest.
            west: float
                Western boundary of the region of interest.
            east: float
                Eastern boundary of the region of interest.

        Returns:
            val: 1d or 2d numpy array
                Data values
            lat: numpy array
                Latitude values
            lon: numpy array
                Longitude values
    """
    # detect format
    ext = path[path.rfind('.'):]

    # load data
    if ext == '.nc':  # NetCDF
        d = Dataset(path)
        val = np.array(d[val_name])
        lat = np.array(d[lat_name])
        lon = np.array(d[lon_name])

    elif ext == '.mat':  # MatLab
        d = sio.loadmat(path)
        val = np.array(d[val_name])
        lat = np.squeeze(np.array(d[lat_name]))
        lon = np.squeeze(np.array(d[lon_name]))

    else:
        print('Unknown file format *{0}'.format(ext))
        exit(1)

    # crop the region of interest
    grid = (np.ndim(val) == 2)
    indices, lat, lon = crop(lat, lon, south, north, west, east, grid=grid)
    val = val[indices]

    # ensure that lat and lon are strictly increasing
    if np.all(np.diff(lat) < 0):
        lat = np.flip(lat, axis=0)
        val = np.flip(val, axis=0)
    if np.all(np.diff(lon) < 0):
        lon = np.flip(lon, axis=0)
        val = np.flip(val, axis=1)

    # flip axes, if necessary
    if lon_axis == 0:
        val = np.swapaxes(val, 0, 1)

    # if axis size are inconsistent, try swapping
    if val.shape[0] != lat.shape[0]:
        val = np.swapaxes(val, 0, 1)

    return val, lat, lon


def crop(lat, lon, south, north, west, east, grid=False):
    """ Select rectangular region bounded by the geographical coordinates
        latlon_SW to the south-west and latlon_NE to the north-east.

        If grid is False, lat and lon must have the same length.

        Args:
            lat: 1d or 2d numpy array
                Latitude values
            lon: 1d or 2d numpy array
                Longitude values
            south: float
                Southern boundary of the region of interest.
            north: float
                Northern boundary of the region of interest.
            west: float
                Western boundary of the region of interest.
            east: float
                Eastern boundary of the region of interest.
            grid: bool
                Specify how to combine elements of lat and lon.

        Returns:
            ind: numpy array
                Selected indices. 1d if grid is False, 2d if grid is True.
            lat: numpy array
                Latitude values
            lon: numpy array
                Longitude values
    """
    ind_lat = np.argwhere((lat >= south) & (lat <= north))
    ind_lat = np.squeeze(ind_lat)

    ind_lon = np.argwhere((lon >= west) & (lon <= east))
    ind_lon = np.squeeze(ind_lon)

    if np.ndim(ind_lat) == 2:
        latc = ind_lat[:, 0] + 1j * ind_lat[:, 1]
        lonc = ind_lon[:, 0] + 1j * ind_lon[:, 1]
        x = np.intersect1d(latc, lonc)
        xr = np.real(x)
        xi = np.imag(x)
        xr = xr.astype(int)
        xi = xi.astype(int)
        ind_lat = np.arange(np.min(xr), np.max(xr) + 1)
        ind_lon = np.arange(np.min(xi), np.max(xi) + 1)
        lat = lat[:, 0]
        lon = lon[0, :]
        ind = np.ix_(ind_lat, ind_lon)

    if grid:
        ind = np.ix_(ind_lat, ind_lon)
    else:
        ind = np.intersect1d(ind_lat, ind_lon)
        ind_lat = ind
        ind_lon = ind

    lat = lat[ind_lat]
    lon = lon[ind_lon]

    return ind, lat, lon


def create_boolean_array(n, step=1):
    arr = np.zeros(n, dtype=bool)
    if step == math.inf: arr[0] = True
    else: arr[::step] = True
    return arr


def toarray(x):
    if isinstance(x, float) or isinstance(x, int): x = [x]
    if isinstance(x, list): x = np.array(x)
    return x
