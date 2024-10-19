""" Geophony module within the kadlu library
"""
import copy
import logging
import numpy as np
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from kadlu.geospatial.ocean import Ocean
from kadlu.sound.sound_speed import SoundSpeed
from kadlu.sound.parabolic_equation import TransmissionLoss
from kadlu.utils import xdist, ydist, LLtoXY, XYtoLL, DLDL_over_DXDY, deg2rad
""" Wind source level parametrization of Kewley et al. 1990.
    Values inferred from Fig. 5.7, Ocean Ambient Noise p. 114.

        * x: frequency in Hz
        * y: wind speed in m/s
        * z: source level in dB re 1 uPa^2 / Hz @ 1m / m^2
"""
_kewley_table = np.array([[40.0, 44.0, 48.0, 53.0, 58.0],\
                 [37.5, 42.5, 48.0, 53.0, 58.0],\
                 [34.0, 39.0, 48.0, 53.0, 58.0]])
_kewley_table = np.swapaxes(_kewley_table, 0, 1)
x=[40, 100, 300],
y=[2.57, 5.14, 10.29, 15.23, 20.58]
x, y = np.meshgrid(x, y)
pts = np.stack([x.flatten(), y.flatten()], axis=1)
v = _kewley_table.flatten()
_kewley_interp = LinearNDInterpolator(
    points=pts, 
    values=v,
)
_kewley_interp_nearest = NearestNDInterpolator(
    x=pts, 
    y=v,
)                                        
#interp2d(x=[40, 100, 300],
#        y=[2.57, 5.14, 10.29, 15.23, 20.58],
#        z=_kewley_table,
#        kind='linear')


def kewley_sl_func(*, freq, wind_uv, **_):
    """ Compute the wind source level according to the
        tabulation of Kewley et al. 1990. (Ocean Ambient Noise p. 114).

        Values outside the tabulation domain are extrapolated via
        nearest-neighbor extrapolation.

        Args:
            freq: float
                Frequency in Hz
            wind_uv: float or array
                Wind speed in m/s

        Returns:
            : array-like
                Source level in units of dB re 1 uPa^2 / Hz @ 1m / m^2
    """
    x1, x2 = np.meshgrid(freq, wind_uv)
    x1 = x1.flatten()
    x2 = x2.flatten()
    values = _kewley_interp(x1, x2)
    indices_nan = np.where(np.isnan(values))
    # replace NaNs with nearest value
    if len(indices_nan) > 0: 
        values_nearest = _kewley_interp_nearest(x1, x2)
        values[indices_nan] = values_nearest[indices_nan]
    
    return np.squeeze(values)


def source_level(freq, x, y, area, ocean, sl_func):
    """ Compute source levels at the specified frequency and coordinates.

        Args:
            freq: float
                Sound frequency in Hz.
            x: float or array
                x-coordinate(s)
            y: float or array
                y-coordinate(s)
            area: float or array
                Source area of each location in units of meters squared.
                Must have same shape as x and y.
            ocean: instance of :class:`kadlu.geospatial.ocean`
                Ocean variables
            sl_func: function
                Source level function

        Returns:
            sl: array-like
                Source levels in units of dB re 1 uPa^2 / Hz @ 1m.
    """
    kwargs = {
        'freq': freq,
        'wind_uv': ocean.wind_uv(x=x, y=y),
        'waveheight': ocean.waveheight(x=x, y=y)
    }
    sl = sl_func(**kwargs)  # source level per unit area
    sl += 10 * np.log10(area)  # scale by area
    return sl


def transmission_loss(
        freq,
        propagation_range,
        lat=None,
        lon=None,
        data_range=None,
        seafloor={
            'sound_speed': 1700,
            'density': 1.5,
            'attenuation': 0.5
        },
        ocean=None,
        return_ocean=False,
        **kwargs
    ):
    """ Initialize transmission loss calculator.

        Use the keyword arguments from :class:`kadlu.geospatial.ocean.Ocean`,
        :class:`kadlu.sound.sound_speed.SoundSpeed` and
        :class:`kadlu.sound.parabolic_equation.TransmissionLoss` to specify
        ocean data sources, sound speed profile, and configure the transmission
        loss computation.

        Args:
            freq: float
                Sound frequency in Hz.
            propagation_range: float
                Propagation range in km. Default is 50 km.
            lat, lon: array-like
                Latitude and longitudes of point-like sound source
            data_range: float
                By default, environmental data is loaded for a bounding box centered
                at (lat,lon) with dimensions (2 x 1.2 x propagation_range) x (2 x 1.2 x propagation_range).
                If the data_range argument is specified, the dimensions of this box
                will be modified to (2 x D) x (2 x D) where D = max(data_range, 1.2*propagation_range).
            seafloor: dict
                Bottom acoustic properties.
            ocean: instance of :class:`kadlu.geospatial.ocean.Ocean`
                Ocean variables. If not specified, will be created based on keyword args provided.              
            return_ocean: bool
                Return ocean object. Default is False.

        Returns:
            transm_loss: instance of :class:`kadlu.sound.parabolic_equation.TransmissionLoss`
                Transmission loss calculator
            ocean: instance of :class:`kadlu.geospatial.ocean.Ocean`
                Ocean variables, only returned if return_ocean is True
    """
    logger = logging.getLogger("kadlu")

    k = copy.copy(kwargs)

    # geographic boundaries
    if lat is not None and lon is not None:
        dist = 1.2 * 1e3 * propagation_range
        if data_range is not None: 
            dist = max(1e3 * data_range, dist)
        
        dlat, dlon = _delta_lat_lon(lat, dist)
        k['south'] = lat - dlat
        k['north'] = lat + dlat
        k['west'] = lon - dlon
        k['east'] = lon + dlon

    # TODO: improve design of how keyword args are passed to the Ocean class constructor
    if ocean is None:
        ocean = Ocean(
            **{
                key: v
                for key, v in k.items()
                if key in ('north', 'south', 'west', 'east', 'top', 'bottom',
                        'start', 'end') or key[0:5] == 'load_' or key == "interp_args" or key == "drop"
            })

    # sound speed
    ss = SoundSpeed(ssp=k['ssp']) if 'ssp' in k.keys() else SoundSpeed(ocean=ocean)  

    # transmission loss calculator
    if 'bottom' in k.keys(): 
        k.pop('bottom')
    
    def bathy_func(x,y,grid=False):
        return ocean.bathymetry(x=x, y=y, grid=grid)

    def bathy_deriv_func(x,y,axis):
        if axis == "x":
            return ocean.bathymetry(x=x, y=y, dx=1)
        elif axis == "y":
            return ocean.bathymetry(x=x, y=y, dy=1)
    
    def sound_speed_func(x,y,z,grid=False):
        return ss(x=x, y=y, depth=z, grid=grid)

    transm_loss = TransmissionLoss(
        freq=freq,
        bathy_func=bathy_func,
        bathy_deriv_func=bathy_deriv_func,
        sound_speed_func=sound_speed_func,
        bottom=seafloor,
        propagation_range=propagation_range,
        **k
    )

    if return_ocean: 
        return transm_loss, ocean
    else: 
        return transm_loss


def geophony(freq,
             depth,
             sl_func=kewley_sl_func,
             seafloor={
                 'sound_speed': 1700,
                 'density': 1.5,
                 'attenuation': 0.5
             },
             below_seafloor=False,
             progress_bar=True,
             **kwargs):
    """ Calculate ocean ambient noise levels.

        Noise levels can be calculated either a set of lat-lon coordinates,
        or on a regular grid within a geographic bounding box.

        If below_seafloor is False (default), the noise level is not computed
        below the seafloor and instead assigned a NaN value.

        Use the keyword arguments from :class:`kadlu.geospatial.ocean.Ocean`,
        :class:`kadlu.sound.sound_speed.SoundSpeed` and
        :class:`kadlu.sound.parabolic_equation.TransmissionLoss` to specify
        ocean data sources, sound speed profile, and configure the transmission
        loss computation.

        Args:
            freq: float
                Sound frequency in Hz.
            depth: array-like
                Depths at which to compute the noise levels
            lat, lon: array-like
                Latitude and longitudes at which to compute the noise levels
            south, north: float
                South-north boundaries to fetch ocean data.
            west, east: float
                West-east boundaries to fetch ocean data.                origin: tuple(float,float)
                    Reference location (origo of XY coordinate system). If not specified, defaults 
                    to the @origin class attribute

            xy_res: float
                Horizontal spacing in km between points at which the noise level is computed.
                Only relevant if specifying a bounding box rather than specific locations.
            sl_func: function
                Source level function
            seafloor: dict
                Bottom acoustic properties.
            below_seafloor: bool
                Compute the noise below the seafloor. Default is False.
            progress_bar: bool
                Display calculation progress bar. Default is True.

        Returns:
            g: dict
                Model output: spl,lats,lons,x,y,z,bathy.

                    * spl: numpy.array with shape (nx,ny,nz)
                        Sound pressure levels in dB re 1 uPa^2 / Hz

                    * lats: numpy.array with shape (ny)
                        Latitude coordinates

                    * lons: numpy.array with shape (nx)
                        Longitude coordinates

                    * x: numpy.array with shape (nx)
                        x coordinates

                    * y: numpy.array with shape (ny)
                        y coordinates

                    * z: numpy.array with shape (nz)
                        Depth coordinates

                    * bathy: numpy.array with shape (nx,ny)
                        Bathymetry values
    """
    if isinstance(depth, list): depth = np.array(depth)
    elif isinstance(depth, (float, int)): depth = np.array([depth])
    depth = np.sort(depth)

    if 'lat' in kwargs.keys():
        assert 'lon' in kwargs.keys(), "both lat and lon must be specified"
        lats = kwargs['lat']
        lons = kwargs['lon']
        if isinstance(lats, (int, float)): lats = np.array([lats])
        if isinstance(lons, (int, float)): lons = np.array([lons])
        x, y = LLtoXY(lats, lons, squeeze=False)
        assert 'propagation_range' in kwargs.keys(
        ), 'propagation_range must be specified'

    else:
        xy_res = kwargs.get('xy_res', 50)  #default bin size is 50km
        s = kwargs.pop('south', -90)
        n = kwargs.pop('north', 90)
        w = kwargs.pop('west', -180)
        e = kwargs.pop('east', 180)
        lats, lons, x, y = _create_xy_grid(s,
                                           n,
                                           w,
                                           e,
                                           x_res=1e3 * xy_res,
                                           y_res=1e3 * xy_res)
        if 'propagation_range' not in kwargs.keys():
            kwargs['propagation_range'] = xy_res / np.sqrt(2)

    if 'c0' not in kwargs.keys(): kwargs['c0'] = 1500

    # loop over lat,lon coordinates
    N = len(lats)
    spl = None
    bathy = []
    if N == 1:
        progress_bar = False
        progress_bar_transm = True
    else:
        progress_bar_transm = False

    for i in tqdm(range(N), disable=not progress_bar):

        lat = lats[i]
        lon = lons[i]

        # initialize transmission loss calculator
        kwargs['lat'] = lat
        kwargs['lon'] = lon
        transm_loss, ocean = transmission_loss(freq=freq,
                                               seafloor=seafloor,
                                               return_ocean=True,
                                               **kwargs)

        # interpolate bathymetry
        b = ocean.bathymetry(lat=lat, lon=lon)
        bathy.append(b)

        if below_seafloor: z = depth
        else: z = depth[depth <= b]

        if len(z) == 0:
            dB = np.empty((1, len(depth)), dtype=float)
            dB[:, :] = np.nan

        else:
            # transmission loss
            rec_depth = 0.25 * kwargs[
                'c0'] / freq  # set receiver depth to 1/4 of the characteristic wave length
            tl, ax = transm_loss.calc(source_depth=z,
                                      rec_depth=rec_depth,
                                      progress_bar=progress_bar_transm)
            tl = tl[:, 0, :, :]

            # source level
            sl = _source_level_polar_grid(freq=freq,
                                          radial_axis=ax['radial_axis'],
                                          azimuthal_axis=ax['azimuthal_axis'],
                                          ocean=ocean,
                                          sl_func=sl_func)

            # integrate SL-TL to obtain sound pressure level
            p = np.power(10, (sl - tl) / 10)
            p = np.squeeze(np.apply_over_axes(np.sum, p, range(
                1, p.ndim)))  # sum over all but the first axis
            dB = 10 * np.log10(p)
            if np.ndim(dB) == 0: dB = np.array([dB])

            # pad, if necessary
            n = len(depth) - len(dB)
            if n > 0:
                pad = np.empty(n)
                pad[:] = np.nan
                dB = np.concatenate((dB, pad))

            dB = dB[np.newaxis, :]

        if spl is None: spl = dB
        else: spl = np.concatenate((spl, dB), axis=0)

    # transform output array to desired shape
    spl = np.reshape(spl, newshape=(len(y), len(x), spl.shape[1]))
    spl = np.swapaxes(spl, 0, 1)

    return {
        'spl': spl,
        'lats': lats,
        'lons': lons,
        'x': x,
        'y': y,
        'z': depth,
        'bathy': bathy
    }


def _delta_lat_lon(lat, dist):
    """ Compute change in latitude and longitude for given distance in meters

        Args:
            lat: array-like
                Latitude
            dist: float
                Distance in meters

        Returns:
            delta_lat, delta_lon: array-like
                Change in latitude and longitude
    """
    delta_lat = 1. / deg2rad * dist * DLDL_over_DXDY(
        lat=lat, lat_deriv_order=1, lon_deriv_order=0)
    delta_lon = 1. / deg2rad * dist * DLDL_over_DXDY(
        lat=lat, lat_deriv_order=0, lon_deriv_order=1)
    return delta_lat, delta_lon


def _create_xy_grid(south, north, west, east, x_res, y_res):
    """ Create 2D surface grid for the geophony computation.

        Args:
            south, north: float
                Latitude range
            west, east: float
                Longitude range
            x_res: float
                Longitude resolution in meters
            y_res: float
                Latitude resolution in meters

        Returns:
            lats, lons: numpy.array, numpy.array
                Latitude and longitude coordinates of the grid points
            x, y: numpy.array, numpy.array
                x and y coordinates of the grid points
    """
    # select latitude closest to the equator
    if np.abs(south) < np.abs(north): lat = south
    else: lat = north

    # compute x and y range
    xd = xdist(lon2=east, lon1=west, lat=lat)
    yd = ydist(lat2=north, lat1=south)

    # number of bins
    nx = int(xd / x_res)
    ny = int(yd / y_res)
    nx += nx % 2
    ny += ny % 2

    # create x and y arrays
    x = np.arange(start=-nx / 2, stop=nx / 2 + 1) * x_res
    y = np.arange(start=-ny / 2, stop=ny / 2 + 1) * y_res

    # convert to lat-lon
    lat_ref = 0.5 * (north + south)
    lon_ref = 0.5 * (east + west)
    lats, lons = XYtoLL(x=x, y=y, lat_ref=lat_ref, lon_ref=lon_ref, grid=True)

    lats = lats.flatten()
    lons = lons.flatten()

    return lats, lons, x, y


def _source_level_polar_grid(freq, radial_axis, azimuthal_axis, ocean,
                             sl_func):
    """ Compute the source levels of each area element on a
        regular grid in poolar coordinates.

        Args:
            freq: float
                Sound frequency in Hz.
            radial_axis: numpy.array
                Radial coordinates
            azimuthal_axis: numpy.array
                Azimuthal coordinates
            ocean: instance of :class:`kadlu.geospatial.ocean.Ocean`
                Ocean variables
            sl_func: function
                Source level function

        Returns:
            sl: numpy.array
                Source levels in units of dB re 1 uPa^2 / Hz @ 1m.
                Has shape (1,nq,nr) where nq is the number of angular bins
                and nr is the number of radial bins.
    """
    r = np.copy(radial_axis)
    q = np.copy(azimuthal_axis)

    # x,y coordinates
    r, q = np.meshgrid(r, q)
    x = r * np.cos(q)
    y = r * np.sin(q)

    # area elements (m^2)
    dr = np.diff(radial_axis)[0]
    dq = np.diff(azimuthal_axis)[0] if len(q) >= 2 else 2 * np.pi
    a = dr * dq * r

    # compute source levels
    x = x.flatten()
    y = y.flatten()
    a = a.flatten()
    sl = source_level(freq=freq,
                      x=x,
                      y=y,
                      area=a,
                      ocean=ocean,
                      sl_func=sl_func)
    sl = np.reshape(sl, newshape=r.shape)  # transform to desired shape
    sl = sl[np.newaxis, :, :]

    return sl
