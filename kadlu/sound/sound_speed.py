""" Sound speed module within the kadlu package
"""
import gsw
import numpy as np
import logging
from datetime import datetime
from kadlu.utils import interp_grid_1d
from kadlu.geospatial.interpolation import get_interpolator


def sound_speed_teos10(lats, lons, z, t, SP):
    """ Compute sound speed from temperature, salinity, and depth
        for a given latitude and longitude using the Thermodynamic
        Equation of Seawater 2010.

        https://teos-10.github.io/GSW-Python/

        Args:
            lats: numpy array
                Latitudes (-90 to 90 degrees)
            lons: numpy array
                Longitudes (-180 to 180 degrees)
            z: numpy array
                Depths (meters)
            t: numpy array
                In-situ temperature (Celsius)
            SP: numpy array
                Practical Salinity (psu)

        Returns:
            c: numpy array
                Sound speed (m/s)
    """
    p = gsw.p_from_z(
        z=-z, lat=lats)  # sea pressure (gsw uses negative z below sea surface)
    SA = gsw.SA_from_SP(SP, p, lons, lats)  # absolute salinity
    CT = gsw.CT_from_t(SA, t, p)  # conservative temperature
    c = gsw.density.sound_speed(SA=SA, CT=CT, p=p)
    return c


class SoundSpeed():
    """ Class for handling computation and interpolation of sound speed.

        The sound speed can be specified via the argument ssp (sound speed
        profile) or computed from the ocean variables (temperature, salinity).

        ssp can be either a single value, in which case the sound speed is
        the same everywhere, or a tuple (c,z) where c is an array of sound
        speed values and z is an array of depths.

        The interp and interp_xyz method may be used to obtain the interpolated
        sound speed at any set of coordinates.

        TODO: provide proper implementation of time/epoch handling.
        TODO: SoundSpeed should inherit from BaseGeospatialInterpolator

        Args:
            ocean: instance of :class:`kadlu.geospatial.ocean.Ocean`
                Ocean variables
            ssp: float or tuple
                Sound speed profile. May be specified either as a float,
                in which case the sound speed is the same everywhere, or as
                a tuple (c,z) where c is an array of sound speeds and z is
                an array of depth values.
            num_depths: int
                Number of depth values for the interpolation grid. The default value is 50.
            rel_err: float
                Maximum deviation of the interpolation, expressed as a ratio of the
                range of sound-speed values. The default value is 0.001.
    """

    def __init__(self, ocean=None, ssp=None, num_depths=50, rel_err=1E-3, time=None):

        assert ocean is not None or ssp is not None, "ocean or ssp must be specified"

        self.logger = logging.getLogger("kadlu")
        self.name = self.__class__.__name__

        if ssp is not None:
            self.logger.debug(f"[{self.name}] Initializing {self.name} instance with SSP = {ssp}")

            if isinstance(ssp, tuple):
                self._interp = get_interpolator(value=ssp[0], depth=ssp[1])
            else:
                self._interp = get_interpolator(value=ssp)

        else:
            self.logger.debug(f"[{self.name}] Initializing {self.name} instance with Ocean data")

            lat_res, lon_res = self._lat_lon_res(
                ocean, default_res=1.0
            )  #default resolution is 1 degree, approx 100 km

            # geographic boundaries
            S, N, W, E = ocean.boundaries['south'], ocean.boundaries[
                'north'], ocean.boundaries['west'], ocean.boundaries['east']

            # patch: ok 2024-07-17
            # only appropriate for HYCOM data!
            if time is None:
                time = ocean.boundaries['start']

            t0 = datetime(2000, 1, 1)
            epoch = (time - t0).total_seconds() / 3600 #hours since since 2000-01-01 00:00

            # lat and lon coordinates
            num_lats = max(3, int(np.ceil((N - S) / lat_res)) + 1)
            lats = np.linspace(S, N, num=num_lats)
            num_lons = max(3, int(np.ceil((E - W) / lon_res)) + 1)
            lons = np.linspace(W, E, num=num_lons)

            # compute depth coordinates
            depths = self._depth_coordinates(
                ocean,
                lats,
                lons,
                num_depths=num_depths,
                rel_err=rel_err,
                epoch=epoch,
            )

            debug_msg = f"[{self.name}] Interpolation sound speed on grid with:"
            debug_msg += f"\n[{self.name}] - resolution: lat,lon={lat_res:.4f},{lon_res:.4f}"
            debug_msg += f"\n[{self.name}] - size: lats,lons,depths={num_lats},{num_lons},{num_depths}"
            debug_msg += f"\n[{self.name}] - extent: lat,lon,depth=({np.min(lats):.4f},{np.max(lats):.4f})"
            debug_msg += f",({np.min(lons):.4f},{np.max(lons):.4f})"
            debug_msg += f",({np.min(depths):.1f},{np.max(depths):.1f})"
            self.logger.debug(debug_msg)

            # interpolate temperature and salinity on lat,lon,epoch,depth grid
            t = ocean.temperature(lat=lats, lon=lons, depth=depths, epoch=epoch, grid=True)
            s = ocean.salinity(lat=lats, lon=lons, depth=depths, epoch=epoch, grid=True)

            # drop time axis with size 1
            t = t[:,:,0,:]
            s = s[:,:,0,:]

            # compute sound speed
            grid_shape = t.shape
            la, lo, de = np.meshgrid(lats, lons, depths)
            la = la.flatten()
            lo = lo.flatten()
            de = de.flatten()
            t = t.flatten()
            s = s.flatten()
            c = sound_speed_teos10(lats=la, lons=lo, z=de, t=t, SP=s)
            c = np.reshape(c, newshape=grid_shape)

            # create interpolator
            self._interp = get_interpolator(value=c, lat=lats, lon=lons, depth=depths, origin=ocean.origin, method='linear')

    def _lat_lon_res(self, ocean, default_res, min_res=1E-5):
        """ Determine lat,lon resolutions for interpolation grid

            Args:
                ocean: instance of :class:`kadlu.geospatial.ocean.Ocean`
                    Ocean variables
                default_res: float
                    Default resolution in degrees
                min_res: float
                    Highest resolution in degrees. Default is 1E-5, corresponding to approx 1 meter.

            Returns:
                lat_res, lon_res: float,float
                    Resolutions in degrees.
        """
        temp_nodes = ocean.interpolators['temperature'].coordinates
        temp_lat, temp_lon = temp_nodes.get("lat",None), temp_nodes.get("lon",None)

        sal_nodes = ocean.interpolators['salinity'].coordinates
        sal_lat, sal_lon = sal_nodes.get("lat",None), sal_nodes.get("lon",None)

        if temp_lat is None or len(temp_lat) <= 1:
            temp_lat_res = default_res
        else:
            temp_lat_res = max(min_res, np.abs(temp_lat[1] - temp_lat[0]))

        if temp_lon is None or len(temp_lon) <= 1:
            temp_lon_res = default_res 
        else:
            temp_lon_res = max(min_res, np.abs(temp_lon[1] - temp_lon[0]))

        if sal_lat is None or len(sal_lat) <= 1:
            sal_lat_res = default_res
        else:
            sal_lat_res = max(min_res, np.abs(sal_lat[1] - sal_lat[0]))

        if sal_lon is None or len(sal_lon) <= 1:
            sal_lon_res = default_res
        else:
            sal_lon_res = max(min_res, np.abs(sal_lon[1] - sal_lon[0]))

        lat_res = min(temp_lat_res, sal_lat_res)
        lon_res = min(temp_lon_res, sal_lon_res)

        return lat_res, lon_res

    def _depth_coordinates(self, ocean, lats, lons, num_depths, rel_err, epoch):
        """ Compute depth coordinates for lat,lon,depth interpolation grid.

            Args:
                ocean: instance of :class:`kadlu.geospatial.ocean`
                    Ocean variables
                lats,lons: numpy.array, numpy.array
                    Latitude and longitude coordinates
                num_depths: int
                    Number of depth values for the interpolation grid. The default value is 50.
                rel_err: float
                    Maximum deviation of the interpolation, expressed as a ratio of the
                    range of sound-speed values. The default value is 0.001.

            Returns:
                depths: numpy.array
                    Depth coordinates
        """
        seafloor_depth = ocean.bathymetry(lat=lats, lon=lons, grid=True)

        # find deepest point
        deepest_point = np.unravel_index(np.argmax(seafloor_depth),
                                         seafloor_depth.shape)

        # depth and lat,lon coordinates at deepest point
        max_depth = seafloor_depth[deepest_point]
        lat = lats[deepest_point[0]]
        lon = lons[deepest_point[1]]

        self.logger.debug(f"[{self.name}] Deepest point: lat,lon,depth={lat:.4f},{lon:.4f},{max_depth:.1f}m")

        # compute temperature, salinity and sound speed for every 1 meter
        z = np.arange(0, int(np.ceil(max_depth)) + 1)
        t = ocean.temperature(lat=lat, lon=lon, depth=z, epoch=epoch, grid=True)
        s = ocean.salinity(lat=lat, lon=lon, depth=z, epoch=epoch, grid=True)
        c = sound_speed_teos10(lats=lat, lons=lon, z=z, t=t, SP=s)

        # determine grid
        indices, _ = interp_grid_1d(
            y=np.squeeze(c),
            x=z,
            num_pts=num_depths,
            rel_err=rel_err
        )

        depths = z[indices]

        return depths


    def __call__(self, **kwargs):
        return self._interp(**kwargs)
