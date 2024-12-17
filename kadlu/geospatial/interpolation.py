""" The interpolation module contains functionalities for interpolating
    geospatial data in spherical and planar geometry in one, two, or
    three dimensions (latitude, longitude, elevation).

    In the two-dimensional case, the interpolation can be made on both regular
    and irregular grids.

    In the three-dimensional case, only interpolation on regular grids has been
    implemented, although an extension to irregular grids (following the same
    methodology as in the two-dimensional case) should be straightforward.

    Contents:
        GridData2D class:
        GridData3D class:
        Interpolator2D class:
        Interpolator3D class:
        Uniform2D class:
        Uniform3D class:
        DepthInterpolator3D class
"""

import numpy as np
from scipy.interpolate import (
    RectSphereBivariateSpline,
    RegularGridInterpolator,
    griddata,
    interp1d,
    interp2d,
    # NearestNDInterpolator,
    # RectBivariateSpline,
)
from kadlu.utils import (
    DLDL_over_DXDY,
    XYtoLL,
    center_point,
    torad,
    # LLtoXY,
    # deg2rad,
)


class GridData2D():
    """ Interpolation of data on a two-dimensional irregular grid.

        Essentially, a wrapper function around scipy's interpolate.griddata.

        https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.griddata.html

        An alternative to griddata could be Rbf, as discussed here:

        https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy

        Attributes:
            u: 1d numpy array
                data points 1st coordinate
            v: 1d numpy array
                data points 2nd coordinate
            r: 1d numpy array
                data values
            method : {‘linear’, ‘nearest’, ‘cubic’}, optional

    """

    def __init__(self, u, v, r, method='linear'):
        self.uv = np.array([u, v]).T
        self.r = r
        self.method = method

    def __call__(self, theta, phi, grid=False, dtheta=0, dphi=0):
        """ Interpolate data

            theta and phi can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the positions (theta_i, phi_i), where theta=(theta_1,...,theta_N) and
            phi=(phi_1,...,phi_N). Note that in this case, theta and phi must have
            the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (theta_i, phi_j), where theta=(theta_1,...,theta_N) and
            phi=(phi_1,...,phi_M). Note that in this case, the lengths of theta
            and phi do not have to be the same.

            Note: Interpolation of derivates not yet implemented; dtheta > 0 or dphi > 0
            will result in AssertionError.

            Args:
                theta: float or array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: float or array
                   2nd coordinate of the points where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of theta and phi.
                dtheta: int
                    Order of theta-derivative
                dphi: int
                    Order of phi-derivative

            Returns:
                ri: Interpolated values
        """
        assert dtheta + dphi == 0, "Interpolation of derivatives not implemented for irregular grids"

        if grid: theta, phi, M, N = self._meshgrid(theta, phi)

        pts = np.array([theta, phi]).T
        ri = griddata(self.uv, self.r, pts, method=self.method)

        if grid:
            ri = np.reshape(ri, newshape=(N, M))
            ri = np.swapaxes(ri, 0, 1)

        return ri

    def _meshgrid(self, theta, phi):
        """ Create grid

            Args:
                theta: 1d numpy array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: 1d numpy array
                   2nd coordinate of the points where the interpolation is to be evaluated

            Returns:
                theta, phi: 2d numpy array
                    Grid coordinates
                M, N: int
                    Number of grid values
        """
        M = 1
        N = 1
        if np.ndim(theta) == 1: M = len(theta)
        if np.ndim(phi) == 1: N = len(phi)
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.flatten()
        phi = phi.flatten()
        return theta, phi, M, N


class Interpolator2D():
    """ Class for interpolating 2D (lat,lon) geospatial data.

        For irregular grids, the data values must be passed as a
        1d array and all three arrays (values, lats, lons) must have
        the same length.

        For regular grids, the data values must be passed as a
        2d array with shape (M,N) where M and N are the lengths
        of the latitude and longitude array, respectively.

        Attributes:
            values: 1d or 2d numpy array
                Values to be interpolated
            lats: 1d numpy array
                Latitude values
            lons: 1d numpy array
                Longitude values
            origin: tuple(float,float)
                Reference location (origo of XY coordinate system).
            method_irreg : {‘linear’, ‘nearest’, ‘cubic’, ‘regularize’}, optional
                Interpolation method used for irregular grids.
                Note that 'nearest' is usually significantly faster than
                the 'linear' and 'cubic'.
                If the 'regularize' is selected, the data is first mapped onto
                a regular grid by means of a linear interpolation (for points outside
                the area covered by the data, a nearest-point interpolation is used).
                The bin size of the regular grid is specified via the reg_bin argument.
            bins_irreg_max: int
                Maximum number of bins along either axis of the regular grid onto which
                the irregular data is mapped. Only relevant if method_irreg is set to
                'regularize'. Default is 2000.
    """

    def __init__(self,
                 values,
                 lats,
                 lons,
                 origin=None,
                 method_irreg='regularize',
                 bins_irreg_max=2000):

        # compute coordinates of origin, if not provided
        if origin is None: origin = center_point(lats, lons)
        self.origin = origin

        # check if bathymetry data are on a regular or irregular grid
        reggrid = (np.ndim(values) == 2)

        # convert to radians
        lats_rad, lons_rad = torad(lats, lons)

        # necessary to resolve a mismatch between scipy and underlying Fortran code
        # https://github.com/scipy/scipy/issues/6556
        if np.min(lons_rad) < 0: self._lon_corr = np.pi
        else: self._lon_corr = 0
        lons_rad += self._lon_corr

        # initialize lat-lon interpolator
        if reggrid:  # regular grid
            if len(lats) > 2 and len(lons) > 2:
                self.interp_ll = RectSphereBivariateSpline(u=lats_rad,
                                                           v=lons_rad,
                                                           r=values)
            elif len(lats) > 1 and len(lons) > 1:
                z = np.swapaxes(values, 0, 1)
                self.interp_ll = interp2d(x=lats_rad,
                                          y=lons_rad,
                                          z=z,
                                          kind='linear')
            elif len(lats) == 1:
                self.interp_ll = interp1d(x=lons_rad,
                                          y=np.squeeze(values),
                                          kind='linear')
            elif len(lons) == 1:
                self.interp_ll = interp1d(x=lats_rad,
                                          y=np.squeeze(values),
                                          kind='linear')

        else:  # irregular grid
            if len(np.unique(lats)) <= 1 or len(np.unique(lons)) <= 1:
                self.interp_ll = GridData2D(u=lats_rad,
                                            v=lons_rad,
                                            r=values,
                                            method='nearest')

            elif method_irreg == 'regularize':

                # initialize interpolators on irregular grid
                if len(np.unique(lats)) >= 2 and len(np.unique(lons)) >= 2:
                    method = 'linear'
                else:
                    method = 'nearest'
                gd = GridData2D(u=lats_rad,
                                v=lons_rad,
                                r=values,
                                method=method)
                gd_near = GridData2D(u=lats_rad,
                                     v=lons_rad,
                                     r=values,
                                     method='nearest')

                # determine bin size for regular grid
                lat_diffs = np.diff(np.sort(np.unique(lats)))
                lat_diffs = lat_diffs[lat_diffs > 1e-4]
                lon_diffs = np.diff(np.sort(np.unique(lons)))
                lon_diffs = lon_diffs[lon_diffs > 1e-4]
                bin_size = (np.min(lat_diffs), np.min(lon_diffs))

                # regular grid that data will be mapped to
                lats_reg, lons_reg = self._create_grid(lats=lats,
                                                       lons=lons,
                                                       bin_size=bin_size,
                                                       max_bins=bins_irreg_max)

                # map to regular grid
                lats_reg_rad, lons_reg_rad = torad(lats_reg, lons_reg)
                lons_reg_rad += self._lon_corr
                vi = gd(theta=lats_reg_rad, phi=lons_reg_rad, grid=True)
                vi_near = gd_near(theta=lats_reg_rad,
                                  phi=lons_reg_rad,
                                  grid=True)
                indices_nan = np.where(np.isnan(vi))
                vi[indices_nan] = vi_near[indices_nan]

                # initialize interpolator on regular grid
                self.interp_ll = RectSphereBivariateSpline(u=lats_reg_rad,
                                                           v=lons_reg_rad,
                                                           r=vi)

            else:
                self.interp_ll = GridData2D(u=lats_rad,
                                            v=lons_rad,
                                            r=values,
                                            method=method_irreg)

        # store data used for interpolation
        self.lat_nodes = lats
        self.lon_nodes = lons
        self.values = values

    def get_nodes(self):
        return (self.values, self.lat_nodes, self.lon_nodes)

    def interp_xy(self, x, y, grid=False, x_deriv_order=0, y_deriv_order=0):
        """ Interpolate using planar coordinate system (xy).

            x and y can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the positions (x_i, y_i), where x=(x_1,...,x_N) and
            y=(y_1,...,y_N). Note that in this case, x and y must have
            the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (x_i, y_j), where x=(x_1,...,x_N) and
            y=(y_1,...,y_M). Note that in this case, the lengths of x
            and y do not have to be the same.

            Args:
                x: float or array
                   x-coordinate of the positions(s) where the interpolation is to be evaluated
                y: float or array
                   y-coordinate of the positions(s) where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of x and y.
                x_deriv_order: int
                    Order of x-derivative
                y_deriv_order: int
                    Order of y-derivative

            Returns:
                zi: Interpolated interpolation values
        """
        lat, lon = XYtoLL(x=x,
                          y=y,
                          lat_ref=self.origin[0],
                          lon_ref=self.origin[1],
                          grid=grid)

        if grid:
            M = lat.shape[0]
            N = lat.shape[1]
            lat = np.reshape(lat, newshape=(M * N))
            lon = np.reshape(lon, newshape=(M * N))

        zi = self.interp(lat=lat,
                         lon=lon,
                         squeeze=False,
                         lat_deriv_order=y_deriv_order,
                         lon_deriv_order=x_deriv_order)

        if x_deriv_order + y_deriv_order > 0:
            r = DLDL_over_DXDY(lat=lat,
                               lat_deriv_order=y_deriv_order,
                               lon_deriv_order=x_deriv_order)
            zi *= r

        if grid:
            zi = np.reshape(zi, newshape=(M, N))

        if np.ndim(zi) == 2:
            zi = np.swapaxes(zi, 0, 1)

        zi = np.squeeze(zi)

        if np.ndim(zi) == 0 or (np.ndim(zi) == 1 and len(zi) == 1):
            zi = float(zi)

        return zi

    def interp(self,
               lat,
               lon,
               grid=False,
               squeeze=True,
               lat_deriv_order=0,
               lon_deriv_order=0):
        """ Interpolate using spherical coordinate system (latitude-longitude).

            lat and lot can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the coordinates (lat_i, lon_i), where lat=(lat_1,...,lat_N)
            and lon=(lon_1,...,lon_N). Note that in this case, lat and
            lon must have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (lat_i, lon_j), where lat=(lat_1,...,lat_N)
            and lon=(lon_1,...,lon_M). Note that in this case, the lengths
            of lat and lon do not have to be the same.

            Derivates are given per radians^n, where n is the overall
            derivative order.

            Args:
                lat: float or array
                    latitude of the positions(s) where the interpolation is to be evaluated
                lon: float or array
                    longitude of the positions(s) where the interpolation is to be evaluated
                grid: bool
                    Specify how to combine elements of lat and lon. If lat and lon have different
                    lengths, specifying grid has no effect as it is automatically set to True.
                lat_deriv_order: int
                    Order of latitude-derivative
                lon_deriv_order: int
                    Order of longitude-derivative

            Returns:
                zi: Interpolated values (or derivates)
        """
        lat = np.squeeze(np.array(lat))
        lon = np.squeeze(np.array(lon))
        lat_rad, lon_rad = torad(lat, lon)
        lon_rad += self._lon_corr

        if isinstance(self.interp_ll, interp2d):
            zi = self.interp_ll.__call__(x=lat_rad,
                                         y=lon_rad,
                                         dx=lat_deriv_order,
                                         dy=lon_deriv_order)
            if grid: zi = np.swapaxes(zi, 0, 1)
            if not grid and np.ndim(zi) == 2: zi = np.diagonal(zi)

        elif isinstance(self.interp_ll, interp1d):
            if len(self.lat_nodes) > 1:
                zi = self.interp_ll(x=lat_rad)
            elif len(self.lon_nodes) > 1:
                zi = self.interp_ll(x=lon_rad)

        else:
            zi = self.interp_ll.__call__(theta=lat_rad,
                                         phi=lon_rad,
                                         grid=grid,
                                         dtheta=lat_deriv_order,
                                         dphi=lon_deriv_order)

        if squeeze:
            zi = np.squeeze(zi)

        if np.ndim(zi) == 0 or (np.ndim(zi) == 1 and len(zi) == 1):
            zi = float(zi)

        return zi

    def _create_grid(self, lats, lons, bin_size, max_bins):
        """ Created regular lat-lon grid with uniform spacing that covers
            a set of (lat,lon) coordinates.

            Args:
                lats: numpy.array
                    Latitude values in degrees
                lons: numpy.array
                    Longitude values in degrees
                bin_size: float or tuple(float,float)
                    Lat and long bin size
                max_bins: int
                    Maximum number of bins along either axis

            Returns:
                : numpy.array, numpy.array
                    Latitude and longitude values of the regular grid
        """
        if isinstance(bin_size, (int, float)): bin_size = (bin_size, bin_size)
        res = []
        for v, dv in zip([lats, lons], bin_size):
            v_min = np.min(v) - dv
            v_max = np.max(v) + dv
            num = max(3, int((v_max - v_min) / dv) + 1)
            num = min(max_bins, num)
            v_reg = np.linspace(v_min, v_max, num=num)
            res.append(v_reg)

        return tuple(res)


class GridData3D():
    """ Interpolation of data on a three-dimensional irregular grid.

        Essentially, a wrapper function around scipy's interpolate.griddata.

        https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.griddata.html

        An alternative to griddata could be Rbf, as discussed here:

        https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy

        Attributes:
            u: 1d numpy array
                data points 1st coordinate
            v: 1d numpy array
                data points 2nd coordinate
            w: 1d numpy array
                data points 3rd coordinate
            r: 1d numpy array
                data values
            method : {‘linear’, ‘nearest’}, optional
    """

    def __init__(self, u, v, w, r, method='linear'):
        self.uvw = np.array([u, v, w]).T
        self.r = r
        self.method = method

    def __call__(self, theta, phi, z, grid=False):
        """ Interpolate data

            theta, phi, z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the coordinates (theta_i, phi_i, z_i), where theta=(theta_1,...,theta_N),
            phi=(phi_1,...,phi_N) and z=(z_,...,z_K). Note that in this case, theta, phi,
            z must all have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (theta_i, theta_j, z_k), where theta=(theta_1,...,theta_N),
            phi=(phi_1,...,phi_M) and z=(z_1,...,z_K). Note that in this case, the lengths
            of theta, phi, z do not have to be the same.

            Args:
                theta: float or array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: float or array
                   2nd coordinate of the points where the interpolation is to be evaluated
                z: float or array
                   3rd coordinate of the points where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of theta and phi.

            Returns:
                ri: Interpolated values
        """
        if grid: theta, phi, z, M, N, K = self._meshgrid(theta, phi, z)

        pts = np.array([theta, phi, z]).T
        ri = griddata(self.uvw, self.r, pts, method=self.method)

        if grid: ri = np.reshape(ri, newshape=(M, N, K))

        return ri

    def _meshgrid(self, theta, phi, z, grid=False):
        """ Create grid

            Args:
                theta: 1d numpy array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: 1d numpy array
                   2nd coordinate of the points where the interpolation is to be evaluated
                z: float or array
                   3rd coordinate of the points where the interpolation is to be evaluated

            Returns:
                theta, phi, z: 3d numpy array
                    Grid coordinates
                M, N, K: int
                    Number of grid values
        """
        M = 1
        N = 1
        K = 1
        if np.ndim(theta) == 1: M = len(theta)
        if np.ndim(phi) == 1: N = len(phi)
        if np.ndim(z) == 1: K = len(z)
        theta, phi, z = np.meshgrid(theta, phi, z)
        theta = theta.flatten()
        phi = phi.flatten()
        z = z.flatten()
        return theta, phi, z, M, N, K


class Interpolator3D():
    """ Class for interpolating 3D (lat,lon,depth) geospatial data.

        For irregular grids, the data values must be passed as a
        1d array and all three arrays (values, lats, lons, depths) must have
        the same length.

        For regular grids, the data values must be passed as a
        3d array with shape (M,N,K) where M,N,K are the lengths
        of the latitude, longitude, and depth arrays, respectively.

        Attributes:
            values: 3d numpy array
                Values to be interpolated
            lats: 1d numpy array
                Latitude values
            lons: 1d numpy array
                Longitude values
            depths: 1d numpy array
                Depth values
            origin: tuple(float,float)
                Reference location (origo of XY coordinate system).
            method : {‘linear’, ‘nearest’}, optional
                Interpolation method. Default is linear
            method_irreg : {‘linear’, ‘nearest’, ‘regularize’}, optional
                Interpolation method used for irregular grids.
                Note 'nearest' is usually significantly faster than 'linear''.
                If the 'regularize' is selected, the data is first mapped onto
                a regular grid by means of a linear interpolation (for points outside
                the area covered by the data, a nearest-point interpolation is used).
                The bin size of the regular grid is specified via the reg_bin argument.
            bins_irreg_max: int
                Maximum number of bins along either axis of the regular grid onto which
                the irregular data is mapped. Only relevant if method_irreg is set to
                'regularize'. Default is 200.
    """

    def __init__(self,
                 values,
                 lats,
                 lons,
                 depths,
                 origin=None,
                 method='linear',
                 method_irreg='regularize',
                 bins_irreg_max=200):

        # compute coordinates of origin, if not provided
        if origin is None: origin = center_point(lats, lons)
        self.origin = origin

        # check if bathymetry data are on a regular or irregular grid
        reggrid = (np.ndim(values) == 3)

        # convert to radians
        lats_rad, lons_rad = torad(lats, lons)

        # necessary to resolve a mismatch between scipy and underlying Fortran code
        # https://github.com/scipy/scipy/issues/6556
        if np.min(lons_rad) < 0: self._lon_corr = np.pi
        else: self._lon_corr = 0
        lons_rad += self._lon_corr

        # initialize lat-lon interpolator
        if reggrid:
            self.interp_ll = RegularGridInterpolator(
                (lats_rad, lons_rad, depths),
                values,
                method=method,
                bounds_error=False,
                fill_value=None)
        else:
            if method_irreg == 'regularize':

                # interpolators on irregular grid
                gd = GridData3D(u=lats_rad,
                                v=lons_rad,
                                w=depths,
                                r=values,
                                method='linear')
                gd_near = GridData3D(u=lats_rad,
                                     v=lons_rad,
                                     w=depths,
                                     r=values,
                                     method='nearest')

                # determine bin size for regular grid
                lat_diffs = np.diff(np.sort(np.unique(lats)))
                lat_diffs = lat_diffs[lat_diffs > 1e-4]
                lon_diffs = np.diff(np.sort(np.unique(lons)))
                lon_diffs = lon_diffs[lon_diffs > 1e-4]
                depth_diffs = np.diff(np.sort(np.unique(depths)))
                depth_diffs = depth_diffs[depth_diffs > 0.1]
                bin_size = (np.min(lat_diffs), np.min(lon_diffs),
                            np.min(depth_diffs))

                # regular grid that data will be mapped to
                lats_reg, lons_reg, depths_reg = self._create_grid(
                    lats=lats,
                    lons=lons,
                    depths=depths,
                    bin_size=bin_size,
                    max_bins=bins_irreg_max)

                # map to regular grid
                lats_reg_rad, lons_reg_rad = torad(lats_reg, lons_reg)
                lons_reg_rad += self._lon_corr
                vi = gd(theta=lats_reg_rad,
                        phi=lons_reg_rad,
                        z=depths_reg,
                        grid=True)
                vi_near = gd_near(theta=lats_reg_rad,
                                  phi=lons_reg_rad,
                                  z=depths_reg,
                                  grid=True)
                indices_nan = np.where(np.isnan(vi))
                vi[indices_nan] = vi_near[indices_nan]

                # interpolator on regular grid
                self.interp_ll = RegularGridInterpolator(
                    (lats_reg_rad, lons_reg_rad, depths_reg),
                    vi,
                    method=method,
                    bounds_error=False,
                    fill_value=None)

            else:
                self.interp_ll = GridData3D(u=lats_rad,
                                            v=lons_rad,
                                            w=depths,
                                            r=values,
                                            method=method_irreg)

        # store grids
        self.lat_nodes = lats
        self.lon_nodes = lons
        self.depth_nodes = depths
        self.values = values

    def get_nodes(self):
        return (self.values, self.lat_nodes, self.lon_nodes, self.depth_nodes)

    def interp_xy(self, x, y, z, grid=False):
        """ Interpolate using planar coordinate system (xy).

            x,y,z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the positions (x_i, y_i, z_i), where x=(x_1,...,x_N),
            y=(y_1,...,y_N), and z=(z_1,...,z_N). Note that in this case,
            x,y,z must have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (x_i, y_j, z_k), where x=(x_1,...,x_N),
            y=(y_1,...,y_M), and z=(z_1,...,z_K). Note that in this case, the
            lengths of x,y,z do not have to be the same.

            Args:
                x: float or array
                   x-coordinate of the positions(s) where the interpolation is to be evaluated
                y: float or array
                   y-coordinate of the positions(s) where the interpolation is to be evaluated
                z: float or array
                   Depth of the positions(s) where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of x,y,z.

            Returns:
                vi: Interpolated values
        """
        M = N = K = 1
        if np.ndim(y) == 1:
            M = len(y)
        if np.ndim(x) == 1:
            N = len(x)
        if np.ndim(z) == 1:
            K = len(z)

        lat, lon, z = XYtoLL(x=x,
                             y=y,
                             lat_ref=self.origin[0],
                             lon_ref=self.origin[1],
                             grid=grid,
                             z=z)

        if grid:
            lat = lat.flatten()
            lon = lon.flatten()
            z = z.flatten()

        vi = self.interp(lat=lat, lon=lon, z=z, squeeze=False)

        if grid:
            vi = np.reshape(vi, newshape=(M, N, K))

        if np.ndim(vi) == 3:
            vi = np.swapaxes(vi, 0, 1)

        vi = np.squeeze(vi)

        if np.ndim(vi) == 0 or (np.ndim(vi) == 1 and len(vi) == 1):
            vi = float(vi)

        return vi

    def interp(self, lat, lon, z, grid=False, squeeze=True):
        """ Interpolate using spherical coordinate system (lat-lon).

            lat,lot,z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the coordinates (lat_i, lon_i, z_i), where lat=(lat_1,...,lat_N),
            lon=(lon_1,...,lon_N) and z=(z_,...,z_K). Note that in this case, lat and
            lon must have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (lat_i, lon_j, z_k), where lat=(lat_1,...,lat_N),
            lon=(lon_1,...,lon_M) and z=(z_1,...,z_K). Note that in this case, the lengths
            of lat and lon do not have to be the same.

            Args:
                lat: float or array
                    Latitude of the positions(s) where the bathymetry is to be evaluated
                lon: float or array
                    Longitude of the positions(s) where the bathymetry is to be evaluated
                z: float or array
                    Depth of the positions(s) where the interpolation is to be evaluated
                grid: bool
                    Specify how to combine elements of lat,lon,z.

            Returns:
                zi: Interpolated bathymetry values (or derivates)
        """
        M = N = K = 1
        if np.ndim(lat) == 1:
            M = len(lat)
        if np.ndim(lon) == 1:
            N = len(lon)
        if np.ndim(z) == 1:
            K = len(z)

        lat = np.squeeze(np.array(lat))
        lon = np.squeeze(np.array(lon))
        lat_rad, lon_rad = torad(lat, lon)
        lon_rad += self._lon_corr

        z = np.squeeze(np.array(z))

        if grid:
            lat_rad, lon_rad, z = np.meshgrid(lat_rad, lon_rad, z)
            lat_rad = lat_rad.flatten()
            lon_rad = lon_rad.flatten()
            z = z.flatten()

        pts = np.column_stack((lat_rad, lon_rad, z))
        vi = self.interp_ll(pts)

        if grid:
            vi = np.reshape(vi, newshape=(M, N, K))

        if squeeze:
            vi = np.squeeze(vi)

        if np.ndim(vi) == 0 or (np.ndim(vi) == 1 and len(vi) == 1):
            vi = float(vi)

        return vi

    def _create_grid(self, lats, lons, depths, bin_size, max_bins):
        """ Created regular lat-lon-depth grid with uniform spacing that covers
            a set of (lat,lon,depth) coordinates.

            Args:
                lats: numpy.array
                    Latitude values in degrees
                lons: numpy.array
                    Longitude values in degrees
                depths: numpy.array
                    Depth valus in meters
                bin_size: tuple(float,float,float)
                    Bin size
                max_bins: int
                    Maximum number of bins along any axis

            Returns:
                : numpy.array, numpy.array, numpy.array
                    Lat, lon, and depth values of the regular grid
        """
        res = []
        for v, dv in zip([lats, lons, depths], bin_size):
            v_min = np.min(v) - dv
            v_max = np.max(v) + dv
            num = max(3, int((v_max - v_min) / dv) + 1)
            num = min(max_bins, num)
            v_reg = np.linspace(v_min, v_max, num=num)
            res.append(v_reg)

        return tuple(res)


class Uniform2D():

    def __init__(self, values, **other):
        self.value = values

    def get_nodes(self):
        return (self.value, None, None)

    def interp_xy(self, x, y, grid=False, x_deriv_order=0, y_deriv_order=0):

        z = self.interp(lat=y,
                        lon=x,
                        grid=grid,
                        squeeze=False,
                        lat_deriv_order=y_deriv_order,
                        lon_deriv_order=x_deriv_order)

        if np.ndim(z) == 3:
            z = np.swapaxes(z, 0, 1)

        z = np.squeeze(z)

        return z

    def interp(self,
               lat,
               lon,
               grid=False,
               squeeze=True,
               lat_deriv_order=0,
               lon_deriv_order=0):

        if np.ndim(lat) == 0: lat = np.array([lat])
        if np.ndim(lon) == 0: lon = np.array([lon])

        if grid:
            s = (len(lat), len(lon))

        else:
            assert len(lat) == len(
                lon
            ), 'when grid is False, lat and lon must have the same length'

            s = len(lat)

        if lat_deriv_order + lon_deriv_order > 0:
            v = 0
        else:
            v = self.value

        z = np.ones(s) * v

        if squeeze:
            z = np.squeeze(z)

        return z


class Uniform3D():

    def __init__(self, values, **other):
        self.value = values

    def get_nodes(self):
        return (self.value, None, None, None)

    def interp_xy(self, x, y, z, grid=False):

        v = self.interp(lat=y, lon=x, z=z, grid=grid, squeeze=False)

        if np.ndim(v) == 3:
            v = np.swapaxes(v, 0, 1)

        v = np.squeeze(v)

        return v

    def interp(self, lat, lon, z, grid=False, squeeze=True):

        if np.ndim(lat) == 0: lat = np.array([lat])
        if np.ndim(lon) == 0: lon = np.array([lon])
        if np.ndim(z) == 0: z = np.array([z])

        if grid:
            s = (len(lat), len(lon), len(z))

        else:
            assert len(lat) == len(lon) == len(
                z), 'when grid is False, lat,lon,z must have the same length'

            s = len(lat)

        v = np.ones(s) * self.value

        if squeeze:
            v = np.squeeze(v)

        return v


class DepthInterpolator3D():
    """ Class for interpolating 3D (lat,lon,depth) geospatial data
        that only varies with depth.

        Attributes:
            values: 1d or 3d numpy array
                Values to be interpolated
            depths: 1d numpy array
                Depth values
            method : {‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’}, optional
                Interpolation method. Default is quadratic.
    """

    def __init__(self, values, depths, method='quadratic'):

        if np.ndim(values) == 3:
            values = values[0, 0, :]

        # initialize 1d interpolator
        self.interp1d = interp1d(x=depths,
                                 y=values,
                                 kind=method,
                                 fill_value="extrapolate")

        # store interpolation data
        self.depth_nodes = depths
        self.values = values

    def get_nodes(self):
        return (self.values, self.depth_nodes)

    def interp_xy(self, x, y, z, grid=False):
        """ Interpolate using planar coordinate system (xy).

            x,y,z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the positions (x_i, y_i, z_i), where x=(x_1,...,x_N),
            y=(y_1,...,y_N), and z=(z_1,...,z_N). Note that in this case,
            x,y,z must have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (x_i, y_j, z_k), where x=(x_1,...,x_N),
            y=(y_1,...,y_M), and z=(z_1,...,z_K). Note that in this case, the
            lengths of x,y,z do not have to be the same.

            Args:
                x: float or array
                   x-coordinate of the positions(s) where the interpolation is to be evaluated
                y: float or array
                   y-coordinate of the positions(s) where the interpolation is to be evaluated
                z: float or array
                   Depth of the positions(s) where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of x,y,z.

            Returns:
                vi: Interpolated values
        """
        v = self.interp(lat=y, lon=x, z=z, grid=grid, squeeze=False)

        if np.ndim(v) == 3:
            v = np.swapaxes(v, 0, 1)

        v = np.squeeze(v)
        return v

    def interp(self, lat, lon, z, grid=False, squeeze=True):
        """ Interpolate using spherical coordinate system (lat-lon).

            lat, lon, z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at
            the coordinates (lat_i, lon_i, z_i), where lat=(lat_1,...,lat_N),
            lon=(lon_1,...,lon_N) and z=(z_,...,z_K). Note that in this case, lat and
            lon must have the same length.

            If grid is set to True, the interpolation will be evaluated at
            all combinations (lat_i, lon_j, z_k), where lat=(lat_1,...,lat_N),
            lon=(lon_1,...,lon_M) and z=(z_1,...,z_K). Note that in this case, the lengths
            of lat and lon do not have to be the same.

            Args:
                lat: float or array
                    Latitude of the positions(s) where the bathymetry is to be evaluated
                lon: float or array
                    Longitude of the positions(s) where the bathymetry is to be evaluated
                z: float or array
                    Depth of the positions(s) where the interpolation is to be evaluated
                grid: bool
                    Specify how to combine elements of lat,lon,z.

            Returns:
                zi: Interpolated bathymetry values (or derivates)
        """
        if np.ndim(lat) == 0: lat = np.array([lat])
        if np.ndim(lon) == 0: lon = np.array([lon])
        if np.ndim(z) == 0: z = np.array([z])

        if grid:
            s = (len(lat), len(lon), len(z))

        else:
            assert len(lat) == len(lon) == len(
                z), 'when grid is False, lat,lon,z must have the same length'

            s = len(lat)

        #v = self.interp(z)
        v = self.interp1d(z)

        if grid:
            v = np.ones(s) * v[np.newaxis, np.newaxis, :]

        if squeeze:
            v = np.squeeze(v)

        return v
