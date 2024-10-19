import logging
import numpy as np
import traceback
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from scipy.interpolate import griddata, RectSphereBivariateSpline, RegularGridInterpolator
from kadlu.utils import as_array, center_point, reverse_index_map, torad, deg2rad, XYtoLL, DLDL_over_DXDY


'''
    TODO: 
     [ ] complete docstrings
     [ ] rescale depth values relative to lat/lon so that "nearest" point is nearest in the sense of euclidian distance ?
'''


""" 
    Names and ordering of dimensions used by all interpolation classes in Kadlu
"""
GEOSPATIAL_DIMS = ["lat", "lon", "epoch", "depth"]


def get_interpolator(value, **kwargs):
    """ Convenience function for initialising a interpolator for a given set of data.

        Args:
            value: array-like
                Data values

        Keyword args:                
            lat: array-like
                Latitudes in degrees. 
            lon: array-like
                Longitudes in degrees. 
            depth: array-like
                Depths in meters below the sea surface. 
            epoch: array-like
                Time in hours since 2000-01-01 00:00:00. 
            origin: tuple(float,float)
                Reference location used as the origo of XY coordinate system. If not specified, 
                the center point of the lat,lon coordinates will be used.
            method: str
                Preferred interpolation method. Allowed values are: nearest, linear, slinear, cubic.
                For high-dimensional data with large numbers of nodes (> 1E6), it is recommended to 
                use one of the two simpler interpolation methods (nearest or linear) to reduce memory 
                usage.
                For data that require interpolation in latitude and longitude only (e.g. bathymetry) 
                the default interpolation method is a bivariate spline method adapted specifically to 
                spherical coordinates with proper handling of the poles and anti-meridian discontinuity.
            name: str
                Name used to identify the interpolator. Optional.
            max_size: int
                Maximum size (no. bins) along any of the axes of the regular interpolation grid. 
                Optional. Only relevant if input data is on an irregular grid.
            grid_shape: dict()
                Shape of the regular interpolation grid. Optional. Only relevant if input data is on an irregular grid.
            bin_size: dict()
                Bin sizes of the regular interpolation grid. Optional. Overwrites `grid_shape`.
                Only relevant if input data is on an irregular grid.
            irreg_method: str
                Interpolation method used for mapping data from an irregular to a regular grid. 
                Options are `nearest` and `linear`. Default is `nearest`.

        Returns:    
            interp: RegularGridGeospatialInterpolator
                Interpolator
    """

    logger = logging.getLogger("kadlu")

    # convert coordinates to numpy arrays
    coors, _ = _coordinates_as_arrays(kwargs)

    kwargs.update(coors)

    # we will only interpolate dimensions with 2 or more points
    dims, _ = _dim_sizes(coors, min_size=2)

    # if data are on a regular grid, return an instance of the RegularGridGeospatialInterpolator    
    if np.ndim(value) > 1 or len(dims) <= 1:
        return RegularGridGeospatialInterpolator(value, **kwargs)

    # if data are on an irregular grid, map them to a regular grid first using the IrregularGridGeospatialInterpolator class
    reg_grid, coverage = _create_regular_grid(return_coverage=True, **kwargs)

    # debug 
    dims, dim_sizes = _dim_sizes(reg_grid)
    debub_msg = f"Created regular grid in {dims} with shape {dim_sizes} and {coverage*100:.1f}% coverage"
    logger.debug(debub_msg)

    # if coverage is low, issue a warning
    if coverage < 0.5:
        warn_msg = f"The data points only cover {coverage*100:.1f}% of the full rectangular box"
        logger.warning(warn_msg)

    # method and name for interpolator on regular grid
    method = kwargs.pop("method", None)
    name = kwargs.pop("name", None)

    # method and name for interpolator on irregular grid
    irreg_method = kwargs.pop("irreg_method", "nearest")
    irreg_name = "irreg"
    if name is not None:
        irreg_name += "_" + name

    # map from irregular -> regular grid using 'linear' interpolation
    irreg_interp = IrregularGridGeospatialInterpolator(value, method=irreg_method, name=irreg_name, **kwargs)
    v = irreg_interp(**reg_grid, grid=True)

    logger.debug(f"Mapped {len(v) if np.ndim(v) > 0 else 1} values to regular grid using method `{irreg_interp.method}`")

    # return an instance of the RegularGridGeospatialInterpolator on the new, regular grid
    return RegularGridGeospatialInterpolator(value=v, method=method, **reg_grid, name=name)


def _dim_sizes(coors, min_size=1):
    """
    
        TODO: function returns a single dict() argument with key=dim and value=dim_size
    """

    dims, dim_sizes = [], []
    for dim in GEOSPATIAL_DIMS:
        if dim in coors:
            siz = len(np.unique(coors[dim]))
            if siz >= min_size:
                dims.append(dim)
                dim_sizes.append(siz)

    return dims, dim_sizes


def _derivative_orders(kwargs):
    """ Helper function for extracting derivative orders dlat, dlon, ddepth, depoch
        from a set of keyword arguments.
    """
    deriv_orders = {dim: 0 for dim in GEOSPATIAL_DIMS}
    for k,v in kwargs.items(): 
        if k[0] == "d" and k[1:] in GEOSPATIAL_DIMS:
            deriv_orders[k[1:]] = 0 if v is None else v

    return deriv_orders


def _coordinates_as_arrays(kwargs, dtype=np.float64):
    """ Helper function for extracting lat,lon,depth,epoch coordinates 
        from a set of keyword arguments and recasting them as numpy arrays.

        Args:
            kwargs: dict
                The dictionary containing the coordinate arrays
            dtype: 
                The desired data type of the numpy arrays. Default is np.float64

        Returns:
            arrays: dict
                The coordinate arrays, recasted as numpy arrays
            scalar: bool
                True if all cordinates had scalar values, and False otherwise.
    """
    arrays = dict()
    scalar = True
    for k in GEOSPATIAL_DIMS:
        v = kwargs.get(k, None)
        if v is not None:
            scalar *= (np.ndim(v) == 0)
            arrays[k] = as_array(v, dtype=dtype)

    return arrays, scalar


def _assert_same_size(*ai):
    """ Helper function for asserting that a set of arrays have the same size"""
    for a in ai:
        assert a.shape == ai[0].shape, "all arrays must have the same size"


def _reshape(a, dims, coors, grid, scalar):
    """ Helper function for reshaping the interpolation output.

        Args:
            a: numpy array
                The input array
            dims: list(str)
                The names of the dimensions of the input array
            coors: dict
                The coordinate arrays
            grid: bool
                How to combine coordinate elements. If False (default) the coordinate arrays must have matching lengths.
            scalar: bool
                True if all coordinates were specified as scalar values, False otherwise.

        Returns:
            a: numpy array
                The reshaped array
    """
    # make a copy so we don't modify the input argument
    dims = dims.copy()

    # dimensionless input
    if dims is None or len(dims) == 0:
        assert np.ndim(a) == 0 or a.shape == (1,), "input array and `dims` have incompatible shapes"
        
        if coors is None or len(coors) == 0:
            return np.squeeze(a)

        # output shape should match coordinate arrays
        if grid:
            shape = np.meshgrid(*coors.values(), indexing="ij")[0].shape
        else:
            shape = len(list(coors.values())[0])

        a = np.ones(shape) * np.squeeze(a)

        if scalar:
            a = np.squeeze(a)

        return a

    # array input
    assert len(coors) > 0, "at least one coordinate array required for reshaping"

    if not grid:
        _assert_same_size(a, *coors.values())
        if scalar:
            a = float(np.squeeze(a))

        return a

    assert len(a.shape) == len(dims), f"input array and `dims` have incompatible shapes: a.shape={a.shape}, len(dims)={len(dims)}"

    # expand the array by adding new dimensions
    for k,v in coors.items():
        if k not in dims:
            new_shape = a.shape + (len(v),)
            a = np.expand_dims(a, axis=-1)  #adds new dimension of shape (1,)
            a = np.broadcast_to(a, new_shape)  #expands the new dimension to the desired size (by replicating the array)
            dims.append(k)

    # re-order the axes to match the specified ordering
    indices = [dims.index(d) for d in GEOSPATIAL_DIMS if d in dims]
    a = np.moveaxis(a, indices, np.arange(len(indices)))

    # always return a (Python) scalar value for scalar coordinates
    if scalar:
        a = float(np.squeeze(a))

    return a


def _complete_grid(**kwargs):
    """ Helper function for adding depth and time dimensions to a lat-lon grid
    
        Args:
            lat: 2d numpy array
                The latitude coordinates of the lat-lon grid
            lon: 2d numpy array
                The longitude coordinates of the lat-lon grid
            depth: array-like
                The depth coordinates, in meters
            epoch: array-like
                The time coordinates, in epoch hours since 2000-01-01 00:00:00
        
        Returns:
            flattened_coors: dict
                The flattened coordinate arrays
            grid_shape: tuple
                The (lat,lon,depth,epoch) grid shape    
    """
    args, _ = _coordinates_as_arrays(kwargs)

    coors = {
        "lat": args.pop("lat"),
        "lon": args.pop("lon"),
    }

    # loop ove new dimensions (depth, epoch)
    for new_dim in args.keys():
        u = args[new_dim]

        for dim in coors.keys():
            # expand existing dimensions
            v = coors[dim]
            new_shape = v.shape + (len(u),)
            v = np.expand_dims(v, axis=-1) 
            coors[dim] = np.broadcast_to(v, new_shape)  

        # expand the new dimension            
        axis = tuple(np.arange(len(coors["lat"].shape)))
        coors[new_dim] = np.ones(coors["lat"].shape) * np.expand_dims(u, axis=axis)

    # the full grid shape
    grid_shape = coors["lat"].shape

    # flattened coordinate arrays
    flattened_coors = {k: v.flatten() for k,v in coors.items()}

    return flattened_coors, grid_shape


class BaseGeospatialInterpolator():
    """ Parent class for all interpolators.

        Child classes must implement the `_eval` method.        

        Interpolators accept the following coordinates,

            * lat: latitude in degrees
            * lon: longitudes in degrees
            * depth: depth in meters below the sea surface
            * epoch: time in hours since 2000-01-01 00:00:00
    
        Args:
            value: array-like
                Data values
            lat: array-like
                Latitudes in degrees. 
            lon: array-like
                Longitudes in degrees. 
            depth: array-like
                Depths in meters below the sea surface. 
            epoch: array-like
                Time in hours since 2000-01-01 00:00:00. 
            name: str
                Name used to identify the interpolator. 
            origin: tuple(float,float)
                Reference location used as the origo of XY coordinate system. If not specified, 
                the center point of the lat,lon coordinates will be used.

        Attrs:
            value: numpy array
                Data values
            coordinates: dict
                Coordinates arrays
    """
    def __init__(
        self, 
        value,
        name="GeospatialInterpolator",
        origin=None,
        **coors,
    ):
        self.name = name
        self.logger = logging.getLogger("kadlu")

        # convert values and coordinates to numpy arrays
        self.value = as_array(value)
        self.coordinates, _ = _coordinates_as_arrays(coors)

        # lat,lon reference location for XY coordinate system
        self.origin = origin

        if self.origin is None:
            self.origin = center_point(self.coordinates.get("lat"), self.coordinates.get("lon"))

        if self.origin is None:
            self.origin = (0, 0)

    def __call__(self, **kwargs):
        """ Evaluate the interpolation at the given coordinate(s)

            Location coordinates can be specified either as (lat,lon) or as 
            (x,y) where x and y refer to the E-W and N-S axes of a planar 
            coordinate system centered at the `origin` reference location.

            Args:
                lat: array-like
                    Latitudes in degrees.
                lon: array-like
                    Longitudes in degrees.
                depth: array-like
                    Depths in meters below the sea surface. 
                epoch: array-like
                    Time in hours since 2000-01-01 00:00:00. 
                x: array-like
                   x coordinates in meters. 
                y: array-like
                   y coordinate in meters. 
                grid: bool
                    How to combine coordinate elements. If False (default) the coordinate arrays must have matching lengths.      
                origin: tuple(float,float)
                    Reference location for XY coordinate system. If not specified, the reference location specified at 
                    initialisation is used. Only relevant if location coordinates are specified as x,y instead of lat,lon.
                dlat: int
                    Order of derivative in latitude.
                dlon: int
                    Order of derivative in longitude.
                ddepth: int
                    Order of derivative in depth.
                depoch: int
                    Order of derivative in time.
                dx: int
                    Order of derivative in x.
                dy: int
                    Order of derivative in y.
        """
        if "x" in kwargs or "y" in kwargs: 
            return self._eval_xy(**kwargs)
    
        else:
            return self._eval(**kwargs)

    def _eval(self, **kwargs):
        """ Must be implemented in child class """
        pass

    def _eval_xy(self, **kwargs):
        kwargs = kwargs.copy()

        grid = kwargs.pop("grid", False)
        origin = kwargs.pop("origin", self.origin)

        # map x,y to lat,lon
        lat, lon = XYtoLL(
            x=kwargs.pop("x", 0),
            y=kwargs.pop("y", 0),
            lat_ref=origin[0],
            lon_ref=origin[1],
            grid=grid
        )

        kwargs.update({"lat": lat, "lon": lon})

        # complete lat-lon grid by including depth and time dimensions
        if grid:
            flattened_coors, grid_shape = _complete_grid(**kwargs)
            kwargs.update(flattened_coors)

        # pass flattened coordinate arrays to lat-lon interpolation method
        v = self._eval(grid=False, **kwargs)

        # derivatives
        kwargs["dlat"] = kwargs.get("dy", 0)
        kwargs["dlon"] = kwargs.get("dx", 0)

        # for derivates, we must multiply by the appropriate Jacobian
        d = _derivative_orders(kwargs)
        if d["lat"] + d["lon"] > 0:
            # convert from deg^-1 to rad^-1
            v /= np.power(deg2rad, d["lat"] + d["lon"])

            # multiply by jacobian
            j = DLDL_over_DXDY(
                lat=lat,
                lat_deriv_order=d["lat"],
                lon_deriv_order=d["lon"],
            )
            v *= j

        # reshape output to desired shape
        if grid:
            v = np.reshape(v, newshape=grid_shape)
            v = np.swapaxes(v, 0, 1)

        return v


class RegularGridGeospatialInterpolator(BaseGeospatialInterpolator):
    """

        Args:
            method: str
                For interpolation of data on a 2d lat-lon grid, it is recommended to 
                leave `method` unspecified (i.e. method=None)

    """
    def __init__(
        self, 
        value,
        name="RegularGridGeospatialInterpolator",
        origin=None,
        method=None,
        **coors,
    ):
        super().__init__(value=value, name=name, origin=origin, **coors)

        # deduce the shape of the interpolation grid from the sizes of the coordinate arrays
        grid_shape = tuple([
            len(self.coordinates[dim]) for dim in GEOSPATIAL_DIMS if dim in self.coordinates
        ])

        # we will only interpolate dimensions with 2 or more points
        self.dims, self.dim_sizes = _dim_sizes(self.coordinates, min_size=2)

        # validate array shapes
        assert_msg = f"value and coordinate arrays have incompatible shapes,  value:{self.value.shape}, coordinates:{grid_shape}"
        assert np.ndim(value) == 0 or self.value.shape == grid_shape or self.value.shape == tuple(self.dim_sizes), assert_msg

        debug_msg = f"[{self.name}] Detected {len(self.dims)} dimensions that require interpolation: {self.dims}"
        self.logger.debug(debug_msg)

        # grab dimensions that require interpolation
        points = [v for k,v in self.coordinates.items() if k in self.dims]
        
        # drop dimensions that don't require interpolation
        values = np.squeeze(self.value)

        # if lat,lon are the only dimensions requiring interpolation, and each has at least 3 points, we can use scipy's RectSphereBivariateSpline
        if self.dims == ["lat", "lon"] and np.all(np.array(self.dim_sizes) >= 3) and method == None:
            self._interp = _RectSphereBivariateSpline(
                points=[self.coordinates["lat"], self.coordinates["lon"]],
                values=values
            )
        
        # in all other cases, we use scipy's RegularGridInterpolator
        elif len(self.dims) > 0:
            self._interp = _RegularGridInterpolator(
                points=points,
                values=values,
                method="linear" if method is None else method,
                bounds_error=False,
                fill_value=None,  #values outside the interpolation domain will be extrapolated
            )

        # except in the trivial case where no dimensions require interpolation
        else:
            self._interp = _ConstantInterpolator(values)

    @property
    def method(self):
        return self._interp.method

    def _eval(self, grid=False, **kwargs):
        # coordinates where interpolation is to be evaluated
        coors, scalar = _coordinates_as_arrays(kwargs)

        # check that the coordinates required for evaluating the interpolation were provided
        for dim in self.dims:
            assert dim in coors, f"[{self.name}] `{dim}` required for evaluating interpolation"

        # check that coordinate arrays have same size
        # TODO: move this check up to the BaseGeospatialInterpolator ?
        if not grid:
            _assert_same_size(*coors.values())

        # grab dimensions that require interpolation
        xi = [v for k,v in coors.items() if k in self.dims]

        # pass to interpolator
        v = self._interp(xi, grid=grid, **kwargs)

        # reshape output to desired shape
        v = _reshape(v, dims=self.dims, coors=coors, grid=grid, scalar=scalar)

        return v


class IrregularGridGeospatialInterpolator(BaseGeospatialInterpolator):
    """
    
        Attrs:
            dims: list(str)
                Names of the dimensions that are being interpolated
    """
    def __init__(
        self, 
        value,
        name="IrregularGridGeospatialInterpolator",
        origin=None,
        method="cubic", 
        **coors,
    ):
        super().__init__(value=value, name=name, origin=origin, **coors)

        # coordinate arrays must have the same size
        _assert_same_size(*self.coordinates.values())

        # we will only interpolate dimensions with 2 or more unique values
        self.dims, _ = _dim_sizes(self.coordinates, min_size=2)
        debug_msg = f"[{self.name}] Detected {len(self.dims)} dimensions that require interpolation: {self.dims}"
        self.logger.debug(debug_msg)

        # stack that coordinates that require interpolation as an array with shape (num_pts, num_dims)
        if len(self.dims) > 0:
            points = np.stack([v for k,v in self.coordinates.items() if k in self.dims], axis=-1)
            self._interp = _IrregularGridInterpolator(points, self.value, method=method)

        else:
            self._interp = _ConstantInterpolator(self.value)

    @property
    def method(self):
        return self._interp.method

    def _eval(self, grid=False, replace_nan=True, method=None, **kwargs):
        # coordinates where interpolation is to be evaluated
        coors, scalar = _coordinates_as_arrays(kwargs)

        # check that the coordinates required for evaluating the interpolation were provided
        for dim in self.dims:
            assert dim in coors, f"{self.name} `{dim}` required for evaluating interpolation"

        # check that coordinate arrays have same size
        # TODO: move this check up to the BaseGeospatialInterpolator ?
        if not grid:
            _assert_same_size(*coors.values())
                
        # grab dimensions that require interpolation
        xi = [v for k,v in coors.items() if k in self.dims]

        # pass to interpolator
        v = self._interp(xi, grid=grid, replace_nan=replace_nan, method=method, **kwargs)

        # reshape output to desired shape
        v = _reshape(v, dims=self.dims, coors=coors, grid=grid, scalar=scalar)
        
        return v


class _RegularGridInterpolator(RegularGridInterpolator):
    """ Helper class for RegularGridGeospatialInterpolator
    
    """

    def __init__(self, points, values, **kwargs):
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("kadlu")

        # validate method
        method = kwargs.pop("method", "slinear")
        self.method = self._validate_method(method, points)

        # initialize interpolator
        super().__init__(
            points=points,
            values=values,
            method=self.method,
            **kwargs
        )

    def _validate_method(self, method, points):
        """ Helper function for validating the requested interpolation method
        
            Args:
                method: str
                    Interpolation method.
                points: tuple(array)
                    Coordinate arrays

            Returns:
                method: str
                    Same as input if validation was successful; otherwise returns an 
                    alternative method compatible with the size of the coordinate arrays

            Raises:
                AssertionError: if the requested method is invalid
        """ 
        # minimum no. data points required for each of the interpolation methods offered by scipy's RegularGridInterpolator
        self.min_data_pts = {
            "nearest": 1,
            "slinear": 2,
            "linear": 2,
            "cubic": 4,
            #"quintic": 6,
        }

        assert_msg = f"[{self.name}] Invalid interpolation method `{method}`. Valid options are: {list(self.min_data_pts.keys())}."
        assert method in self.min_data_pts, assert_msg

        # check that we have sufficient points for the chosen interpolation method
        # automatically switch to an alternative method that meets the requirement, if needed
        min_siz = np.min([len(arr) for arr in points])
        if min_siz < self.min_data_pts[method]:
            sorted_dict = sorted(self.min_data_pts.items(), key=lambda item: item[1], reverse=True)
            for k,v in sorted_dict:
                if v <= min_siz:
                    alt_method = k 
                    break          

            warn_msg = f"[{self.name}] Too few points for `{method}` interpolation. Switching to `{alt_method}`"
            self.logger.warning(warn_msg)
            method = alt_method

        return method

    def __call__(self, xi, grid=False, **kwargs):
        n_deriv = np.sum([n for n in _derivative_orders(kwargs).values()])
        if n_deriv > 0:
            err_msg = f"[{self.name}] Interpolation of derivatives only implemented for spherical (lat,lon) geometries"
            raise NotImplementedError(err_msg)

        if grid and len(xi) > 1:
            xi = np.meshgrid(*xi, indexing="ij")
            grid_shape = xi[0].shape
            xi = [x.flatten() for x in xi]

        pts = np.array(xi).T
        v = super().__call__(pts)

        if grid and len(xi) > 1: 
            # reshape flattened array back to grid shape
            v = np.reshape(v, newshape=grid_shape)

        return v


class _RectSphereBivariateSpline(RectSphereBivariateSpline):
    """ Helper class for RegularGridGeospatialInterpolator
    
    """

    def __init__(self, points, values, **kwargs):
        u, v = torad(points[0], points[1])

        super().__init__(
            u=u,
            v=v,
            r=values,
            **kwargs
        )

        self.method = "spline"

    def __call__(self, xi, grid=False, dlat=0, dlon=0, **kwargs):
        lats_rad, lons_rad = torad(xi[0], xi[1])

        # when `grid` is True, we must ensure latitudes and longitudes are strictly increasing
        if grid:
            lat_indices = np.argsort(lats_rad)
            lats_rad = lats_rad[lat_indices]
            lon_indices = np.argsort(lons_rad)
            lons_rad = lons_rad[lon_indices]

        # evaluate the interpolation function
        v = super().__call__(
            theta=lats_rad,
            phi=lons_rad,
            grid=grid,
            dtheta=dlat,
            dphi=dlon
        )

        # convert derivatives from rad^-1 to deg^-1
        if dlat + dlon > 0:
            v *= np.power(deg2rad, dlat + dlon)

        if grid:
            # reverse the index mapping so lat,lon values appear in their original order
            indices = reverse_index_map(lat_indices)
            v = v[indices]
            v = np.swapaxes(v, 0, 1)
            indices = reverse_index_map(lon_indices)
            v = v[indices]
            v = np.swapaxes(v, 0, 1)

        return v


class _IrregularGridInterpolator():
    """ Helper class for IrregularGridGeospatialInterpolator
    
    """

    def __init__(
        self, 
        points,
        values, 
        method="cubic", 
        fill_value=np.nan,
        batch_size=10000,
        **kwargs,
    ):
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("kadlu")

        self.points = points
        self.values = values

        # available interpolation methods, in order of preference
        self.methods = ["cubic", "linear", "nearest"]
        
        # `cubic` only available for 1D and 2D data
        if self.points.shape[1] > 2: 
            del self.methods[0]
            if method == "cubic":
                method = "linear"

        self._validate_method(method)

        self.method = method
        self.fill_value = fill_value
        self.batch_size = batch_size

    def _validate_method(self, method):
        """ Helper function for validating the requested interpolation method
        
            Args:
                method: str
                    Interpolation method.

            Raises:
                AssertionError: if the requested method is invalid
        """ 
        assert_msg = f"[{self.name}] Invalid interpolation method `{method}`. Valid options are: {self.methods}."
        assert method in self.methods, assert_msg

    def __call__(self, xi, grid=False, replace_nan=True, method=None, **kwargs):
        n_deriv = np.sum([n for n in _derivative_orders(kwargs).values()])
        if n_deriv > 0:
            err_msg = f"[{self.name}] Interpolation of derivatives not implemented for irregular grids"
            raise NotImplementedError(err_msg)

        if method is None:
            method = self.method

        self._validate_method(method)

        if grid and len(xi) > 1:
            xi = np.meshgrid(*xi, indexing="ij")
            grid_shape = xi[0].shape
            xi = [x.flatten() for x in xi]

        pts = np.array(xi).T

        # alternative interpolation methods to try, if the selected one fails
        alt_methods = self.methods.copy()
        alt_methods.remove(method)

        counter = 0 #attempt counter
        while True:
            try:
                # evaluate the interpolation function
                # (split into smaller batches to not use excessive memory)

                debug_msg = f"[{self.name}] Interpolating {pts.shape[0]} data points on {pts.shape[1]}D irregular grid"\
                    + f" using method `{method}` and batch size of {self.batch_size}"
                self.logger.debug(debug_msg)

                v = self._batched_eval(pts, method)

                break

            except (Exception, QhullError):
                msg = f"[{self.name}] Interpolation on irregular grid with method=`{method}` failed. To view full error report, re-run in DEBUG mode." 
                self.logger.debug(traceback.format_exc())

                # try again, with a different interpolation method; continue until successful or run out of options
                if counter < len(alt_methods):
                    method = alt_methods[counter]
                    counter += 1

                    msg += f" Switching to method=`{method}`."
                    self.logger.warning(msg)

                else:
                    raise Exception(msg)
                
        self.method = method

        # replace NaN values with nearest valid value
        if replace_nan and method != "nearest":
            indices_nan = np.where(np.isnan(v))
            nearest_value = self._batched_eval(pts, "nearest")
            v[indices_nan] = nearest_value[indices_nan]

        # if griddata return a 2d array with same (num_points, 1) we drop the last axis
        if np.ndim(v) == 2:
            v = v[:, 0]

        if grid and len(xi) > 1: 
            # reshape flattened array back to grid shape
            v = np.reshape(v, newshape=grid_shape)

        return v

    def _batched_eval(self, pts, method):
        """ Helper function for splitting evaluation of griddata function into multiple smaller requests"""
        n_batches = int(np.ceil(pts.shape[0] / self.batch_size))
        v = []
        if n_batches > 1:
            info_msg = f"Interpolating {pts.shape[0]} data points on {pts.shape[1]}D irregular grid"\
                + f" using method `{method}` and batch size of {self.batch_size}"
            self.logger.info(info_msg)

        for i in tqdm(range(n_batches), disable = n_batches < 2):
            a = i * self.batch_size
            b = a + self.batch_size
            vi = griddata(self.points, self.values, pts[a:b], method=method, fill_value=self.fill_value, rescale=False)
            v.append(vi)

        v = np.concatenate(v, axis=0)
        return v


class _ConstantInterpolator():
    """ Convenience class for handling constant/uniform data.
    """

    def __init__(self, value):
        self.value = np.squeeze(value)
        self.method = None

    def __call__(self, *args, **kwargs):
        n_deriv = np.sum([n for n in _derivative_orders(kwargs).values()])
        if n_deriv > 0:
            return 0
        else:
            return self.value


def _create_regular_grid(max_size=100, grid_shape=None, bin_size=None, return_coverage=False, **kwargs):
    """ Creates regular grid with uniform spacing that covers a set of (lat,lon,depth,epoch) coordinates.

        If grid_shape=None and bin_size=None, the grid spacing is set to the smallest non-zero difference 
        between two coordinate values. If necessary, the grid spacing is inflated so as to not exceed the 
        specified max size.

        Args:
            max_size: int
                Maximum size (no. bins) along any of the axes of the grid. Optional. Default is 100.
            grid_shape: dict()
                Grid shape. Optional.
            bin_size: dict()
                Bin sizes. Optional. Overwrites `grid_shape`.
            return_coverage: bool
                Also compute the grid's *coverage*. The coverage is a number between 0-1 that indicates how 
                well the input coordinates cover the volume spanned by the regular grid. It is computed as 
                the  ratio of the convex hull the input coordinates to the volume of the created grid. 
                
        Keyword args:                
            lat: array-like
                Latitudes in degrees. 
            lon: array-like
                Longitudes in degrees. 
            depth: array-like
                Depths in meters below the sea surface. 
            epoch: array-like
                Time in hours since 2000-01-01 00:00:00. 

        Returns:
            reg_coors: dict
                The coordinates of the regular grid
            coverage: float
                Only returned if `return_coverage=True`
    """
    # coordinate arrays, ordered as lat,lon,depth,epoch
    coors, _ = _coordinates_as_arrays(kwargs)

    # coordinate min/max values
    coors_range = {k: [np.min(v), np.max(v)] for k,v in coors.items()}

    # if the grid shape is not specified, use the smallest non-zero coordinate 
    # spacing as bin size, inflated if necessary to not exceed the max size
    if grid_shape is None:
        grid_shape = dict()

        for k,arr in coors.items():
            unique_values = np.unique(arr)

            if len(unique_values) == 1:
                n_nodes = 1

            else:
                start, stop = coors_range[k][0], coors_range[k][1]
                if bin_size is None:
                    step = np.min(np.diff(np.sort(unique_values)))
                else:
                    step = bin_size[k]
                
                n_nodes = int((stop - start) / step) + 1

            if max_size is not None:
                n_nodes = min(max_size, n_nodes)

            grid_shape[k] = n_nodes 

    # regularly spaced coordinates
    reg_coors = dict()
    for k,arr in coors.items():
        start, stop = coors_range[k][0], coors_range[k][1]
        num = grid_shape[k]
        reg_coors[k] = np.unique(np.linspace(start=start, stop=stop, num=num))

    if not return_coverage:
        return reg_coors

    # compute "coverage"
    
    # only consider dimensions with at least two unique values
    dims, _ = _dim_sizes(reg_coors, min_size=2)

    # volume of the regular grid
    grid_vol = 1
    for k in dims:
        start, stop = coors_range[k][0], coors_range[k][1]
        grid_vol *= (stop - start)

    # convex hull of the input coordinates
    pts = np.array([arr for k,arr in coors.items() if k in dims])
    pts = np.swapaxes(pts, 0, 1)
    h = ConvexHull(pts)

    coverage = h.volume / grid_vol

    return reg_coors, coverage        