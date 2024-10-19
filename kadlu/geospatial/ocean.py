""" The ocean module provides an interface to fetching, loading
    and interpolating ocean variables.
"""
import logging
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import kadlu
from kadlu.geospatial.interpolation import get_interpolator, GEOSPATIAL_DIMS
from kadlu.geospatial.data_sources.data_util import fmt_coords, fmt_time
from kadlu.geospatial.data_sources.source_map import load_map, precip_type_map
from kadlu.utils import center_point


# Ocean variable types 
kadlu_vartypes = np.unique([f.rsplit('_', 1)[0] for f in load_map.keys()])


def _drop_dims(data, drop):
    """ Helper function for dropping dimensions and averaging over degenerate data points """
    if drop is None:
        drop = []

    if isinstance(drop, str):
        drop = [drop]

    if len(drop) == 0:
        return data

    v = data.pop("value")
    n_dims = len(data)

    # case I: regular grids
    if np.ndim(v) == 0 or np.ndim(v) == n_dims:
        # remove coordinate arrays for dropped dimensions
        for dim in drop:
            if dim in data:
                del data[dim]

        # average data values over dropped dimensions
        axis = tuple([i for i,d in enumerate(GEOSPATIAL_DIMS) if d in drop and i < np.ndim(v)])
        data["value"] = np.average(v, axis=axis)

    # case II: irregular grids
    else:
        # remove coordinate arrays for dropped dimensions
        for dim in drop:
            if dim in data:
                del data[dim]

        # average data values over dropped dimensions
        # TODO: implement this step
        data["value"] = v

        logger = logging.getLogger("kadlu")
        logger.warning("Averaging over degenerate data points not implemented when dropping dimensions from irregular grids")

    return data


def _validate_loadvars(loadvars):
    """ Check validity of data loading arguments provided to the Ocean class at initialisation.
    
        Args:
            loadvars: dict
                Data loading arguments

        Returns:
            vartypes: list(str)
                Data types

        Raises:
            TypeError: if invalid argments are encountered
    """
    vartypes = []
    for key in loadvars.keys():
        vartype = key.lstrip("load_")
        if vartype not in kadlu_vartypes:
            err_msg = f'{key} is not a valid argument. Valid datasource args include: {", ".join([f"load_{v}" for v in kadlu_vartypes])}'
            raise TypeError(err_msg)
        
        vartypes.append(vartype)

    return vartypes


class Ocean():
    """ Class for retrieving ocean data variables.

        Data will be loaded using the given data sources and 
        geographic, depth, and temporal boundaries.

        It is also possible to write your own data loading function. 
        
        The boundary arguments supplied to the Ocean class will be passed 
        to the the data loading function, i.e., north, south, west, east, 
        top, bottom, start, end.

        TODO: 
            * [ ] Implement averaging across degenerate data points for irregular grids
            * [ ] Re-implement interpolation of precipitation type
            * [ ] Modify data loading classes so they return the data as a dict with keys lat,lon,epoch,depth instead of a numpy array.

        Args:
            north, south: float
                Latitude boundaries, in degrees
            west, east: float
                Longitude boundaries, in degrees
            top, bottom: float
                Depth range, in metres
            start, end: datetime
                UTC time range
            drop: list(str)
                Dimensions to be dropped. If dropping a dimension leads to degeneracy (multiple data 
                points with same coordinates) the average value is used. NOT YET IMPLEMENTED
            interp_args: dict
                Used for passing keyword arguments to the interpolator. See 
                `kadlu.geospatial.interpolation.get_interpolator` for allowed arguments.
            **loadvars:
                Keyword args supplied as 'load_{v}' where v is either an
                integer, float, array of shape [val, lat, lon[, epoch[, depth]]], 
                dict with keys value, lat, lon, epoch, depth, or a string source identifier 
                (e.g. `era5`) as described in the `source_map`

        Attrs:
            origin: tuple(float, float)
                Latitude and longitude coordinates of the centre point of the
                geographic bounding box. This point serves as the origin of the
                planar x-y coordinate system.
            boundaries: dict
                Bounding box for the ocean volume in space and time
            interpolators: dict
                Dictionary of data interpolators
    """
    def __init__(
        self,
        south=kadlu.defaults['south'],
        west=kadlu.defaults['west'],
        north=kadlu.defaults['north'],
        east=kadlu.defaults['east'],
        bottom=kadlu.defaults['bottom'],
        top=kadlu.defaults['top'],
        start=kadlu.defaults['start'],
        end=kadlu.defaults['end'],
        drop=None,
        interp_args=None,
        **loadvars,
    ):
        self.name = self.__class__.__name__

        self.logger = logging.getLogger("kadlu")

        default_value = 0

        if interp_args is None:
            interp_args = dict()
            
        # confirm validity of data loading args
        vartypes = _validate_loadvars(loadvars)

        # ocean spatio-temporal boundaries
        self.boundaries = dict(
            south=south,
            north=north,
            west=west,
            east=east,
            top=top,
            bottom=bottom,
            start=start,
            end=end
        )

        # log info
        info_msg = f"Initializing Ocean in region {fmt_coords(self.boundaries)}"\
                    + f" for time period {fmt_time(self.boundaries)}"\
                    + f" with variables: {vartypes}"
        self.logger.info(info_msg)

        # center point of XY coordinate system
        self.origin = center_point(lat=[south, north], lon=[west, east])

        # load data and initialize interpolators
        self.interpolators = dict()
        for vartype in kadlu_vartypes:

            # get the data loading argument for the given data type;
            # if no argument was provided, use 0 as the default data value
            load_arg = loadvars.get(f"load_{vartype}", default_value)

            # load the data
            if callable(load_arg):
                data = load_arg(**self.boundaries)

            elif isinstance(load_arg, str):
                key = f'{vartype}_{load_arg.lower()}'
                assert key in load_map.keys(), f"No entry found for {key} in Kadlu's load map:\n{load_map}"

                data = load_map[key](**self.boundaries)

            elif isinstance(load_arg, (int, float)):
                data = [load_arg]

            elif isinstance(load_arg, (list, tuple, np.ndarray, dict)):
                if len(load_arg) > 5:
                    err_msg = f'Invalid array shape for load_{vartype}. Arrays must be ordered by [val, lat, lon[, epoch[, depth]]].'
                    raise ValueError(err_msg)

                data = load_arg

            else:
                err_msg = f'Invalid type for load_{vartype}. Valid types include string, float, array, dict, and callable'
                raise TypeError(err_msg)

            # if the data are not already organized into a dict, place the arrays in a dict 
            # using the standard kadlu ordering: value,lat,lon,epoch,depth
            if not isinstance(data, dict):
                keys = ["value"] + GEOSPATIAL_DIMS
                data = {keys[i]: arr for i,arr in enumerate(data)}

            # info message
            v = data["value"]
            if not v is default_value:
                info_msg = f"Finished loading {vartype}"
                for k,arr in data.items():
                    s = arr.shape if np.ndim(arr) > 0 else "scalar"
                    info_msg += f"\n  {k.rjust(5)}: shape={s} min={np.min(arr):.3f} max={np.max(arr):.3f} avg={np.mean(arr):.3f}"

                self.logger.info(info_msg)

            # drop dimensions
            data = _drop_dims(data, drop)

            # pass data to interpolator
            self.interpolators[vartype] = get_interpolator(**data, name=vartype, origin=self.origin, **interp_args)

        """
        # TODO: review this, update as needed
        self.precip_src = loadvars['load_precip_type'] if 'load_precip_type' in loadvars.keys() else None
        """

        # create interpolation method for every variable type
        for vartype,interp in self.interpolators.items():
            setattr(self, vartype, interp)



'''
    def bathymetry_deriv(self, axis, **kwargs):
        """ Interpolates the bathymetric slope along either the south-north (latitude, y) or east-west (longitude, x) axis.

            Args:
                axis: str
                    Axis along which to compute the derivative. Can be `lat`, `lon`, `x`, or `y`

            Returns:
                : array-like
                    The slope values, in m/deg if axis is lat/lon, or dimensionless if axis is x/y
        """
        assert axis in ('lat', 'lon', 'x', 'y'), 'axis must be `lat`, `lon`, `x`, or `y`'

        dlat = (axis == "lat")
        dlon = (axis == "lon")
        dx   = (axis == "x")
        dy   = (axis == "y")

        return self.interps['bathymetry'](dlat=dlat, dlon=dlon, dx=dx, dy=dy, **kwargs)
'''

"""
    def precip_type(self, **kwargs):
        "TODO: provide documentation for this method"
        callback, varmap = precip_type_map[self.precip_src]
        v, y, x, t = callback(west=min(lon),
                              east=max(lon),
                              south=min(lat),
                              north=max(lat),
                              start=self.boundaries['start'],
                              end=self.boundaries['end'])
        return np.array([
            varmap[v]
            for v in NearestNDInterpolator((y, x, t), v)(kwargs["lat"], lon, epoch)
        ])
"""