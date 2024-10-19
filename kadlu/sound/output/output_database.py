import os
import netCDF4 as nc4
import numpy as np
import json


class Dataset(nc4.Dataset):
    """
    netCDF-4 database interface.
    It is used with the geophony function in the Kadlu library to save the modeled data and such data will be used by the Soundscape Atlas.
    This dataset inherits all characteristics of netCDF4 database class with some improvements. It comes with default metadata, saving user's time to define attributes,
    Users are free to change the defaults attributes or use their customerised settings as long as these settings follow the templates we provide.
    """

    def __init__(self, fname, op, format='NETCDF4', path=""):
        """ initialize a dataset instance with given filename, type of operation as well as path in which you want the file to be existed

            args:
                fname: string
                    string value for the datbase file name
                op: string
                    type of the operation (aka access mode in netCDF)
                    Available options ("r", "w", "r+", "a")
                    r means read-only, and no data can be modified.
                    w means write, a new file is created if this op is used and at the same time, the existing file with this name will be deleted
                    a and r+ mean append: an existing file is opened for reading and writing
                format: string
                    describe the netCDF file format. Available options are ('NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET', 'NETCDF4_64BIT_DATA').
                    The default value is 'NETCDF4', which uses the HDF5 to store the data and uses netCDF4 API features. More introduction are available here: https://unidata.github.io/netcdf4-python/netCDF4/index.html
                path: string
                    path of which the file to be stored
                    if the path is not given by the user, we will use the sys.path[0] to determine the position.
            return: nothing
        """
        assert fname != "", 'filename cannot be empty!'
        super(Dataset, self).__init__(fname, op, format)
        if op == 'w':
            if not path:
                path = os.path.dirname(os.path.realpath(__file__))
            attrs, dims, vars = json.load(open(os.path.join(path, 'attrs.json'))), \
                                json.load(open(os.path.join(path, 'dims.json'))), \
                                json.load(open(os.path.join(path, 'vars.json')))
            self._build_attrs(attrs)
            self._build_dims(dims)
            self._build_vars(vars)

    def _build_attrs(self, attrs):
        """ Add all predefined attributes into database
            args:
                attrs: dictionary
                    defines the attributes of the dataset. If not path is given when initializing the dataset instance, the attributes are populated with predefined values.
            return: nothing
        """
        self.setncatts(attrs)

    def _build_dims(self, dims):
        """ create dimensions from predefined dimension list
            args:
                dims: dictionary
                    defines the dimensions of the dataset. The key of dictionary is the dimension name, value is the size of dimension.
            return: nothing

        """
        for k, v in dims.items():
            self.createDimension(k, v)

    def _build_vars(self, vars):
        """ create variable from predefined variable list
            args:
                vars: dictionary
                    define the Variable of dataset. The key of dictionary is the variable name, the value of the dictionary defines the schema of that variable.

            return: nothing
        """
        for var_name, obj in vars.items():
            dtype, dimens, fill_value = "", "", None
            if isinstance(obj["dimensions"], str):
                # if dimension is a str, we do nothing
                dtype = obj['dtype']
                dimens = (obj["dimensions"])
            elif isinstance(obj["dimensions"], list):
                # if dimension is a list, we convert it to tuple
                dtype = obj['dtype']
                dimens = tuple(obj['dimensions'])
            elif obj['dimensions'] is None:
                # if dimension is none, the tuple is set empty
                dtype = obj['dtype']
                dimens = tuple()
            # otherwise we raise an error
            else:
                raise AssertionError(
                    "Dimension should be defined as either a single string or a list of Strings"
                )

            # check what the default value is
            fill_value = self.check_fill_value(obj)
            # create variable
            new_var = self.createVariable(var_name,
                                          dtype,
                                          dimensions=dimens,
                                          fill_value=fill_value)
            # add attributes describing the variable to the newly created variable
            new_var.setncatts(obj['property'])

    def check_fill_value(self, attrs):
        """ function to check the fillvalue (default value) of a Variable
            args:
                attrs: dictionary
                    one Variable attributes.
            return:
                fill_value: object
                    return np.nan if '_FillValue' is null, otherwise, return given "_FillValue"
        """
        fill_value = None
        if '_FillValue' in attrs["property"]:
            if attrs["property"]['_FillValue'] is None:
                fill_value = np.nan
            else:
                fill_value = attrs['property']['_FillValue']
            del attrs['property']['_FillValue']
        return fill_value

    def get_var(self, var):
        """ getter function to get variable from variable list
            args:
                var: string
                    the name of the Variable in the NetCDF dataset
            return:
                variable:
                    return None if given variable does not exist other return requested variable.

        """
        return self.variables[var] if var in self.variables else None

    def new_var(self, var, dtype, dimensions=(), attrs=dict()):
        """ create a new variable in the database with give parameters
            args:
                var: string
                    the variable name
                dtype: string
                    data type of the variable. the Support data types include
                    'S1' or 'c' => NC_CHAR,
                    'i1' or 'b' or 'B' => NC_BYTE,
                    'u1' => NC_UBYTE,
                    'i2' or 'h' or 's' => NC_SHORT,
                    'u2' => NC_USHORT,
                    'i4' or 'i' or 'l' =>  NC_INT,
                    'i8' => NC_INT64,
                    'u8' => NC_UINT64,
                    'f4' or 'f' => NC_FLOAT,
                    'f8' or 'd' => NC_DOUBLE.
                dimensions: tuple
                    dimensions must be tuple containing dimensions names and these dimensions have to be defined previously using 'createDimension'
                    The default value is an empty tuple, which means the variable is scalar.
                attrs: dictionary
                    attrs are the properties describing the Variable, for example, stanrdard_name, long_name, fillvalue and etc.
            return:
                new_var: netCDF Variable object
                    return netCDF Variable object if it is successfully created otherwise return existing variable
        """
        if var not in self.variables:
            new_var = self.createVariable(
                var,
                dtype,
                dimensions,
                fill_value=self.check_fill_value(attrs))
            new_var.setncatts(attrs['property'])
            return new_var
        else:
            return self.variables[var]

    def get_dim(self, d):
        """ return dimension variable by given dimension name
            args:
                d: string
                    The name of Dimension variable.
            return:
                dimension: Object
                    return None if the given dimension name is not existed otherwise return requested dimension variable
        """
        return self.dimensions[d] if d in self.dimensions else None

    def new_dim(self, d, size):
        """ create a new dimension variable
            args:
                d: string
                    dimension name.
            return:
                dimension: netCDF Dimension Object
                    return newly created netCDF Dimension object or existing dimension object
        """
        if d not in self.dimensions:
            return self.createDimension(d, size)
        else:
            return self.dimensions[d]

    def get_global_attr(self, name):
        """ return global attribute by name
            args:
                name: string
                    the name of the global attribute
            return:
                attribute: string
                    return the global attribute
        """
        return self.getncattr(name)

    def set_global_attr(self, name, value):
        """ add a new global attribute into database
            args:
                name: string
                    the name of the global attribute
                value: string
                    the description of the attribute
        """
        self.setncattr_string(name, value)
