""" Unit tests for the the 'geospatial.interpolation' module in the 'kadlu' package

    Authors: Oliver Kirsebom
    contact: oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/kadlu
             Project goal: Tools for underwater soundscape modeling

    License:

"""
import pytest
import os
import numpy as np
import kadlu
import kadlu.geospatial.interpolation as ki
from kadlu.utils import load_data_from_file, center_point, LLtoXY

path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

np.random.seed(1)

gitlab = os.environ.get("GITLAB_CI") is not None


def test_derivative():
    """ Check interpolation of derivates"""
    # with constant value
    v0 = 12.2
    interp = ki.IrregularGridGeospatialInterpolator(value=v0)
    dv = interp(lon=1.1, dlat=1)
    assert dv == 0

    # with irregular grid 
    v0 = [0, 10, 20]
    lat0 = [0, 1, 2]
    interp = ki.IrregularGridGeospatialInterpolator(value=v0, lat=lat0)
    with pytest.raises(NotImplementedError):
        interp(lat=1.5, dlat=2)

    # with regular grid 
    v0 = [0, 10, 20]
    lat0 = [0, 1, 2]
    interp = ki.RegularGridGeospatialInterpolator(value=v0, lat=lat0)
    with pytest.raises(NotImplementedError):
        interp(lat=1.5, dlat=2)

    # with regular grip in spherical geometry
    lat0 = np.array([44, 45, 46, 47, 48])
    lon0 = np.array([60, 61, 62, 63])
    epoch0 = 700
    f = lambda x: np.sin(x * np.pi/180.) #value ~sin(lon)
    df = lambda x: np.cos(x * np.pi/180.) * np.pi/180. #derivative ~cos(lon) in units of deg^-1
    v0 = [f(lon0) for _ in range(lat0.shape[0])]
    interp = ki.RegularGridGeospatialInterpolator(value=v0, lat=lat0, lon=lon0, epoch=epoch0)
    dv = interp(lat=[44.5, 46.0], lon=[60.5, 62.0], dlon=1)
    assert dv[0] == pytest.approx(df(60.5), rel=1E-2)
    assert dv[1] == pytest.approx(df(62.0), rel=1E-2)
    dv = interp(lat=[44.5, 46.0], lon=[60.5, 62.0], dlat=1)
    assert dv[0] == pytest.approx(0., abs=1E-6)
    assert dv[1] == pytest.approx(0., abs=1E-6)


def test_irreg_interp_scalar():
    """ Check that we can use the irregular grid interpolator with scalar input """
    v0 = 12.2
    interp = ki.IrregularGridGeospatialInterpolator(value=v0)

    assert interp.coordinates == {}

    # dimensionless interpolation
    v = interp()
    assert v == v0

    # scalar coordinate
    v = interp(lon=1.1)
    assert v == v0

    # multiple scalar coordinates
    v = interp(lon=1.1, epoch=77)
    assert v == v0

    # single coordinate array
    lat = [1, 2]
    v = interp(lat=lat)
    assert v.shape == (len(lat),)
    assert np.all(v == v0)

    # multiple coordinate arrays
    lat = [1, 2]
    lon = [10, 11]
    v = interp(lat=lat, lon=lon)
    assert v.shape == (len(lat),)
    assert np.all(v == v0)

    # if arrays have different sizes, an assertion error is thrown
    with pytest.raises(AssertionError):
        lat = [1, 2]
        lon = [10, 11, 12]
        interp(lat=lat, lon=lon)

    # grid 
    lat = 1
    epoch = [10, 11]
    v = interp(lat=lat, epoch=epoch, grid=True)
    assert v.shape == (1, 2)
    assert np.all(v == v0)

    # another grid 
    depth = [100, 200, 300]    
    epoch = [10, 11]
    v = interp(depth=depth, epoch=epoch, grid=True)
    assert v.shape == (2, 3)
    assert np.all(v == v0)

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0)
    assert v == v0

    # XY interpolation arrays
    v = interp(x=[0, 100], y=[0, -100])
    assert v.shape == (2,)
    assert np.all(v == v0)

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, -100, 200], epoch=333, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(v == v0)


def test_irreg_interp_single_array():
    """ Check that we can use the irregular grid interpolator with single array input """
    v0 = [0, 10, 20]
    lat0 = [0, 1, 2]
    interp = ki.IrregularGridGeospatialInterpolator(value=v0, lat=lat0)

    assert interp.method == "cubic"
    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.dims == ["lat"])

    # dimensionless interpolation is not allowed
    with pytest.raises(AssertionError):
        interp()

    # scalar coordinate
    v = interp(lat=1)
    assert v == v0[1]

    # method was downgraded from 'cubic' to 'linear'
    assert interp.method == "linear"

    # multiple scalar coordinates
    v = interp(lat=1, epoch=77)
    assert v == v0[1]

    # linear interpolation is ok
    assert interp.method == "linear"

    # array coordinate
    lat = [1, 2]
    v = interp(lat=lat)
    assert v.shape == (2,)
    assert np.all(v == [10, 20])

    # multiple array coordinates
    lat = [1, 2, 0]
    epoch = [700, 800, 900]
    v = interp(lat=lat, epoch=epoch)
    assert v.shape == (3,)
    assert np.all(v == [10, 20, 0])

    # grid
    lat = [1, 2]
    depth = [22, 42, 62]
    v = interp(lat=lat, depth=depth, grid=True)
    assert v.shape == (2, 3)
    assert np.all(v == [[10, 10, 10], [20, 20, 20]])

    # another grid
    lat = 1
    depth = [22, 42, 62]
    v = interp(lat=lat, depth=depth, grid=True)
    assert v.shape == (1,3)
    assert np.all(v == [[10, 10, 10]])

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0)
    assert v == v0[0]

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 100], y=[0, one_deg_m])
    assert v.shape == (2,)
    assert np.all(np.isclose(v, v0[:2], rtol=1E-3))

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], epoch=333, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(np.isclose(v[0, :, 0], v0, rtol=1E-3))


def test_irreg_interp_multiple_arrays():
    """ Check that we can use the irregular grid interpolator with multiple array input """
    lat0 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    depth0 = [0, 0, 0, 100, 100, 100, 200, 200, 200]
    v0 = [0,  1,  2, 10, 11, 12, 18, 19, 20]
    interp = ki.IrregularGridGeospatialInterpolator(value=v0, lat=lat0, depth=depth0)

    # default method is cubic
    assert interp.method == "cubic"

    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.coordinates["depth"] == depth0)
    assert np.all(interp.dims == ["lat", "depth"])

    # dimensionless interpolation is not allowed
    with pytest.raises(AssertionError):
        interp()

    # scalar coordinate
    v = interp(lat=1, depth=100)
    assert isinstance(v, float)
    assert v == pytest.approx(v0[4], rel=1E-6)

    # array coordinates
    v = interp(lat=[1, 2], depth=[100, 200])
    assert v.shape == (2,)
    assert v[0] == pytest.approx(v0[4], rel=1E-6)
    assert v[1] == pytest.approx(v0[8], rel=1E-6)

    # grid
    v = interp(lat=[1, 2], depth=[100, 200, 50], epoch=44, grid=True)
    assert v.shape == (2, 1, 3)
    assert v[0, 0, 0] == pytest.approx(v0[4], rel=1E-6)
    assert v[0, 0, 1] == pytest.approx(v0[7], rel=1E-6)
    assert v[1, 0, 0] == pytest.approx(v0[5], rel=1E-6)
    assert v[1, 0, 1] == pytest.approx(v0[8], rel=1E-6)

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0, depth=0)
    assert v == v0[0]

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 100], y=[0, one_deg_m], depth=[0, 100])
    assert v.shape == (2,)
    assert np.all(np.isclose(v, [v0[0], v0[4]], rtol=1E-3))

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], depth=100, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(np.isclose(v[0, :, 0], v0[3:6], rtol=1E-3))


def test_reg_interp_scalar():
    """ Check that we can use the regular grid interpolator with scalar input """
    v0 = 12.2
    interp = ki.RegularGridGeospatialInterpolator(value=v0)

    assert interp.coordinates == {}

    # dimensionless interpolation
    v = interp()
    assert v == v0

    # scalar coordinate
    v = interp(lon=1.1)
    assert v == v0

    # multiple scalar coordinates
    v = interp(lon=1.1, epoch=77)
    assert v == v0

    # single coordinate array
    lat = [1, 2]
    v = interp(lat=lat)
    assert v.shape == (len(lat),)
    assert np.all(v == v0)

    # multiple coordinate arrays
    lat = [1, 2]
    lon = [10, 11]
    v = interp(lat=lat, lon=lon)
    assert v.shape == (len(lat),)
    assert np.all(v == v0)

    # if arrays have different sizes, an assertion error is thrown
    with pytest.raises(AssertionError):
        lat = [1, 2]
        lon = [10, 11, 12]
        interp(lat=lat, lon=lon)

    # grid 
    lat = 1
    epoch = [10, 11]
    v = interp(lat=lat, epoch=epoch, grid=True)
    assert v.shape == (1, 2)
    assert np.all(v == v0)

    # another grid 
    depth = [100, 200, 300]    
    epoch = [10, 11]
    v = interp(depth=depth, epoch=epoch, grid=True)
    assert v.shape == (2, 3)
    assert np.all(v == v0)

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0)
    assert v == v0

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 100], y=[0, one_deg_m])
    assert v.shape == (2,)
    assert np.all(v == v0)

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], epoch=333, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(v == v0)


def test_reg_interp_single_array():
    """ Check that we can use the regular grid interpolator with single array input """
    v0 = [0, 10, 20]
    lat0 = [0, 1, 2]
    interp = ki.RegularGridGeospatialInterpolator(value=v0, lat=lat0)

    # default is linear
    assert interp.method == "linear"

    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.dims == ["lat"])

    # dimensionless interpolation is not allowed
    with pytest.raises(AssertionError):
        interp()

    # scalar coordinate
    v = interp(lat=1)
    assert v == v0[1]

    # multiple scalar coordinates
    v = interp(lat=1, epoch=77)
    assert v == v0[1]

    # array coordinate
    lat = [1, 2]
    v = interp(lat=lat)
    assert v.shape == (2,)
    assert np.all(v == [10, 20])

    # multiple array coordinates
    lat = [1, 2, 0]
    epoch = [700, 800, 900]
    v = interp(lat=lat, epoch=epoch)
    assert v.shape == (3,)
    assert np.all(v == [10, 20, 0])

    # grid
    lat = [1, 2]
    depth = [22, 42, 62]
    v = interp(lat=lat, depth=depth, grid=True)
    assert v.shape == (2, 3)
    assert np.all(v == [[10, 10, 10], [20, 20, 20]])

    # another grid
    lat = 1
    depth = [22, 42, 62]
    v = interp(lat=lat, depth=depth, grid=True)
    assert v.shape == (1,3)
    assert np.all(v == [[10, 10, 10]])

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0)
    assert v == v0[0]

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 100], y=[0, one_deg_m])
    assert v.shape == (2,)
    assert np.all(np.isclose(v, v0[:2], rtol=1E-3))

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], epoch=333, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(np.isclose(v[0, :, 0], v0, rtol=1E-3))


def test_reg_interp_multiple_arrays():
    """ Check that we can use the regular grid interpolator with multiple array input """
    lat0 = [0, 1, 2]
    depth0 = [0, 100, 200]
    v0 = [[ 0,  1,  2],
          [10, 11, 12],
          [18, 19, 20]]
    interp = ki.RegularGridGeospatialInterpolator(value=v0, lat=lat0, depth=depth0)

    # default method is linear
    assert interp.method == "linear"

    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.coordinates["depth"] == depth0)
    assert np.all(interp.dims == ["lat", "depth"])

    # dimensionless interpolation is not allowed
    with pytest.raises(AssertionError):
        interp()

    # scalar coordinate
    v = interp(lat=1, depth=100)
    assert isinstance(v, float)
    assert v == pytest.approx(v0[1][1], rel=1E-6)

    # array coordinates
    v = interp(lat=[1, 2], depth=[100, 200])
    assert v.shape == (2,)
    assert v[0] == pytest.approx(v0[1][1], rel=1E-6)
    assert v[1] == pytest.approx(v0[2][2], rel=1E-6)

    # grid
    v = interp(lat=[1, 2], depth=[100, 200, 50], epoch=44, grid=True)
    assert v.shape == (2, 1, 3)
    assert v[0, 0, 0] == pytest.approx(v0[1][1], rel=1E-6)
    assert v[0, 0, 1] == pytest.approx(v0[1][2], rel=1E-6)
    assert v[1, 0, 0] == pytest.approx(v0[2][1], rel=1E-6)
    assert v[1, 0, 1] == pytest.approx(v0[2][2], rel=1E-6)

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0, depth=0)
    assert v == v0[0][0]

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 100], y=[0, one_deg_m], depth=[0, 100])
    assert v.shape == (2,)
    assert np.all(np.isclose(v, [v0[0][0], v0[1][1]], rtol=1E-3))

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], depth=100, grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(np.isclose(v[0, :, 0], np.array(v0)[:,1], rtol=1E-3))


def test_reg_interp_lat_lon_arrays():
    """ Check that we can use the regular grid interpolator on lat-lon input """
    lat0 = np.array([44, 45, 46, 47, 48])
    lon0 = np.array([60, 61, 62, 63])
    epoch0 = 700
    v0 = np.random.rand(len(lat0), len(lon0))

    interp = ki.RegularGridGeospatialInterpolator(value=v0, lat=lat0, lon=lon0, epoch=epoch0)

    assert isinstance(interp._interp, ki._RectSphereBivariateSpline)
    assert interp.method == "spline"

    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.coordinates["lon"] == lon0)
    assert np.all(interp.dims == ["lat","lon"])

    # multiple array coordinates
    lat = [44, 45, 46]
    lon = [60, 61, 60]
    v = interp(lat=lat, lon=lon)
    assert v.shape == (3,)
    assert v[0] == pytest.approx(v0[0,0], rel=1E-6)
    assert v[1] == pytest.approx(v0[1,1], rel=1E-6)
    assert v[2] == pytest.approx(v0[2,0], rel=1E-6)

    # grid
    lat = [44, 45, 46]
    lon = [60, 61]
    v = interp(lat=lat, lon=lon, grid=True)
    assert v.shape == (3, 2)
    assert np.all(np.isclose(v, v0[:3, :2], rtol=1E-6))
    v = interp(lat=lat, lon=lon, depth=[321, 1321], grid=True)
    assert v.shape == (3, 2, 2)
    assert np.all(np.isclose(v[:, :, 0], v0[:3, :2], rtol=1E-6))

    # XY interpolation scalar coordinates
    v = interp(x=0, y=0, depth=0, origin=(45, 61))
    assert v == pytest.approx(v0[1][1], rel=1E-6)

    # XY interpolation arrays
    one_deg_m = 111195 #approximate distance corresponding to 1 degree change in latitude 
    v = interp(x=[0, 0], y=[0, one_deg_m], depth=[0, 100], origin=(46, 61))
    assert v.shape == (2,)
    assert np.all(np.isclose(v, [v0[2][1], v0[3][1]], rtol=1E-3))

    # XY interpolation grid
    v = interp(x=[0, 100], y=[0, one_deg_m, 2*one_deg_m], depth=100, origin=(46, 61), grid=True)
    assert v.shape == (2, 3, 1)
    assert np.all(np.isclose(v[0, :, 0], np.array(v0)[2:,1], rtol=1E-3))


def test_create_regular_grid():
    """ Check that the _create_regular_grid helper function returns grids of expected shape and range
    """
    # coordinates provide 50% coverage of rectangle with corners (0,10), (2,10), (2,210), (0,210)
    lat = [1, 2, 1, 0, 1]
    depth = [10, 110, 210, 110, 110]

    # regular grid with specified shape
    g, c = ki._create_regular_grid(
        lat=lat, depth=depth, grid_shape={"lat":11, "depth":11}, return_coverage=True
    )
    assert np.all(g["lat"] == np.linspace(0, 2, 11))
    assert np.all(g["depth"] == np.linspace(10, 210, 11))
    assert c == 0.5

    # regular grid with specified bin size
    g, c = ki._create_regular_grid(
        lat=lat, depth=depth, bin_size={"lat":0.5, "depth":40}, return_coverage=True
    )
    assert np.all(g["lat"] == np.linspace(0, 2, 5))
    assert np.all(g["depth"] == np.linspace(10, 210, 6))
    assert c == 0.5

    # regular grid with bins automatically inferred
    g, c = ki._create_regular_grid(
        lat=lat, depth=depth, return_coverage=True
    )
    assert np.all(g["lat"] == np.linspace(0, 2, 3))
    assert np.all(g["depth"] == np.linspace(10, 210, 3))
    assert c == 0.5


def test_get_interpolator():
    """ Check that the get_interpolator convenience method returns interpolators as it should"""
    # data on a regular grid
    lat0 = np.array([44, 45, 46, 47, 48])
    lon0 = np.array([60, 61, 62, 63])
    epoch0 = 700
    v0 = np.random.rand(len(lat0), len(lon0))
    interp = ki.get_interpolator(value=v0, lat=lat0, lon=lon0, epoch=epoch0)
    assert isinstance(interp, ki.RegularGridGeospatialInterpolator)
    assert interp.method == "spline"
    assert np.all(interp.coordinates["lat"] == lat0)
    assert np.all(interp.coordinates["lon"] == lon0)
    assert np.all(interp.dims == ["lat","lon"])

    # data on a irregular grid
    lat0 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    depth0 = [0, 0, 0, 100, 100, 100, 200, 200, 200]
    v0 = [0,  1,  2, 10, 11, 12, 18, 19, 20]
    interp = ki.get_interpolator(value=v0, lat=lat0, depth=depth0)
    assert isinstance(interp, ki.RegularGridGeospatialInterpolator)
    assert interp.method == "linear" #default
    assert np.all(interp.coordinates["lat"] == np.unique(lat0))
    assert np.all(interp.coordinates["depth"] == np.unique(depth0))
    assert np.all(interp.dims == ["lat", "depth"])


"""
    =====================================================
    |                                                   |
    |   Legacy (kadlu < v2.4.0) tests below this point  |
    |                                                   | 
    =====================================================
"""

def test_irregular_interp_2d():
    """Check that IrregularGridGeospatialInterpolator behaves as expected for (very simple) 2D data"""
    # check that we can interpolate a single data point
    lat = [1.]
    lon = [2.]
    v = [3.]
    itp = ki.IrregularGridGeospatialInterpolator(value=v,lat=lat,lon=lon,method="linear")
    assert itp.method == None
    # specifying x,y as lists, we get an array back
    w = itp(lat=lat, lon=lon)
    assert w.shape == (1,)
    assert w == v
    # returned array has expected shape when grid = True
    w = itp(lat=[1,3], lon=[2,4,5], grid=True)
    assert w.shape == (2,3)
    assert np.all(w == 3.)
    # check that we can interpolate two data points
    lat = [1., 2.]
    lon = [2., 4.]
    v = [3., 6.]
    itp = ki.IrregularGridGeospatialInterpolator(value=v,lat=lat,lon=lon,method="linear")
    assert itp.method == "linear"
    w = itp(lat=lat[0], lon=lon[0])
    # check that method was downgraded to 'nearest' after first call
    assert itp.method == "nearest"
    assert w == v[0]
    w = itp(lat=[1,3], lon=[2,3,5], grid=True)
    assert w.shape == (2,3)
    assert w[0,0] == 3. #1,2 -> 1,2
    assert w[0,1] == 3. #1,3 -> 1,2
    assert w[0,2] == 6. #1,5 -> 2,4 
    assert w[1,0] == 3. or w[1,0] == 6. #3,2 -> 1,2 or 2,4
    assert w[1,1] == 6. #3,3 -> 2,4
    assert w[1,2] == 6. #3,5 -> 2,4
    # check that NaN values get replaced with nearest valid value
    lat = [1., 2., 0., -3.]
    lon = [2., 4., -2., 2.]
    v = [3., 6., 12., 9.]
    itp = ki.IrregularGridGeospatialInterpolator(value=v,lat=lat,lon=lon,method="linear")
    assert itp.method == "linear"
    # interpolate outside the convex hull
    w = itp(lat=2, lon=4.1, replace_nan=False)
    # check that method is unchanged by call
    assert itp.method == "linear"
    assert np.isnan(w)
    w = itp(lat=2, lon=4.1)
    assert w == 6.


def test_irregular_interp_3d():
    """Check that IrregularGridGeospatialInterpolator behaves as expected for 3D data"""
    # create fake data
    lat = np.linspace(0., 3., 4) #[0,1,2,3]
    lon = np.linspace(0., 4., 5) #[0,1,2,3,4]
    depth = np.linspace(0., 100., 3) #[0,50,100]
    lat, lon, depth = np.meshgrid(lat, lon, depth, indexing="ij")
    lat = lat.flatten()
    lon = lon.flatten()
    depth = depth.flatten()
    temp = np.ones((4, 5, 3))
    for i in range(3):
        temp[:, :, i] *= i  # temperature increases with depth: 0,1,2
    for i in range(5):
        temp[:, i, :] *= i  # temperature increases with longitude 0,0,0,0,0; 0,1,2,3,4; 0,2,4,6,8
    
    temp = temp.flatten()
    
    # initialize interpolator
    ip = ki.IrregularGridGeospatialInterpolator(value=temp, lat=lat, lon=lon, depth=depth, method="linear")

    # --- 4 latitudes ---
    lats = [1.2, 2.2, 0.2, 1.55]
    # --- 4 longitudes ---
    lons = [0.0, 1.0, 2.0, 4.0]
    # --- 4 depths ---
    depths = [0, 25, 50, 150]

    # interpolate all at once
    temps = ip(lat=lats, lon=lons, depth=depths)
    assert temps[0] == pytest.approx(0.0, abs=0.01)
    assert temps[1] == pytest.approx(0.5, abs=0.01)
    assert temps[2] == pytest.approx(2.0, abs=0.01)
    assert np.all(np.logical_and(temps >= -5, temps <= 105))

    # interpolate one at a time
    ti = list()
    for lat, lon, depth in zip(lats, lons, depths):
        ti.append(ip(lat=lat, lon=lon, depth=depth))

    # check that the all-at-once and one-at-a-time
    # approaches give the same result
    for tii, t in zip(ti, temps):
        assert tii == pytest.approx(t, rel=1e-3)


def test_interp_2x2_grid():
    values = np.array([[0, 2], [0, 2]])
    lats = np.array([0, 1])
    lons = np.array([0, 1])
    ip = ki.get_interpolator(value=values, lat=lats, lon=lons, method="linear")
    assert ip(lat=0, lon=0) == 0
    assert ip(lat=1, lon=1) == 2
    res = ip(lat=0, lon=0.5)
    assert np.abs(res - 1.) < 1e-6
    res = ip(lat=[0.5, 0.5], lon=[0.2, 0.3])
    assert np.all(np.abs(res - [0.4, 0.6]) < 1e-6)
    res = ip(lat=[0.5, 0.6, 0.7], lon=[0.2, 0.3], grid=True)
    assert res.shape == (3, 2)
    assert np.all(np.abs(res - [0.4, 0.6]) < 1e-6)


def test_interp_1x1_grid():
    values = np.array([[17]])
    lats = np.array([0])
    lons = np.array([0])
    ip = ki.get_interpolator(value=values, lat=lats, lon=lons)
    assert ip(lat=0, lon=0) == 17
    assert ip(lat=1, lon=1) == 17
    v = ip(lat=[1,15,-1], lon=[1,200], grid=True)
    assert v.shape == (3, 2)
    assert np.all(v == 17)


def test_interp_1x2_grid():
    values = np.array([[0, 2]])
    lats = np.array([0])
    lons = np.array([0, 1])
    ip = ki.get_interpolator(value=values, lat=lats, lon=lons)
    assert ip(lat=0, lon=0) == 0
    assert ip(lat=1, lon=1) == 2
    res = ip(lat=0, lon=0.5)
    assert np.abs(res - 1.) < 1e-6


def test_interpolate_bathymetry_using_latlon_coordinates():

    # load bathy data
    path = path_to_assets + '/bornholm.mat'
    bathy, lat, lon = load_data_from_file(path)

    # initialize interpolator
    ip = ki.get_interpolator(value=bathy, lat=lat, lon=lon)

    # interpolate at a single point on the grid
    z = ip(lat=lat[0], lon=lon[0])
    z = int(z)
    assert z == bathy[0, 0]

    # interpolate at a single point between two grid points
    x = (lat[1] + lat[2]) / 2
    z = ip(lat=x, lon=lon[0])
    z = float(z)
    zmin = min(bathy[1, 0], bathy[2, 0])
    zmax = max(bathy[1, 0], bathy[2, 0])
    assert z >= zmin
    assert z <= zmax

    # interpolate at two points
    x1 = (lat[1] + lat[2]) / 2
    x2 = (lat[2] + lat[3]) / 2
    z = ip(lat=[x1, x2], lon=lon[0], grid=True)
    zmin = min(bathy[1, 0], bathy[2, 0])
    zmax = max(bathy[1, 0], bathy[2, 0])
    assert z[0] >= zmin
    assert z[0] <= zmax
    zmin = min(bathy[2, 0], bathy[3, 0])
    zmax = max(bathy[2, 0], bathy[3, 0])
    assert z[1] >= zmin
    assert z[1] <= zmax

    # interpolate with grid = True/False
    x1 = (lat[1] + lat[2]) / 2
    x2 = (lat[2] + lat[3]) / 2
    y1 = (lon[1] + lon[2]) / 2
    y2 = (lon[2] + lon[3]) / 2
    z = ip(lat=[x1, x2], lon=[y1, y2], grid=False)
    assert np.ndim(z) == 1
    assert z.shape[0] == 2
    z = ip(lat=[x1, x2], lon=[y1, y2], grid=True)
    assert np.ndim(z) == 2
    assert z.shape[0] == 2
    assert z.shape[1] == 2


def test_interpolation_grids_are_what_they_should_be():
    # load data and initialize interpolator
    path = path_to_assets + '/bornholm.mat'
    _, lat, lon = load_data_from_file(path)
    lat_c = 0.5 * (lat[0] + lat[-1])
    lon_c = 0.5 * (lon[0] + lon[-1])
    assert lat_c, lon_c == center_point(lat, lon)


def test_interpolation_tables_agree_on_latlon_grid():
    # load data and initialize interpolator
    path = path_to_assets + '/bornholm.mat'
    bathy, lat, lon = load_data_from_file(path)
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)

    # lat fixed
    ilat = int(len(ip.coordinates["lat"]) / 2)
    lat = ip.coordinates["lat"][ilat]
    for lon in ip.coordinates["lon"]:
        bll = ip(lat=lat, lon=lon)
        x, y = LLtoXY(
            lat=lat,
            lon=lon,
            lat_ref=ip.origin[0],
            lon_ref=ip.origin[1]
        )
        bxy = ip(x=x, y=y)
        assert bxy == pytest.approx(bll, rel=1e-3) or bxy == pytest.approx(bll, abs=0.1)

    # lon fixed
    ilon = int(len(ip.coordinates["lon"]) / 2)
    lon = ip.coordinates["lon"][ilon]
    for lat in ip.coordinates["lat"]:
        bll = ip(lat=lat, lon=lon)
        x, y = LLtoXY(
            lat=lat,
            lon=lon,
            lat_ref=ip.origin[0],
            lon_ref=ip.origin[1]
        )
        bxy = ip(x=x, y=y)
        assert bxy == pytest.approx(bll, rel=1e-3) or bxy == pytest.approx(bll, abs=0.1)


def test_interpolation_tables_agree_anywhere():
    # load data and initialize interpolator
    path = path_to_assets + '/bornholm.mat'
    bathy, lat, lon = load_data_from_file(path)
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)

    # --- at origo ---
    lat_c, lon_c = center_point(lat, lon)
    #lat_c = ip.origin.latitude
    #lon_c = ip.origin.longitude
    z_ll = ip(lat=lat_c, lon=lon_c)  # interpolate using lat-lon
    z_ll = float(z_ll)
    z_xy = ip(x=0, y=0)  # interpolate using x-y
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1e-3) or z_xy == pytest.approx(z_ll, abs=0.1)

    # --- 0.1 degrees north of origo ---
    lat = lat_c + 0.1
    lon = lon_c
    x, y = LLtoXY(lat=lat, lon=lon, lat_ref=lat_c, lon_ref=lon_c)
    z_ll = ip(lat=lat, lon=lon)
    z_ll = float(z_ll)
    z_xy = ip(x=x, y=y)
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1e-3) or z_xy == pytest.approx(z_ll, abs=0.1)

    # --- 0.08 degrees south of origo ---
    lat = lat_c - 0.08
    lon = lon_c
    x, y = LLtoXY(lat=lat, lon=lon, lat_ref=lat_c, lon_ref=lon_c)
    z_ll = ip(lat=lat, lon=lon)
    z_ll = float(z_ll)
    z_xy = ip(x=x, y=y)
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1e-3) or z_xy == pytest.approx(z_ll, abs=0.1)

    # --- at shifted origo ---
    bathy, lat, lon = load_data_from_file(path)
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon, origin=(55.30, 15.10))
    lat_c = ip.origin[0]
    lon_c = ip.origin[1]
    z_ll = ip(lat=lat_c, lon=lon_c)  # interpolate using lat-lon
    z_ll = float(z_ll)
    z_xy = ip(x=0, y=0)  # interpolate using x-y
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1e-3) or z_xy == pytest.approx(z_ll, abs=0.1)


def test_interpolation_tables_agree_on_ll_grid_for_dbarclays_data():
    # load data and initialize interpolator
    path = path_to_assets + '/BathyData_Mariana_500kmx500km.mat'
    bathy, lat, lon = load_data_from_file(
        path,
        lat_name='latgrat',
        lon_name='longrat',
        val_name='mat',
        lon_axis=0
    )
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)

    # lat fixed
    ilat = int(len(ip.coordinates["lat"]) / 2)
    lat = ip.coordinates["lat"][ilat]
    for lon in ip.coordinates["lon"]:
        bll = ip(lat=lat, lon=lon)
        x, y = LLtoXY(
            lat=lat,
            lon=lon,
            lat_ref=ip.origin[0],
            lon_ref=ip.origin[1]
        )
        bxy = ip(x=x, y=y)
        assert bxy == pytest.approx(bll, rel=1e-3) or bxy == pytest.approx(bll, abs=0.1)

    # lon fixed
    ilon = int(len(ip.coordinates["lon"]) / 2)
    lon = ip.coordinates["lon"][ilon]
    for lat in ip.coordinates["lat"]:
        bll = ip(lat=lat, lon=lon)
        x, y = LLtoXY(
            lat=lat,
            lon=lon,
            lat_ref=ip.origin[0],
            lon_ref=ip.origin[1]
        )
        bxy = ip(x=x, y=y)
        assert bxy == pytest.approx(bll, rel=1e-3) or bxy == pytest.approx(bll, abs=0.1)


def test_interpolation_tables_agree_anywhere_for_dbarclays_data():
    # load data and initialize interpolator
    path = path_to_assets + '/BathyData_Mariana_500kmx500km.mat'
    bathy, lat, lon = load_data_from_file(
        path,
        lat_name='latgrat',
        lon_name='longrat',
        val_name='mat',
        lon_axis=0
    )
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)

    # --- at origo ---
    lat_c = ip.origin[0]
    lon_c = ip.origin[1]
    z_ll = ip(lat=lat_c, lon=lon_c)  # interpolate using lat-lon
    z_ll = float(z_ll)
    z_xy = ip(x=0, y=0)  # interpolate using x-y
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1E-3) or z_ll == pytest.approx(z_xy, abs=0.1)

    # --- at shifted origo ---
    bathy, lat, lon = load_data_from_file(
        path,
        lat_name='latgrat',
        lon_name='longrat',
        val_name='mat',
        lon_axis=0
    )
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon, origin=(9., 140.))
    lat_c = ip.origin[0]
    lon_c = ip.origin[1]
    z_ll = ip(lat=lat_c, lon=lon_c)  # interpolate using lat-lon
    z_ll = float(z_ll)
    z_xy = ip(x=0, y=0)  # interpolate using x-y
    z_xy = float(z_xy)
    assert z_ll == pytest.approx(z_xy, rel=1E-3) or z_ll == pytest.approx(z_xy, abs=0.1)


def test_mariana_trench_is_in_correct_location():
    # load data and initialize interpolator
    path = path_to_assets + '/BathyData_Mariana_500kmx500km.mat'
    bathy, lat, lon = load_data_from_file(
        path,
        lat_name='latgrat',
        lon_name='longrat',
        val_name='mat',
        lon_axis=0
    )
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)
    d = ip(lat=11.3733, lon=142.5917)
    assert d < -10770
    d = ip(lat=12.0, lon=142.4)
    assert d > -3000
    d = ip(lat=11.4, lon=143.1)
    assert d < -9000


def test_can_interpolate_multiple_points_in_ll():
    # load data and initialize interpolator
    path = path_to_assets + '/bornholm.mat'
    bathy, lat, lon = load_data_from_file(path)
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)
    # coordinates of origin
    lat_c = ip.origin[0]
    lon_c = ip.origin[1]
    # --- 4 latitudes ---
    lats = [lat_c, lat_c + 0.1, lat_c - 0.2, lat_c + 0.03]
    # --- 4 longitudes ---
    lons = [lon_c, lon_c + 0.15, lon_c - 0.08, lon_c - 0.12]
    # interpolate all at once
    values_all = ip(lat=lats, lon=lons)
    # interpolate one at a time
    values_one = list()
    for lat, lon in zip(lats, lons):
        values_one.append(ip(lat=lat, lon=lon))
    # check that the values agree
    for a,b in zip(values_one, values_all):
        assert a == pytest.approx(b, rel=1e-3)
    # interpolate on a grid
    values_grid = ip(lat=lats, lon=lons, grid=True)
    # check that diagonal values agree
    for i,v in enumerate(values_all):
        assert values_grid[i,i] == pytest.approx(v, rel=1e-3)


def test_can_interpolate_multiple_points_in_xy():
    # load data and initialize interpolator
    path = path_to_assets + '/bornholm.mat'
    bathy, lat, lon = load_data_from_file(path)
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)
    # --- 4 x coordinates ---
    xs = [0, 1000, -2000, 300]
    # --- 4 y coordinates ---
    ys = [0, 1500, 800, -120]
    # interpolate all at once
    values_all = ip(x=xs, y=ys)
    # interpolate one at a time
    values_one = list()
    for x, y in zip(xs, ys):
        values_one.append(ip(x=x, y=y))
    # check that the values agree
    for a,b in zip(values_one, values_all):
        assert a == pytest.approx(b, rel=1e-3)
    # interpolate on a grid
    values_grid = ip(x=xs, y=ys, grid=True)
    # check that diagonal values agree
    for i,v in enumerate(values_all):
        assert values_grid[i,i] == pytest.approx(v, rel=1e-3)


def test_can_interpolate_regular_grid():
    # create fake data
    lat = np.array([44, 45, 46, 47, 48])
    lon = np.array([60, 61, 62, 63])
    bathy = np.random.rand(len(lat), len(lon))
    # initialize interpolator
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)
    # check value at grid point
    b = ip(lat=45, lon=62)
    assert b == pytest.approx(bathy[1, 2], abs=1E-9)


def test_can_interpolate_irregular_grid():
    # create fake data
    lat = np.array([0.0, 1.0, 1.5, 2.1, 3.0])
    lon = np.array([0.0, 2.0, 0.2, 0.7, 1.2])
    bathy = np.array([-90.0, -200.0, -140.0, -44.0, -301.0])
    # initialize interpolator
    ip = ki.get_interpolator(bathy, lat=lat, lon=lon)
    # --- 4 latitudes ---
    lats = [0.01, 1.0, 0.5, 2.1]
    # --- 4 longitudes ---
    lons = [0.01, 2.0, 1.0, 0.71]
    # interpolate all at once
    depths = ip(lat=lats, lon=lons)
    assert depths[1] == pytest.approx(-200, abs=0.1)
    assert depths[2] < -90 and depths[2] > -200
    # interpolate one at a time
    zi = list()
    for lat, lon in zip(lats, lons):
        zi.append(ip(lat=lat, lon=lon))
    # check that the all-at-once and one-at-a-time
    # approaches give the same result
    for z, d in zip(zi, depths):
        assert z == pytest.approx(d, rel=1e-3)


def test_can_interpolate_irregular_3d_grid():
    # create fake data
    lat = np.linspace(0., 3., 4)
    lon = np.linspace(0., 4., 5)
    depth = np.linspace(0., 100., 3)
    lat, lon, depth = np.meshgrid(lat, lon, depth, indexing="ij")
    lat = lat.flatten()
    lon = lon.flatten()
    depth = depth.flatten()
    temp = np.ones((4, 5, 3))
    for i in range(3):
        temp[:, :, i] *= i  # temperature increases with depth
    for i in range(5):
        temp[:, i, :] *= i  # temperature increases with longitude
    
    temp = temp.flatten()
    
    # initialize interpolator
    ip = ki.get_interpolator(temp, lat=lat, lon=lon, depth=depth)

    # --- 4 latitudes ---
    lats = [1.2, 2.2, 0.2, 1.55]
    # --- 4 longitudes ---
    lons = [0.0, 1.0, 2.0, 4.0]
    # --- 4 depths ---
    depths = [0, 25, 50, 150]
    # interpolate all at once
    temps = ip(lat=lats, lon=lons, depth=depths)
    assert temps[0] == pytest.approx(0.0, abs=0.01)
    assert temps[1] == pytest.approx(0.5, abs=0.01)
    assert temps[2] == pytest.approx(2.0, abs=0.01)
    assert np.all(np.logical_and(temps >= -5, temps <= 105))
    # interpolate one at a time
    ti = list()
    for lat, lon, depth in zip(lats, lons, depths):
        ti.append(ip(lat=lat, lon=lon, depth=depth))
    # check that the all-at-once and one-at-a-time
    # approaches give the same result
    for tii, t in zip(ti, temps):
        assert tii == pytest.approx(t, rel=1e-3)


def test_can_interpolate_irregular_3d_grid_where_one_dimension_only_has_1_point():
    # create fake data
    lat = np.linspace(0., 3., 4)
    lon = np.linspace(0., 4., 5)
    depth = np.linspace(0., 100., 1)
    lat, lon, depth = np.meshgrid(lat, lon, depth, indexing="ij")
    lat = lat.flatten()
    lon = lon.flatten()
    depth = depth.flatten()
    temp = np.ones((4, 5, 1))
    for i in range(5):
        temp[:, i, :] *= i  # temperature increases with longitude
    
    temp = temp.flatten()
    
    # initialize interpolator
    ip = ki.get_interpolator(temp, lat=lat, lon=lon, depth=depth)

    # --- 4 latitudes ---
    lats = [1.2, 2.2, 0.2, 1.55]
    # --- 4 longitudes ---
    lons = [0.0, 1.0, 2.0, 4.0]
    # --- 4 depths ---
    depths = [0, 25, 50, 150]
    # interpolate all at once
    temps = ip(lat=lats, lon=lons, depth=depths)
    assert temps[0] == pytest.approx(0.0, abs=0.01)
    assert temps[1] == pytest.approx(1.0, abs=0.01)
    assert temps[2] == pytest.approx(2.0, abs=0.01)
    assert np.all(np.logical_and(temps >= -5, temps <= 105))
    # interpolate one at a time
    ti = list()
    for lat, lon, depth in zip(lats, lons, depths):
        ti.append(ip(lat=lat, lon=lon, depth=depth))
    # check that the all-at-once and one-at-a-time
    # approaches give the same result
    for tii, t in zip(ti, temps):
        assert tii == pytest.approx(t, rel=1e-3)


def test_can_interpolate_gebco_data():
    # load data and initialize interpolator
    south, west = 43, -60
    north, east = 44, -59
    bathy, lat, lon = kadlu.load(
        var='bathymetry',
        source='gebco',
        south=south,
        north=north,
        west=west,
        east=east
    )

    ip = ki.get_interpolator(bathy, lat=lat, lon=lon, method='nearest')
    # --- 4 latitudes ---
    lats = [43.3, 43.2, 43.7, 43.5]
    # --- 4 longitudes ---
    lons = [-59.6, -59.8, -59.2, -59.3]
    # interpolate
    depths = ip(lat=lats, lon=lons)
    zi = list()
    for lat, lon in zip(lats, lons):
        zi.append(ip(lat=lat, lon=lon))
    for z, d in zip(zi, depths):
        assert z == pytest.approx(d, rel=1e-3)
    # interpolate on grid
    depths_grid = ip(lat=lats, lon=lons, grid=True)
    assert depths_grid.shape[0] == 4
    assert depths_grid.shape[1] == 4
    for i in range(4):
        assert depths_grid[i, i] == depths[i]


def test_interpolate_uniform_3d_data():
    N = 10
    np.random.seed(1)
    # create fake data
    val = np.ones(shape=(N, N, N)) 
    lat = np.arange(N)
    lon = np.arange(N)
    depth = np.arange(N)
    # initialize interpolator
    ip = ki.get_interpolator(val, lat=lat, lon=lon, depth=depth)
    # check interpolation at a few random points
    lats = np.random.rand(3) * (N - 1)
    lons = np.random.rand(3) * (N - 1)
    depths = np.random.rand(3) * (N - 1)
    vi = ip(lat=lats, lon=lons, depth=depths)
    for v in vi:
        assert v == pytest.approx(1, abs=1E-9)
    # check interpolation on a grid
    lats = np.random.rand(3) * (N - 1)
    lons = np.random.rand(4) * (N - 1)
    depths = np.random.rand(5) * (N - 1)
    vi = ip(lat=lats, lon=lons, depth=depths, grid=True)
    assert vi.shape[0] == 3
    assert vi.shape[1] == 4
    assert vi.shape[2] == 5
    assert np.all(np.abs(vi - 1.0) < 1E-9)


def test_interpolate_3d_data_with_constant_slope():
    N = 10
    np.random.seed(1)
    # create fake data
    val = np.ones(shape=(N, N, N))
    for k in range(N):
        val[:, :, k] = k * val[:, :, k]
    lat = np.arange(N)
    lon = np.arange(N)
    depth = np.arange(N)
    # initialize interpolator
    ip = ki.get_interpolator(val, lat=lat, lon=lon, depth=depth)
    # check interpolation
    lats = np.array([4, 4, 4])
    lons = np.array([4, 4, 4])
    depths = np.array([4, 4.5, 5])
    vi = ip(lat=lats, lon=lons, depth=depths)
    assert vi[0] == pytest.approx(4, abs=1E-9)
    assert vi[1] == pytest.approx(4.5, abs=1E-9)
    assert vi[2] == pytest.approx(5, abs=1E-9)
    # check interpolation on grid
    lats = np.array([2, 3, 4])
    lons = np.array([1, 2, 3])
    depths = np.array([4, 4.5, 5])
    vi = ip(lat=lats, lon=lons, depth=depths, grid=True)
    assert np.all(np.abs(vi[:, :, 0] - 4) < 1E-9)
    assert np.all(np.abs(vi[:, :, 1] - 4.5) < 1E-9)
    assert np.all(np.abs(vi[:, :, 2] - 5) < 1E-9)


def test_interpolate_3d_data_using_xy_coordinates():
    N = 11
    np.random.seed(1)
    # create fake data
    val = np.ones(shape=(N, N, N))
    for k in range(N):
        val[:, :, k] = k * val[:, :, k]
    lat = np.arange(N)
    lon = np.arange(N)
    depth = np.arange(N)
    # initialize interpolator
    ip = ki.get_interpolator(val, lat=lat, lon=lon, depth=depth)
    # check interpolation
    x = np.array([0, 100, 200])
    y = np.array([0, 100, 200])
    depths = np.array([4, 4.5, 5])
    vi = ip(x=x, y=y, depth=depths)
    assert vi[0] == pytest.approx(4, abs=1E-9)
    assert vi[1] == pytest.approx(4.5, abs=1E-9)
    assert vi[2] == pytest.approx(5, abs=1E-9)


def test_interpolate_3d_outside_grid():
    N = 10
    np.random.seed(1)
    # create fake data
    val = np.ones(shape=(N, N, N))
    lat = np.arange(N)
    lon = np.arange(N)
    depth = np.arange(N)
    # initialize interpolator
    ip = ki.get_interpolator(val, lat=lat, lon=lon, depth=depth)
    # check interpolation outside grid
    lats = 20
    lons = 5
    depths = 5
    vi = ip(lat=lats, lon=lons, depth=depths)
    assert vi == pytest.approx(1, abs=1E-3)


def test_interpolate_uniform_2d():
    ip = ki.get_interpolator(17)
    v = ip(lat=5, lon=2000)
    assert v == 17
    v = ip(lat=[5, 12, 13], lon=[2000, 0, 1])
    assert np.all(v == 17)
    assert v.shape[0] == 3
    v = ip(lat=[5, 12, 13], lon=[2000, 0], grid=True)
    assert np.all(v == 17)
    assert v.shape[0] == 3
    assert v.shape[1] == 2


def test_interpolate_uniform_3d():
    ip = ki.get_interpolator(17)
    v = ip(lat=5, lon=2000, depth=-10)
    assert v == 17
    v = ip(lat=[5, 12, 13], lon=[2000, 0, 1], depth=[0, 2, -3])
    assert np.all(v == 17)
    assert v.shape[0] == 3
    v = ip(lat=[5, 12, 13], lon=[2000, 0], depth=[0, 2, -3], grid=True)
    assert np.all(v == 17)
    assert v.shape[0] == 3
    assert v.shape[1] == 2
    assert v.shape[2] == 3


def test_interpolate_depth_3d():
    ip = ki.get_interpolator(
        value=[0, 1, 4, 9],
        depth=[0, 1, 2, 3],
        method='cubic'
    )
    # inside range
    v = ip(lat=5, lon=2000, depth=1.5)
    assert v == pytest.approx(1.5 * 1.5, rel=1E-4)
    # outside range
    v = ip(lat=5, lon=2000, depth=3.5)
    assert v == pytest.approx(3.5 * 3.5, rel=1E-4)
