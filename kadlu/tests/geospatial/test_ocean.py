import os
import logging
import pytest
import numpy as np
from datetime import datetime

from kadlu.geospatial.ocean import Ocean

gitlab = os.environ.get("GITLAB_CI") is not None

bounds = dict(
    start=datetime(2019, 1, 9),
    end=datetime(2019, 1, 9, 3),
    south=42,
    west=-62.5,
    north=44,
    east=-60.5,
    top=0,
    bottom=5000
)

test_lat, test_lon, test_depth = bounds['south'], bounds['west'], bounds['top']


def test_null_ocean():
    """ Test that ocean is initialized with all variables set to null (0)"""
    o = Ocean(load_bathymetry=0, **bounds)
    assert o.bathymetry(lat=test_lat, lon=test_lon) == 0
    assert o.temperature(lat=test_lat, lon=test_lon, depth=test_depth) == 0
    assert o.salinity(lat=test_lat, lon=test_lon, depth=test_depth) == 0
    assert o.wavedir(lat=test_lat, lon=test_lon) == 0
    assert o.waveheight(lat=test_lat, lon=test_lon) == 0
    assert o.waveperiod(lat=test_lat, lon=test_lon) == 0
    assert o.wind_uv(lat=test_lat, lon=test_lon) == 0
    assert o.origin == (43.0, -61.5)
    assert o.boundaries == bounds


def test_uniform_bathy():
    """ Test that ocean can be initialized with uniform bathymetry"""
    o = Ocean(load_bathymetry=500.5, **bounds)
    assert o.bathymetry(lat=test_lat, lon=test_lon) == 500.5
    assert o.temperature(lat=test_lat, lon=test_lon, depth=test_depth) == 0


def test_interp_uniform_temp():
    """ Test that we can interpolate a uniform ocean temperature on any set of coordinates"""
    o = Ocean(load_temperature=16.1, **bounds)
    assert o.temperature(lat=41.2, lon=-66.0, depth=-33.0) == 16.1


def test_uniform_bathy_deriv():
    """ Test that uniform bathy has derivative zero"""
    o = Ocean(load_bathymetry=-500.5, **bounds)
    assert o.bathymetry(lat=1, lon=17, dlon=1) == 0


def test_load_as_array_on_irregular_grid():
    """ Test that ocean can be initialized with data sampled on an irregular grid, passed as arrays
        Also test that we can retrieve interpolated data.
    """
    # create 100 data points with bathymetry and salinity data sampled irregularly within the bounding box
    n = 100
    lats = np.random.uniform(low=bounds["south"], high=bounds["north"], size=n)
    lons = np.random.uniform(low=bounds["west"], high=bounds["east"], size=n)
    depths = np.random.uniform(low=bounds["top"], high=bounds["bottom"], size=n)
    bathy = np.ones(n) * 60.2
    sal = np.ones(n) * 0.23

    # gather in a dict
    bathy = {"value":bathy, "lat":lats, "lon":lons}
    sal = {"value":sal, "lat":lats, "lon":lons, "depth":depths}

    # initialize Ocean class
    o = Ocean(load_bathymetry=bathy, load_salinity=sal, **bounds, interp_args={"max_size":150, "method":"linear"})

    # get interpolated values
    b = o.bathymetry(lat=43, lon=-61) #inside data region
    assert np.isclose(b, 60.2, rtol=1e-5)

    b = o.bathymetry(lat=49, lon=-60) #outside data region
    assert np.isclose(b, 60.2, rtol=1e-5)

    s = o.salinity(lat=43, lon=-61, depth=100) #inside data region 
    assert np.isclose(s, 0.23, rtol=1e-5) 

    s = o.salinity(lat=49, lon=-60, depth=10000) #outside data region
    assert np.isclose(s, 0.23, rtol=1e-5)


@pytest.mark.hycom_access
def test_small_full_ocean():
    """ Test that the ocean can be initialized for a very small region 
        using GEBCO bathymetry and HYCOM temperature and salinity
    """

    bounds = dict(
        start=datetime(2019, 1, 9, 0),
        end=datetime(2019, 1, 9, 12), #12 hours -> four 3-hour bins
        south=38.2,
        west=-55.4,
        north=39.3,
        east=-54.3,
        top=0,
        bottom=1
    )

    o = Ocean(
        load_bathymetry='gebco',
        load_temperature='hycom',
        load_salinity='hycom',
        #load_wavedirection='wwiii', #TODO: uncomment once bug in WWIII module has been fixed
        #load_waveheight='wwiii',
        #load_waveperiod='wwiii',
        #load_wind_uv='wwiii',
        **bounds,
        drop=["epoch"]
    )

    # check bathy at one location
    b = o.bathymetry(lat=39, lon=-55)        
    assert b == pytest.approx(5298, abs=10)

    # check salinity at one location, depth and instant in time
    t = datetime(2019, 1, 9, 7) 
    t0 = datetime(2000, 1, 1)
    #epoch = (t - t0).total_seconds() / 3600 #hours since since 2000-01-01 00:00
    s = o.salinity(lat=39, lon=-55, depth=0.5)
    assert s == pytest.approx(36.27, abs=0.05)