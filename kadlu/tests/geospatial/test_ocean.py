import logging
from datetime import datetime

from kadlu.geospatial.ocean import Ocean

bounds = dict(start=datetime(2015, 1, 9),
              end=datetime(2015, 1, 9, 3),
              south=44,
              west=-64.5,
              north=46,
              east=-62.5,
              top=0,
              bottom=5000)

test_lat, test_lon, test_depth = bounds['south'], bounds['west'], bounds['top']


def test_null_ocean():
    """ Test that ocean is initialized with all variables set to
        null (0) when default=False"""

    o = Ocean(load_bathymetry=0, **bounds)

    assert o.bathymetry(test_lat, test_lon) == 0
    assert o.temperature(test_lat, test_lon, test_depth) == 0
    assert o.salinity(test_lat, test_lon, test_depth) == 0
    assert o.wavedir(test_lat, test_lon) == 0
    assert o.waveheight(test_lat, test_lon) == 0
    assert o.waveperiod(test_lat, test_lon) == 0
    assert o.wind_uv(test_lat, test_lon) == 0
    assert o.origin == (45, -63.5)
    assert o.boundaries == bounds


def test_uniform_bathy():
    """ Test that ocean can be initialized with uniform bathymetry"""
    o = Ocean(load_bathymetry=500.5, **bounds)

    assert o.bathymetry(test_lat, test_lon) == 500.5
    assert o.temperature(test_lat, test_lon, test_depth) == 0


def test_interp_uniform_temp():
    """ Test that we can interpolate a uniform ocean temperature
        on any set of coordinates"""
    o = Ocean(load_temperature=16.1, **bounds)
    assert o.temperature(lat=41.2, lon=-66.0, depth=-33.0) == 16.1


def test_uniform_bathy_deriv():
    """ Test that uniform bathy has derivative zero"""
    o = Ocean(load_bathymetry=-500.5, **bounds)
    assert o.bathymetry_deriv(lat=1, lon=17, axis='lon') == 0


def test_small_full_ocean():
    """ test that the ocean can be initialized for a very small region """

    bounds = dict(start=datetime(2015, 1, 9),
                  end=datetime(2015, 1, 9, 3),
                  south=44.2,
                  west=-64.4,
                  north=44.21,
                  east=-64.39,
                  top=0,
                  bottom=1)
    try:
        o = Ocean(load_bathymetry='gebco',
                  load_temperature='hycom',
                  load_salinity='hycom',
                  load_wavedirection='wwiii',
                  load_waveheight='wwiii',
                  load_waveperiod='wwiii',
                  load_wind_uv='wwiii',
                  **bounds)
    except AssertionError as err:
        # this is intended behaviour
        logging.info('CAUGHT EXCEPTION: ' + str(err))
        pass
    except Exception as err:
        raise err


def test_wind_water_uv():
    o = Ocean(load_water_u='hycom',
              load_water_v='hycom',
              load_water_uv='hycom',
              load_wind_u='wwiii',
              load_wind_v='wwiii',
              load_wind_uv='wwiii',
              **bounds)
