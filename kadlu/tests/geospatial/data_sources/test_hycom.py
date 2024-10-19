import pytest
import kadlu
import numpy as np
from datetime import datetime
from kadlu.geospatial.data_sources.hycom import Hycom
from kadlu.geospatial.data_sources.data_util import reshape_4D


@pytest.mark.hycom_access
def test_fetch_load_salinity():
    """ Check that we can load salinity for a small region across the anti-meridian 
        down to 100 meters depth.       
    
        Note: If the data already exists in the local database instance, it will not 
        be re-downloaded.
    """
    south, west = 44, 179
    north, east = 45, -179
    top, bottom = 0, 100
    start = datetime(2020, 1, 10)
    end = datetime(2020, 1, 10, 12)

    data = kadlu.load(
        var='salinity',
        source='hycom',
        south=south,
        north=north,
        west=west,
        east=east,
        start=start,
        end=end,
        top=top,
        bottom=bottom
    )

    data = reshape_4D(data)

    # check that lat/lon coordinates are what they should be
    lats_answ = np.arange(44.0, 45.01, 0.04)
    assert np.all(np.isclose(data["lats"], lats_answ))
    lons_answ = np.concatenate([np.arange(-179.92, -179.03, 0.08), np.arange(179.04, 180.01, 0.08)])
    assert np.all(np.isclose(data["lons"], lons_answ))

    # check that salinity is 33 +- 1
    assert np.all(np.isclose(data["values"], 33, atol=1))


@pytest.mark.hycom_access
def test_load_water_uv():
    """ Check that we can load water speeds for a small region of the Atlantic ocean
        down to 500 m depth.
    
        Note: If the data already exists in the local database instance, it will not 
        be re-downloaded.
    """
    start = datetime(2020, 1, 10)
    end = datetime(2020, 1, 10, 12)

    bounds = dict(
        south=44.01,
        north=44.31,
        west=-63.1,
        east=-62.9,
        top=0,
        bottom=500,
        start=start,
        end=end,
    )

    data = kadlu.load(source='hycom', var='water_uv', **bounds)

    assert data.shape == (5, 34780)

    data = reshape_4D(data)

    # check that lat/lon,depth,epoch coordinates are what they should be

    lats_answ = np.array([44.04000092, 44.08000183, 44.11999893, 44.15999985, 44.20000076, 44.24000168, 44.27999878])
    assert np.all(np.isclose(data["lats"], lats_answ, rtol=1E-6))

    lons_answ = np.array([-63.03997803, -62.96002197])
    assert np.all(np.isclose(data["lons"], lons_answ, rtol=1E-6))

    depths_answ = np.array([  0.,   2.,   4.,   6.,   8.,  10.,  12.,  15.,  20.,  25.,  30., 35.,  40.,  45.,  50.,  60.,  70.,  80.,  90., 100., 125., 150., 200.])
    assert np.all(np.isclose(data["depths"], depths_answ, rtol=1E-6))

    times_answ = np.array([175536., 175539., 175542., 175545., 175548.])
    assert np.all(np.isclose(data["times"], times_answ, rtol=1E-6))


@pytest.mark.hycom_access
def test_load_water_u():
    """ Check that we can load water speed u component for a small region of the Atlantic ocean
        down to 500 m depth.

        Note: If the data already exists in the local database instance, it will not 
        be re-downloaded.
    """
    start = datetime(2020, 1, 10)
    end = datetime(2020, 1, 10, 12)

    bounds = dict(
        south=44.01,
        north=44.31,
        west=-63.1,
        east=-62.9,
        top=0,
        bottom=500,
        start=start,
        end=end,
    )

    H = Hycom()

    H.fetch_hycom(var="water_u", max_attempts=4, kwargs=bounds)

    data = H.load_water_u(**bounds)

    assert len(data) > 0  #check that something was loaded


'''

    TODO: finish implementing below test to check that 
            there is no water_u data on land

def test_no_data_on_land():

    start = datetime(2020, 1, 10)
    end = datetime(2020, 1, 10, 12)

    # west coast of Olympic peninsula, WA
    bounds = dict(
        south=47.5,
        north=48.4,
        west=-125,
        east=-124,
        top=0,
        bottom=500,
        start=start,
        end=end,
    )

    H = Hycom()
    data = H.load_water_u(**bounds)

    lat = data[1]
    lon = data[2]
    import matplotlib.pyplot as plt
    plt.scatter(lon, lat)
    plt.show()
'''
