import os
import pytest
import kadlu
import numpy as np
from datetime import datetime, timedelta
from kadlu.geospatial.data_sources.cmems import clear_cache_cmems

# use @pytest.mark.cmems_access as decorater for tests that require access to the CMEMS API to fetch data


github = os.environ.get("GITHUB_ACTIONS") is not None


# whether to automatically download data using the CMEMS API, if the data are not already in the local database
# (should normally be set to False since we don't want to perform API calls every time the tests are run.)
fetch = False


# test region: 1x1 degree area in middle of the north Atlantic; 3-hour time window
kwargs = dict(
    south=36.1,
    west=-39.2,
    north=37.1,
    east=-38.2,
    start=datetime(2024, 7, 1, 8, 0, 0),
    end=datetime(2024, 7, 1, 10, 0, 0),
    top=0,
    bottom=5000,
)


# clear any cached CMEMS data files
clear_cache_cmems()


def test_cmems_load():
    """ Check that we can load water_u data (that have already been fetched) for the test region """
    val, lat, lon, time = kadlu.load(source='cmems', var='water_u', fetch=fetch, **kwargs)
    if len(val) == 0:
        pytest.xfail(
            "Unable to test if CMEMS wave data are loaded correctly "\
            "because data have not been fetched from the CMEMS data repository."
        )

    assert (len(val) == len(lat) == len(lon))
    assert len(np.unique(time)) == 3 #3 hours
    assert np.all(np.unique(time) == [214759.5, 214760.5, 214761.5])
    assert np.all(np.unique(lat) == [36.166668, 36.25, 36.333332, 36.416668, 36.5, 36.583332, 36.666668, 36.75, 36.833332, 36.916668, 37.0, 37.083332])
    assert np.all(np.unique(lon) == [-39.166668, -39.083332, -39.0, -38.916668, -38.833332, -38.75, -38.666668, -38.583332, -38.5, -38.416668, -38.333332, -38.25])
    assert np.max(val) > 0. and np.max(np.abs(val)) < 1.0


@pytest.mark.cmems_access
def test_cmems_fetch_and_load():
    """ Check that we can fetch and load water_v data for the test region 
        (OBS: data will only be fetched if not already present in the local database)
    """
    val, lat, lon, time = kadlu.load(source='cmems', var='water_v', fetch=True, **kwargs)
    assert (len(val) == len(lat) == len(lon))
    assert len(np.unique(time)) == 3 #3 hours
    assert np.all(np.unique(time) == [214759.5, 214760.5, 214761.5])
    assert np.all(np.unique(lat) == [36.166668, 36.25, 36.333332, 36.416668, 36.5, 36.583332, 36.666668, 36.75, 36.833332, 36.916668, 37.0, 37.083332])
    assert np.all(np.unique(lon) == [-39.166668, -39.083332, -39.0, -38.916668, -38.833332, -38.75, -38.666668, -38.583332, -38.5, -38.416668, -38.333332, -38.25])
    assert np.max(val) > 0. and np.max(np.abs(val)) < 1.0

