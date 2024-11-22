import os
import pytest
import kadlu
import numpy as np
from datetime import datetime, timedelta


github = os.environ.get("GITHUB_ACTIONS") is not None


# test region: 1x1 degree area in middle of the north Atlantic; 3-hour time window
kwargs = dict(
    south=36.1,
    west=-39.2,
    north=37.1,
    east=-38.2,
    start=datetime.now(),
    end=datetime.now()+timedelta(hours=2),
    top=0,
    bottom=5000,
)


def test_gfs_fetch_and_load():
    """ Check that we can fetch and load wind_u data for the test region 
        (OBS: data will only be fetched if not already present in the local database)
    """
    val, lat, lon, time = kadlu.load(source='gfs', var='wind_u', fetch=True, **kwargs)
    assert (len(val) == len(lat) == len(lon))
    assert len(np.unique(time)) == 3 
    assert np.all(np.unique(lat) == [36.25, 36.50, 36.75, 37.00])
    assert np.all(np.unique(lon) == [-39.00, -38.75, -38.50, -38.25])
    v = np.max(np.abs(val))
    assert v > 0. and v < 100.

