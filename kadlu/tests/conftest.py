import pytest
import os
import numpy as np
import kadlu

path_to_assets = os.path.join(os.path.dirname(__file__),"assets")

# set storage dir for pytest
path_to_storage_dir = os.path.join(os.path.dirname(__file__),"kadlu-data")
if not os.path.exists(path_to_storage_dir):
    os.makedirs(path_to_storage_dir)

kadlu.storage_cfg(setdir=path_to_storage_dir)


@pytest.fixture
def one():
    return 1

@pytest.fixture
def bathy_canyon():
    depth = 2e3 #2km
    sigma = 0.5
    def canyon_axis(x):
        y = 45  # + (x - 61) 
        return y

    x = np.linspace(59, 63, num=200) #lons
    y = np.linspace(42, 48, num=200) #lats
    xv, yv = np.meshgrid(x, y)
    bathy = depth * np.exp(-(yv - canyon_axis(xv))**2 / (2 * sigma**2))
    return (bathy, y, x)
