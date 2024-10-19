""" Unit tests for the sound speed module within the kadlu library
"""
import pytest
import os
import numpy as np
from kadlu.sound.sound_speed import SoundSpeed
from kadlu.geospatial.ocean import Ocean

path_to_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "assets")


def test_ocean_or_ssp_must_be_specified():
    with pytest.raises(AssertionError):
        SoundSpeed(num_depths=50, rel_err=None)


def test_sound_speed_from_uniform_data():
    # environment data provider
    o = Ocean(load_bathymetry=1000, load_temperature=4, load_salinity=3)
    # instance of sound speed class
    _ = SoundSpeed(o, num_depths=50, rel_err=None)


def test_sound_speed_from_uniform_ssp():
    # instance of sound speed class
    ss = SoundSpeed(ssp=1499, num_depths=50, rel_err=None)
    # evaluate
    c = ss(x=0, y=0, depth=0)
    assert c == 1499


def test_sound_speed_from_ssp():
    # sound speed profile
    z0 = np.array([0, 10, 20, 30, 60])
    c0 = np.array([1500, 1510, 1512, 1599, 1489])
    # instance of sound speed class
    ss = SoundSpeed(ssp=(c0, z0), num_depths=50, rel_err=None)
    # evaluate
    c = ss(x=0, y=0, depth=z0, grid=True)
    assert np.all(np.abs(c - c0) < 1E-6)
