""" Unit tests for the the 'sound.geophony' module in the 'kadlu' package

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
from kadlu.sound.geophony import geophony, transmission_loss, kewley_sl_func, source_level
from kadlu.geospatial.ocean import Ocean
from kadlu.utils import R1_IUGG, deg2rad

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir), "assets")


def test_kewley_sl_func():
    sl1 = kewley_sl_func(freq=10, wind_uv=0)
    sl2 = kewley_sl_func(freq=40, wind_uv=2.57)
    assert sl1 == sl2
    assert sl2 == 40.0
    sl3 = kewley_sl_func(freq=40, wind_uv=5.14)
    assert sl3 == 44.0
    sl4 = kewley_sl_func(freq=100, wind_uv=5.14)
    assert sl4 == 42.5


def test_source_level():
    ok = {'load_bathymetry': 10000, 'load_wind_uv': 5.14}
    o = Ocean(**ok)
    sl = source_level(freq=10,
                      x=0,
                      y=0,
                      area=1,
                      ocean=o,
                      sl_func=kewley_sl_func)
    assert sl == 44.0
    sl = source_level(freq=100,
                      x=[0, 100],
                      y=[0, 100],
                      area=[1, 2],
                      ocean=o,
                      sl_func=kewley_sl_func)
    assert sl[0] == 42.5
    assert sl[1] == sl[0] + 10 * np.log10(2)


def test_geophony_flat_seafloor():
    """ Check that we can execute the geophony method for a
        flat seafloor and uniform sound speed profile"""
    kwargs = {
        'load_bathymetry': 10000,
        'load_wind_uv': 1.0,
        'ssp': 1480,
        'angular_bin': 90,
        'dr': 1000,
        'dz': 1000
    }
    geo = geophony(freq=100,
                   south=44,
                   north=46,
                   west=-60,
                   east=-58,
                   depth=[100, 2000],
                   xy_res=71,
                   **kwargs)
    spl = geo['spl']
    x = geo['x']
    y = geo['y']
    assert x.shape[0] == 3
    assert y.shape[0] == 5
    assert spl.shape[0] == 3
    assert spl.shape[1] == 5
    assert spl.shape[2] == 2
    assert np.all(np.diff(x) == 71e3)
    assert np.all(np.diff(y) == 71e3)
    # try again, but this time for specific location
    kwargs = {
        'load_bathymetry': 10000,
        'load_wind_uv': 1.0,
        'ssp': 1480,
        'angular_bin': 90,
        'dr': 1000,
        'dz': 1000,
        'propagation_range': 50
    }
    geo = geophony(freq=100, lat=45, lon=-59, depth=[100, 2000], **kwargs)


def test_geophony_in_canyon(bathy_canyon):
    """ Check that we can execute the geophony method for a
        canyon-shaped bathymetry and uniform sound speed profile"""
    kwargs = {
        'load_bathymetry': bathy_canyon,
        'load_wind_uv': 1.0,
        'ssp': 1480,
        'angular_bin': 90,
        'dr': 1000,
        'dz': 1000
    }
    z = [100, 1500, 3000]
    geo = geophony(freq=10,
                   south=43,
                   north=46,
                   west=60,
                   east=62,
                   depth=z,
                   xy_res=71,
                   **kwargs)
    spl = geo['spl']
    x = geo['x']
    y = geo['y']
    assert spl.shape[0] == x.shape[0]
    assert spl.shape[1] == y.shape[0]
    assert spl.shape[2] == len(z)
    assert np.all(np.diff(x) == 71e3)
    assert np.all(np.diff(y) == 71e3)
    # check that noise is NaN below seafloor and non Nan above
    bathy = np.swapaxes(
        np.reshape(geo['bathy'], newshape=(y.shape[0], x.shape[0])), 0, 1)
    bathy = bathy[:, :, np.newaxis]
    xyz = np.ones(shape=bathy.shape) * z
    idx = np.nonzero(xyz >= bathy)
    assert np.all(np.isnan(spl[idx]))
    idx = np.nonzero(xyz < bathy)
    assert np.all(~np.isnan(spl[idx]))


def test_transmission_loss_real_world_env():
    """ Check that we can initialize a transmission loss object
        for a real-world environment and obtain the expected result """

    from datetime import datetime
    bounds = dict(start=datetime(2015, 1, 1),
                  end=datetime(2015, 1, 2),
                  top=0,
                  bottom=10000)
    src = dict(load_bathymetry='gebco',
               load_temperature='hycom',
               load_salinity='hycom')
    sound_source = {
        'freq': 200,
        'lat': 43.8,
        'lon': -59.04,
        'source_depth': 12
    }
    seafloor = {'sound_speed': 1700, 'density': 1.5, 'attenuation': 0.5}
    transm_loss, ocean = transmission_loss(seafloor=seafloor,
                                           propagation_range=20,
                                           **src,
                                           **bounds,
                                           **sound_source,
                                           dr=100,
                                           angular_bin=45,
                                           dz=50,
                                           return_ocean=True)

    # check fetching and interpolation of ocean variables
    lats = np.linspace(43.7, 43.9, num=3)
    lons = np.linspace(-59.14, -58.94, num=3)
    seafloor_depth = ocean.bathymetry(lat=lats, lon=lons, grid=True)
    max_depth = 1712.83265634
    depths = np.linspace(0, max_depth, num=3)
    temp = ocean.temperature(lat=lats, lon=lons, depth=depths, grid=True)
    salinity = ocean.salinity(lat=lats, lon=lons, depth=depths, grid=True)

    # GEBCO-NetCDF 2020
    #answ_seafloor_depth = np.array([[267.75, 606.44, 1398.15],
    #                                [99.94, 273.12, 1651.99],
    #                                [80.17, 370.65, 1307.41]])

    # GEBCO-NetCDF 2021
    answ_seafloor_depth = np.array([[ 269.1 , 616.11, 1466.31],
                                    [ 101.66, 268.88, 1595.38],
                                    [  85.25, 419.49, 1435.14]])

    answ_temp = np.array([[[7.32, 4.67, 4.67], [5.49, 10., 10.],
                           [3.98, 9.36, 9.36]],
                          [[7.35, 4.04, 3.99], [6.3, 4.47, 4.47],
                           [3.88, 5.36, 5.36]],
                          [[5.66, 4.17, 3.72], [6.83, 4.07, 3.76],
                           [4.5, 4.31, 4.31]]])

    answ_salinity = np.array([[[33.3, 34.97, 34.97], [32.52, 34.79, 34.79],
                               [31.71, 34.32, 34.32]],
                              [[33.31, 34.95, 34.95], [33., 34.95, 34.95],
                               [31.77, 34.94, 34.94]],
                              [[32.96, 34.95, 34.96], [33.23, 34.95, 34.95],
                               [32.1, 34.95, 34.95]]])

    ##print(temp.round(2))

    np.testing.assert_array_almost_equal(seafloor_depth,
                                         answ_seafloor_depth,
                                         decimal=2)

    np.testing.assert_array_almost_equal(temp, answ_temp, decimal=2)
    np.testing.assert_array_almost_equal(salinity, answ_salinity, decimal=2)

    # check transmission loss calculation results
    tl_h, ax_h, tl_v, ax_v = transm_loss.calc(vertical=True)

    answ_h = np.array(
        [[97.6, 117.2, 114.9, 120.4, 119.0, 120.1, 121.1, 121.9, 122.7, 123.3],
         [97.6, 116.4, 117.7, 117.6, 120.2, 122.0, 121.4, 123.7, 120.1, 122.7],
         [97.6, 114.3, 117.2, 118.4, 120.6, 120.2, 122.2, 120.4, 125.3, 121.1],
         [97.6, 117.6, 117.9, 117.8, 117.7, 121.1, 119.9, 120.7, 122.4, 123.6],
         [97.6, 114.7, 117.5, 117.5, 116.9, 120.1, 121.7, 120.1, 119.9, 121.3],
         [97.6, 110.4, 115.5, 119.6, 119.0, 117.9, 117.7, 123.4, 120.7, 122.3],
         [97.6, 110.2, 114.9, 117.7, 123.5, 120.1, 121.0, 121.5, 123.5, 120.4],
         [97.6, 112.0, 115.6, 118.1, 119.1, 120.2, 121.2, 122.0, 122.8,
          123.5]])

    answ_v = np.array(
        [[31.9, 65.4, 68.1, 70.1, 71.1, 72.2, 73.2, 74.0, 74.8, 75.5, 76.1],
         [
             53.4, 147.1, 144.3, 151.1, 145.6, 150.4, 150.7, 150.0, 150.5,
             151.3, 152.0
         ],
         [
             59.5, 164.6, 161.3, 168.0, 162.7, 167.4, 167.7, 167.0, 167.5,
             168.3, 169.0
         ],
         [
             63.7, 174.9, 171.6, 178.2, 173.0, 177.7, 178.0, 177.3, 177.7,
             178.6, 179.2
         ],
         [
             67.6, 182.5, 179.2, 185.9, 180.6, 185.3, 185.6, 184.9, 185.4,
             186.2, 186.9
         ],
         [
             71.8, 189.1, 185.9, 192.5, 187.3, 192.0, 192.3, 191.6, 192.1,
             192.9, 193.6
         ],
         [
             77.9, 194.6, 193.3, 201.1, 196.7, 200.7, 201.4, 201.7, 202.3,
             203.1, 203.7
         ],
         [
             98.9, 213.8, 214.7, 222.8, 219.7, 222.9, 223.8, 224.6, 225.3,
             226.0, 226.7
         ]])

    res_h = tl_h[0, 0, :, ::20]
    assert np.all(np.abs(res_h - answ_h) < 0.03 * np.abs(answ_h)) #agree within 3%

    res_v = tl_v[0, 1::10, ::20, 0]
    assert np.all(np.abs(res_v - answ_v) < 0.06 * np.abs(answ_v)) #agree within 6%

    assert tl_h.shape == (1, 1, 8, 200), f'tl_h.shape = {tl_h.shape}'
    assert tl_v.shape == (1, 73, 201, 8), f'tl_v.shape = {tl_v.shape}'


def test_transmission_loss_flat_seafloor():
    """ Check that we can initialize a transmission loss object
        for a flat seafloor and uniform sound speed profile """
    transm_loss = transmission_loss(freq=100,
                                    source_depth=75,
                                    propagation_range=0.5,
                                    load_bathymetry=2000,
                                    ssp=1480,
                                    angular_bin=10)
    tl_h, ax_h, tl_v, ax_v = transm_loss.calc(vertical=True)
    answ = np.genfromtxt(os.path.join(path_to_assets,
                                      'lloyd_mirror_f100Hz_SD75m.csv'),
                         delimiter=",")
    assert answ.shape == tl_v[0, :, :, 0].shape
    np.testing.assert_array_almost_equal(-tl_v[0, 1:, :, 0],
                                         answ[1:, :],
                                         decimal=3)


#def test_test():
#    from datetime import datetime
#    bounds = dict(
#               south=43.53, north=44.29, west=-59.84, east=-58.48,
#               start=datetime(2015,1,1), end=datetime(2015,1,2),
#               top=0, bottom=10000
#             )
#    src = dict(load_bathymetry='chs', load_temperature='hycom', load_salinity='hycom')
#    sound_source = {'freq': 200, 'lat': 43.8, 'lon': -59.04, 'source_depth': 12}
#    o = Ocean(**src, **bounds)
#    seafloor = {'sound_speed':1700,'density':1.5,'attenuation':0.5}
#    transm_loss = transmission_loss(seafloor=seafloor, propagation_range=20, **src, **bounds, **sound_source, ssp=1480, dz=50)
#    transm_loss.calc(vertical=False)
