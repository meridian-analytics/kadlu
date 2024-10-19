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


@pytest.mark.hycom_access
def test_transmission_loss_real_world_env():
    """ Check that we can initialize a transmission loss object
        for a real-world environment and obtain the expected result 
        
        #TODO: understand why test fails for 'cubic' interpolation
                has to do with poor interpolation of temperature and salinity for certain parts of the coordinate space ...
        
    """

    from datetime import datetime
    bounds = dict(start=datetime(2019, 1, 1),
                  end=datetime(2019, 1, 2),
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
    transm_loss, ocean = transmission_loss(
        seafloor=seafloor,
        propagation_range=20,
        **src,
        **bounds,
        **sound_source,
        dr=100,
        angular_bin=45,
        dz=50,
        #drop="epoch", #drop time dimension
        #interp_args={"method":"linear"},  
        return_ocean=True,
    )

    # check fetching and interpolation of ocean variables
    lats = np.linspace(43.7, 43.9, num=3)
    lons = np.linspace(-59.14, -58.94, num=3)
    t = datetime(2019, 1, 1, 12) 
    t0 = datetime(2000, 1, 1)
    epoch = (t - t0).total_seconds() / 3600 #hours since since 2000-01-01 00:00
    seafloor_depth = ocean.bathymetry(lat=lats, lon=lons, grid=True)
    max_depth = 1712.83265634
    depths = np.linspace(0, max_depth, num=3)
    temp = ocean.temperature(lat=lats, lon=lons, depth=depths, epoch=epoch, grid=True)
    salinity = ocean.salinity(lat=lats, lon=lons, depth=depths, epoch=epoch, grid=True)

    temp = temp[:,:,0,:]
    salinity = salinity[:,:,0,:]

    answ_seafloor_depth = np.array([[266.47, 606.66, 1474.88], [101.38, 270.35, 1632.57], [84.09, 398.28, 1407.04]])
    answ_temp = np.array([[[3.23, 4.24, 3.78], [3.67, 4.15, 3.78], [4.25, 4.23, 3.79]], [[2.84, 4.18, 3.78], [3.02, 4.18, 3.79], [3.46, 4.2, 3.79]], [[2.88, 4.16, 3.79], [2.85, 4.16, 3.79], [3.01, 4.22, 3.76]]])
    answ_salinity = np.array([[[32.4, 34.93, 34.95], [32.48, 34.93, 34.95], [32.79, 34.93, 34.95]], [[32.26, 34.93, 34.95], [32.36, 34.93, 34.95], [32.49, 34.93, 34.95]], [[31.92, 34.93, 34.95], [32.17, 34.93, 34.95], [32.29, 34.93, 34.96]]])

    #print(seafloor_depth.round(2).tolist())
    #print(temp.round(2).tolist())
    #print(salinity.round(2).tolist())

    np.testing.assert_array_almost_equal(seafloor_depth, answ_seafloor_depth, decimal=2)
    np.testing.assert_array_almost_equal(temp, answ_temp, decimal=2)
    np.testing.assert_array_almost_equal(salinity, answ_salinity, decimal=2)

    # check transmission loss calculation results
    tl_h, ax_h, tl_v, ax_v = transm_loss.calc(vertical=True)

    answ_h = np.array(
        [[97.5, 116.9, 116.9, 117.0, 120.7, 120.1, 121.4, 122.4, 123.5, 124.4], [97.5, 114.9, 113.9, 119.4, 122.4, 120.6, 120.6, 120.4, 120.2, 123.2], [97.5, 114.2, 116.6, 118.7, 119.9, 119.6, 121.9, 121.6, 120.9, 124.0], [97.5, 113.9, 118.7, 119.3, 116.4, 120.4, 123.3, 119.1, 120.7, 126.3], [97.5, 115.0, 121.2, 118.9, 120.1, 119.3, 120.1, 122.1, 122.2, 122.5], [97.5, 116.8, 115.3, 117.1, 123.4, 118.9, 118.4, 123.7, 121.9, 120.5], [97.5, 115.4, 119.6, 118.6, 118.4, 120.3, 123.6, 119.4, 123.4, 121.8], [97.5, 116.5, 119.7, 118.6, 120.0, 121.3, 122.5, 123.6, 124.5, 125.5]]
    )

    answ_v = np.array(
        [[31.9, 65.4, 68.5, 70.0, 71.1, 72.1, 73.4, 74.5, 75.6, 76.5, 77.3], [53.4, 142.2, 161.3, 144.4, 141.4, 143.5, 144.7, 145.5, 146.6, 147.5, 148.4], [59.5, 158.7, 176.4, 161.4, 158.5, 160.5, 161.7, 162.5, 163.6, 164.5, 165.4], [63.7, 168.8, 186.2, 171.7, 168.7, 170.7, 172.0, 172.8, 173.9, 174.8, 175.7], [67.5, 176.4, 193.7, 179.3, 176.4, 178.4, 179.6, 180.4, 181.5, 182.4, 183.3], [71.6, 183.0, 200.2, 185.9, 183.0, 184.9, 186.2, 187.0, 188.1, 189.0, 189.9], [77.3, 194.4, 211.7, 195.6, 194.7, 196.9, 198.3, 199.0, 200.1, 200.9, 201.7], [93.1, 212.6, 220.4, 213.7, 215.8, 217.6, 219.0, 219.7, 220.8, 221.6, 222.4]]
    )

    res_h = tl_h[0, 0, :, ::20]
    assert np.all(np.abs(res_h - answ_h) < 0.03 * np.abs(answ_h)) #agree within 3%
    #print(np.round(res_h, 1).tolist())

    res_v = tl_v[0, 1::10, ::20, 0]
    assert np.all(np.abs(res_v - answ_v) < 0.06 * np.abs(answ_v)) #agree within 6%
    #print(np.round(res_v, 1).tolist())

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
