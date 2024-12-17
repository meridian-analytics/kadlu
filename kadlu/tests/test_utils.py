""" Unit tests for the the 'utility' module in the 'kadlu' package

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
import math
import numpy as np
from kadlu.utils import LLtoXY, XYtoLL, interp_grid_1d, create_boolean_array

def test_can_convert_single_point_from_ll_to_xy():
    lat_ref = 45
    lon_ref = 10
    lats = 45
    lons = 20
    x, y = LLtoXY(lats, lons, lat_ref, lon_ref)    
    assert x == pytest.approx(788E3, 1E3)
    assert y == pytest.approx(0, 1E3)
    lats = 46
    lons = 10
    x, y = LLtoXY(lats, lons, lat_ref, lon_ref)    
    assert x == pytest.approx(0, 1E3)
    assert y == pytest.approx(111E3, 1E3)

def test_can_convert_several_points_from_ll_to_xy():
    lat_ref = 45
    lon_ref = 10
    lats = [45, 45]
    lons = [20, 30]
    x, y = LLtoXY(lats, lons, lat_ref, lon_ref)    
    assert x[0] == pytest.approx(788E3, 1E3)
    assert y[0] == pytest.approx(0, 1E3)
    assert x[1] == 2*x[0]
    assert y[1] == pytest.approx(0, 1E3)

def test_can_convert_single_point_from_xy_to_ll():
    lat_ref = 45
    lon_ref = 10
    xs = 788E3
    ys = 0
    lat, lon = XYtoLL(xs, ys, lat_ref, lon_ref)    
    assert lon == pytest.approx(lon_ref+10, 0.1)
    assert lat == pytest.approx(lat_ref, 0.1)
    xs = 0
    ys = 111E3
    lat, lon = XYtoLL(xs, ys, lat_ref, lon_ref)    
    assert lon == pytest.approx(lon_ref, 0.1)
    assert lat == pytest.approx(lat_ref+1, 0.1)

def test_interp_grid_1d():
    # example function y(x)
    x = np.linspace(0, 2*np.pi, num=100)
    y = np.sin(pow(x/(2*np.pi),2)*x)
    x = np.concatenate((np.linspace(-10,0,num=100), x))
    y = np.concatenate((np.zeros(100), y))
    # compute grid
    indices, max_err = interp_grid_1d(y=y, x=x, num_pts=101, rel_err=0.02)
    assert len(indices) == 15
    assert max_err == pytest.approx(0.01842, abs=0.001)
    answ = np.array([0, 115, 133, 156, 163, 168, 172, 176, 185, 188, 190, 192, 194, 196, 199])
    assert np.all(indices == answ) 

def test_can_convert_grid_from_xy_to_ll():
    lat_ref = 45
    lon_ref = 10
    xs = [0, 788E3]
    ys = [0, 111E3]
    lat, lon = XYtoLL(xs, ys, lat_ref, lon_ref, grid=True)    
    assert lat.shape[0] == 2
    assert lat.shape[1] == 2
    assert lon[0,0] == pytest.approx(lon_ref, 0.1)
    assert lat[0,0] == pytest.approx(lat_ref, 0.1)
    assert lon[1,0] == pytest.approx(lon_ref, 0.1)
    assert lat[1,0] == pytest.approx(lat_ref+1, 0.1)
    assert lon[0,1] == pytest.approx(lon_ref+10, 0.1)
    assert lat[0,1] == pytest.approx(lat_ref, 0.1)

def test_can_convert_grid_from_xy_to_ll_with_z_coordinate():
    lat_ref = 45
    lon_ref = 10
    xs = [0, 788E3]
    ys = [0, 111E3]
    zs = [0, 100, 200]
    lat, lon, _ = XYtoLL(xs, ys, lat_ref, lon_ref, grid=True, z=zs)    
    assert lat.shape[0] == 2
    assert lat.shape[1] == 2
    assert lat.shape[2] == 3
    assert np.all(np.abs(lon[0,0,:] - lon_ref) < 0.1)
    assert np.all(np.abs(lat[0,0,:] - lat_ref) < 0.1)
    assert np.all(np.abs(lon[1,0,:] - lon_ref) < 0.1)
    assert np.all(np.abs(lat[1,0,:] - (lat_ref+1)) < 0.1)
    assert np.all(np.abs(lon[0,1,:] - (lon_ref+10)) < 0.1)
    assert np.all(np.abs(lat[0,1,:] - lat_ref) < 0.1)

def test_create_boolean_array():
    a = create_boolean_array(n=4, step=math.inf)
    assert np.all(a == np.array([True, False, False, False]))
    a = create_boolean_array(n=6, step=3)
    assert np.all(a == np.array([True, False, False, True, False, False]))
