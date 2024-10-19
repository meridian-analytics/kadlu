""" Unit tests for the the 'geospatial.bathy_chs' module in the 'kadlu' package

    Authors: Oliver Kirsebom
    contact: oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/kadlu
             Project goal: Tools for underwater soundscape modeling
     
    License:

"""
import pytest
import numpy as np

#from kadlu.geospatial.data_sources.chs import Chs
import kadlu

#source = kadlu.Chs()

# gulf st lawrence - northumberland strait
south, west = 45.91, -63.81
north, east = 46.12, -62.92

northumberland_strait_deepest_point = 37
northumberland_strait_highest_point = -2
# http://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/fishing-marine-charts-navigation.html?title=Northumberland+Strait+boating+app#9/46.0018/-63.1677

def test_fetch_bathy():
    #if not source.fetch_bathymetry(south=south, north=north, west=west, east=east):
    if not kadlu.load(source='chs', var='bathymetry', south=south, north=north, west=west, east=east):
        print('chs query was fetched already, skipping...')
    return 

def test_load_bathy_northumberland_strait():
    bathy, lat, lon = kadlu.load(source='chs', var='bathymetry', south=south, north=north, west=west, east=east)
    assert (len(bathy) == len(lat) == len(lon))
    assert (len(bathy) > 0)
    assert np.all(np.logical_and(lat >= south, lat <= north))
    assert np.all(np.logical_and(lon >= west, lon <= east))
    assert np.all(bathy <= northumberland_strait_deepest_point)
    assert np.all(bathy >= northumberland_strait_highest_point)

def test_load_bathy():
    bathy, lat, lon = kadlu.load(source='chs', var='bathymetry', south=43.1, west=-59.8, north=43.8, east=-59.2)
    assert (len(bathy) == len(lat) == len(lon))
    assert (len(bathy) > 0), 'no data'
    assert np.all(np.logical_and(lat >= 43.1, lat <= 43.8))
    assert np.all(np.logical_and(lon >= -59.8, lon <= -59.2))
    assert np.all(bathy >= -15000)
    assert np.all(bathy <= 10000)
