""" Unit tests for the the 'sound.output' module in the 'kadlu' package

    Authors: Oliver Kirsebom
    contact: oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/kadlu
             Project goal: Tools for underwater soundscape modeling

    License:

"""
import pytest
import kadlu.sound.output.output_database as od
import os
import numpy as np
def test_create_dataset():
    # create sample dataset
    d = od.Dataset('sample.nc', 'w')
    d.close()

    # read the sample dataset into memory
    d = od.Dataset('sample.nc', 'r')

    # check if created test dataset exists
    filepath = d.filepath()
    assert os.path.isfile(filepath)

    # check dimensions attributes of sample dataset
    assert d.get_dim('X').size == 1061
    assert d.get_dim('Y').size == 711
    assert d.get_dim('SPL').size == 531
    assert d.get_dim('F').size == 1
    assert d.get_dim('nv').size == 2

    # check variables of sample dataset

    ## check variable 'spl'
    assert d.get_var('spl').getncattr('long_name') == "sound_pressure_level_in_water_threshold"
    assert d.get_var('spl').getncattr('coverage_content_type') == 'coordinate'
    assert d.get_var('spl').getncattr('axis') == 'SPL'
    assert d.get_var('spl').get_dims()[0].size == 531
    ## check variable 'y'
    assert d.get_var('y').ndim == 2
    assert d.get_var('y').dtype == np.double
    assert d.get_var('y').getncattr('units') == 'km'

    # check global attributes of sample dataset
    assert d.getncattr('naming_authority') == 'MERIDIAN'
    assert d.getncattr('history') == 'version 1.0'
    d.close()

    # add new variable into sample datasets
    d = od.Dataset('sample.nc', 'r+')
    d.new_var('test_var', 'f8', dimensions=('X', 'Y'), attrs={
        "property": {
            'standard_name': 'test_standard_name',
            'long_name': 'test_long_name',
            'units': 'km'
        }
    })
    d.set_global_attr('test_new_attr', 'add a new test global attribute')

    ## test newly added variable
    assert d.get_var('test_var').dtype == np.double
    assert d.get_var('test_var').ndim ==2
    assert d.get_var('test_var').getncattr('standard_name') == 'test_standard_name'

    ## test newly added global attribute
    assert d.get_global_attr('test_new_attr') == 'add a new test global attribute'
    d.close()

    # remove sample dataset after test
    os.remove(filepath)