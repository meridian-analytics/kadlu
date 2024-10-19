import kadlu


def test_load_bathy():
    bathy = kadlu.load(source='gebco', var='bathymetry', **kadlu.defaults)
