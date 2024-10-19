import kadlu
from datetime import datetime

# gulf st lawrence
start = datetime(2014, 2, 3, 0, 0, 0, 0)
end = datetime(2014, 2, 3, 3, 0, 0, 0)
y, x, offset = 47.133, -63.51, 1
south, west = y - offset, x - offset
north, east = y + offset, x + offset


def test_wwiii_load_windwaveheight():
    result = kadlu.load(source='wwiii',
                        var='waveheight',
                        south=south,
                        west=west,
                        north=north,
                        east=east,
                        start=start,
                        end=end)


def test_wwiii_load_wavedir():
    result = kadlu.load(source='wwiii',
                        var='wavedir',
                        south=south,
                        west=west,
                        north=north,
                        east=east,
                        start=start,
                        end=end)


def test_load_wind():
    ns_offset = 1
    ew_offset = 1

    uvval, lat, lon, epoch = kadlu.load(source='wwiii',
                                        var='wind_uv',
                                        start=datetime(2016, 3, 9),
                                        end=datetime(2016, 3, 11),
                                        south=44.5541333 - ns_offset,
                                        west=-64.17682 - ew_offset,
                                        north=44.5541333 + ns_offset,
                                        east=-64.17682 + ew_offset,
                                        top=0,
                                        bottom=0)

    assert (len(uvval) == len(lat) == len(lon))
    assert len(uvval) > 0
