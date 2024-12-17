import kadlu
from datetime import datetime

# gulf st lawrence - small test area
south = 45.01
north = 45.99923
west = -63.99
east = -63
top = 0
bottom = 4000
start = datetime(2000, 1, 10)
end = datetime(2000, 1, 10, 12)

bounds = dict(
    south=45,
    north=46,
    west=-64,
    east=-63,
    top=0,
    bottom=4000,
    start=datetime(2000, 1, 10),
    end=datetime(2000, 1, 10, 12),
)


def test_fetch_load_salinity():
    south, west = 44, 179
    north, east = 45, -179
    top, bottom = 0, 100

    val, lat, lon, time, depth = kadlu.load(var='salinity',
                                            source='hycom',
                                            south=south,
                                            north=north,
                                            west=west,
                                            east=east,
                                            start=start,
                                            end=end,
                                            top=top,
                                            bottom=bottom)


def test_load_water_uv():
    bounds = dict(
        south=44.01,
        north=44.31,
        west=-63.1,
        east=-62.9,
        top=0,
        bottom=500,
        start=datetime(2000, 1, 10),
        end=datetime(2000, 1, 10, 12),
    )

    data = kadlu.load(source='hycom', var='water_uv', **bounds)
