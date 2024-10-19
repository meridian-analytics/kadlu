import os
import logging
from datetime import datetime, timedelta
from multiprocessing import Process, Queue

import numpy as np
import imageio
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#import cartopy
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
from scipy.interpolate import griddata

from kadlu.geospatial.data_sources.source_map import load_map
from kadlu.geospatial.data_sources.data_util import fmt_coords
from kadlu.geospatial.data_sources.data_util import storage_cfg

'''

logging.getLogger('cartopy').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


config = dict(
        bgcontour   = lambda v: np.linspace(min(v)-.1, max(v)+.1, 3),
        bathymetry=dict(
            cm      = plt.cm.bone.reversed(),
            alpha   = 0.9,
            levels  = lambda v, n=12: np.linspace(1, max(v)-1, n),
            norm    = lambda v: matplotlib.colors.Normalize(vmin=0, vmax=max(v)-1),
            title   = 'bathymetry (metres)'),
        temp=dict(
            cm      = plt.cm.coolwarm, 
            alpha   = 0.8,
            levels  = lambda v, n=12: np.linspace(min(v)-.1, max(v)+.1, n),
            norm    = lambda v=None: matplotlib.colors.Normalize(vmin=-5, vmax=20),
            title   = 'temperature (celsius)'),
        salinity=dict(
            cm      = plt.cm.viridis,
            alpha   = 0.7,
            levels  = lambda v, n=12: np.linspace(min(v)+.1, max(v), n),
            norm    = lambda v=None: matplotlib.colors.Normalize(vmin=20, vmax=40),
            title   = 'salinity (g/kg salt in water)'),
        waveheight=dict(
            cm      = plt.cm.BuPu,
            alpha   = 0.85,
            levels  = lambda v, n=12: np.linspace(min(v)+.1, max(v), n),
            norm    = lambda v=None: matplotlib.colors.Normalize(vmin=0, vmax=15),
            title   = 'wave height (metres)')
    )


def plot2D(var, source, plot_wind=False, save=False, **kwargs): 
    """
    
        import kadlu
        var = 'bathymetry'
        source = 'gebco'
        plot_wind, save = False, False
        kwargs = kadlu.defaults
        kadlu.plot2D(var, source, **kwargs)

    """
    var = var.lower()
    source = source.lower()

    if f'{var}_{source}' not in load_map.keys():
        raise KeyError(f'could not find source for variable. valid vars and '
                       f'sources: {[k.split("_") for k in load_map.keys()]}')

    if 'start' not in kwargs.keys():
        kwargs['start'], kwargs['end'] = datetime.now(), datetime.now()

    data = load_map[f'{var}_{source}'](**kwargs)
    val, lat, lon = data[:3].astype(float)

    # project data onto coordinate space
    extent = ccrs.Mercator().transform_points(
            ccrs.Geodetic(),
            np.array([kwargs['west'], kwargs['east']]), 
            np.array([kwargs['south'], kwargs['north']])
        )[:,:-1]
    projected_lonlat = ccrs.Mercator().transform_points(
            ccrs.Geodetic(),
            lon,
            lat
        )
    plon = projected_lonlat[:,0]
    plat = projected_lonlat[:,1]
    num_lats = 1000
    num_lons = 1000
    lons = np.linspace(start=min(plon), stop=max(plon), num=num_lons)
    lats = np.linspace(start=min(plat), stop=max(plat), num=num_lats)
    data = griddata(points=(plon, plat), values=val, xi=(lons[None,:],lats[:,None]), method='linear')
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    fg = (.92, .92, .92, 1)
    fname = f'{var}_{kwargs["start"].date().isoformat()}.png'
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, 
            title=config[var]['title']+f'\n{kwargs["start"].date().isoformat()}',
            projection=ccrs.Mercator(), 
            facecolor=config[var]['cm'](256), 
            frameon=True
        )
    ax.contourf(lons, lats, data,
                transform=ccrs.Mercator(),
                levels=config[var]['levels'](val),
                cmap=config[var]['cm'], 
                alpha=config[var]['alpha'],
                zorder=8
            )
    ax.contour(lons, lats, data,
                transform=ccrs.Mercator(),
                levels=config[var]['levels'](val),
                cmap=config[var]['cm'],
                alpha=1,
                linewidths=2,
                zorder=9
            )

    if plot_wind is not False:
        if plot_wind.lower() == 'era5': 
            #windfcnU, windfcnV = (Era5().load_wind_u, Era5().load_wind_v)
            windfcnU, windfcnV = load_map['wind_u_era5'], load_map['wind_v_era5']
        elif plot_wind.lower() == 'wwiii': 
            #windfcnU, windfcnV = (Wwiii().load_wind_u, Wwiii().load_wind_v)
            windfcnU, windfcnV = load_map['wind_u_wwiii'], load_map['wind_v_wwiii']
        else: 
            raise ValueError('invalid wind source. must be \'era5\' or \'wwiii\'')

        uval, ulat, ulon, utime = windfcnU(**kwargs)
        vval, vlat, vlon, vtime = windfcnV(**kwargs)
        assert(len(vval) == len(uval))  # this can be fixed with an SQL JOIN in load module
        if len(np.unique(ulat)) == 1 or len(np.unique(ulon)) == 1:
            raise RuntimeError(f'Not enough datapoints to plot windspeeds in region {fmt_coords(kwargs)}')

        ax.quiver(ulon, ulat, uval, vval, transform=ccrs.PlateCarree(), 
                regrid_shape=20, zorder=10)

    ax.add_feature(coast, facecolor=fg, edgecolor=(0,0,0,1), zorder=11)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--',
            zorder=12)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER

    # this appears to be a known bug
    # https://github.com/SciTools/cartopy/issues/1332
    for tick in ax.get_xticklabels(): 
        tick.set_rotation(45)

    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    ax.tick_params(axis='x', rotation=45)
    vnorm = val if var == 'bathy' or var == 'bathymetry' else None
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=config[var]['norm'](vnorm),
                cmap=config[var]['cm']))

    if save is not False:
        if not os.path.isdir(f'{storage_cfg()}figures'): 
            os.mkdir(f'{storage_cfg()}figures')
        logging.info(f'saving figure to {storage_cfg()}figures/{fname if save is True else save}')
        plt.savefig(f'{storage_cfg()}figures/{fname if save is True else save}', 
                bbox_inches='tight', dpi=200, figsize=(12,8), optimize=True)
        plt.close()
    else: 
        plt.show()

    return


def animate(var, source, kwargs, step=timedelta(hours=12), fps=30, plot_wind=False, debug=False):
    """
    args:
        var: string
            view the list of available sources and variables with ``print(kadlu.source_map)``
        source: string
            view the list of available sources and variables with ``print(kadlu.source_map)``
        kwargs: dictionary
            dict containing boundary coordinates. example:
            ``kwargs=dict(south=44.25, west=-64.5, north=44.70, east=-63.33, top=0, bottom=5000, start=datetime(2015, 3, 1), end=datetime(2015, 3, 1, 12))``
        step: timedelta
            temporal resolution of individual video frames
        fps: int
            frames per second
        plot_wind: string
            if plot_wind is set to one of the available wind sources in kadlu.source_map, 
            wind direction markers will be overlayed on the map

    the resulting animation will be stored in the 'animated' subdirectory within the kadlu_data folder
    """

    # prepare folder and check for existing frames
    dirname = storage_cfg() + 'figures/'
    if not os.path.isdir(dirname): os.mkdir(dirname)
    png = lambda f: f if '.png' in f else None
    old = map(png, list(os.walk(dirname))[0][2])
    _rm = [os.remove(f'{dirname}{x}') for x in old if debug and x is not None]

    # generate image frames
    qry = kwargs.copy()
    cur = datetime(kwargs['start'].year, kwargs['start'].month, kwargs['start'].day)
    while cur <= kwargs['end']:
        qry['start'] = cur
        qry['end'] = cur + step
        fname = f'{var}_{cur.isoformat()}.png'
        if not os.path.isfile(f'{dirname}/{fname}'): 
            plot2D(var, source, plot_wind=plot_wind, save=fname, **qry)
        cur += step

    # filename and path for output
    fname = (f'{var}_{kwargs["start"].date().isoformat()}'
             f'_{kwargs["end"].date().isoformat()}.mp4')
    savedir = f'{storage_cfg()}animated{os.path.sep}'
    if not os.path.isdir(savedir): os.mkdir(savedir)

    # aggregate frames within query range and append to mp4 file
    logging.info(f'animating {fname}...')
    fmt = f'{var}_%Y-%m-%dT%H:%M:%S.png'
    frames = sorted([f'{dirname}{i}' for i in 
            map(png, list(os.walk(f'{dirname}'))[0][2]) if i is not None
            and datetime.strptime(i, fmt) >= kwargs['start']
            and datetime.strptime(i, fmt) <= kwargs['end']])
    with imageio.get_writer(f'{savedir}{fname}', mode='I', macro_block_size=4,
            format='FFMPEG', fps=fps) as w:
        list(map(w.append_data, map(imageio.imread, frames)))

    logging.info(f'saved animation to {savedir}{fname}')
    return 
'''


def plot_transm_loss_horiz(transm_loss, radial_axis, azimuthal_axis):
    """ Plot the transmission loss on a horizontal plane in polar coordinates.

        Args:
            transm_loss: numpy.array
                Transmission loss, has shape (nq,nr).
            radial_axis: numpy.array
                Radial axis, has shape (nr)
            azimuth_axis: numpy.array
                Azimuthal axis, has shape (nq)

        Returns:
            fig: matplotlib.figure.Figure
                A figure object.
    """
    # "complete the circle"
    azimuthal_axis = np.concatenate([azimuthal_axis, [np.pi]])
    transm_loss = np.concatenate([transm_loss, transm_loss[0:1,:]], axis=0)
    # convert to x,y meshgrid
    r, q = np.meshgrid(radial_axis, azimuthal_axis)
    x = r * np.cos(q) / 1e3
    y = r * np.sin(q) / 1e3
    # contour plot
    fig, ax = plt.subplots()
    img = ax.contourf(x, y, transm_loss, 100, cmap=matplotlib.colormaps.get_cmap('viridis_r'))
    # labels
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    plt.title('Transmission loss')
    fig.colorbar(img, ax=ax, format='%2.0f dB')# colobar
    return fig


def plot_transm_loss_vert(transm_loss, vertical_axis, radial_axis, bathy_func=None, ssp_func=None):
    """ Plot the transmission loss on a vertical plane in carthesian coordinates.

        If a bathymetry interpolation function is provided, the seafloor will be 
        drawn superimposed on the transmission loss plot.

        Args:
            transm_loss: numpy.array
                Transmission loss, has shape (nz,nr).
            vertical_axis: numpy.array
                Vertical axis, has shape (nz)
             radial_axis: numpy.array
                Radial axis, has shape (nr)
            bathy_func: function
                Bathymetry interpolation function in radial variable, r
            ssp_func: function
                Sound speed interpolation function in radial variable, z

        Returnedme :
            fig: matplotlib.figure.Figure
                A figure object.
    """
    logger = logging.getLogger("kadlu")

    x, y = np.meshgrid(radial_axis, vertical_axis)

    # min and max transmission loss (excluding sea surface bin)
    tl_min = np.min(transm_loss[1:,:])
    tl_max = np.max(transm_loss[1:,:])

    # contour plot
    fig, ax = plt.subplots()
    img = ax.contourf(x/1e3, y, transm_loss, 100, vmin=tl_min, vmax=tl_max, cmap=matplotlib.colormaps.get_cmap('viridis_r'))
    ax.invert_yaxis()

    # labels
    ax.set_xlabel('r (km)')
    ax.set_ylabel('z (m)')
    plt.title('Transmission loss')
    fig.colorbar(img, ax=ax, format='%2.0f dB') # colobar
    
    # superimpose bathymetry
    if bathy_func is not None: 
        r_max = np.max(radial_axis)
        nr = min(10000, int(r_max / 10))
        r = np.linspace(0, r_max, num=nr)
        bathy = bathy_func(r)
    
        logger.debug('Superimposing bathymetry on TL plot') 
        logger.debug(f'Radial coordinates range (m): [{r[0]:.1f}, {r[-1]:.1f}],  bathymetry range (m): [{np.min(bathy):.1f}, {np.max(bathy):.1f}]')
    
        idx = np.argwhere(bathy <= vertical_axis[-1])
    
        logger.debug(f'Displaying {len(idx)} data points (out of {len(bathy)}) with bathy <= {vertical_axis[-1]:.1f} m')
    
        ax.plot(r[idx]/1e3, bathy[idx], 'w')
    
    # superimpose ssp
    if ssp_func is not None: 
        z_max = np.max(np.max(vertical_axis))
        nz = min(1000, int(z_max))
        z = np.linspace(0, z_max, num=nz)
        ssp = ssp_func(z)
        axt = ax.twiny()
        dc = np.max(ssp) - np.min(ssp)
        axt.set_xlim(np.min(ssp)-0.2*dc, np.max(ssp)+0.2*dc)        
        axt.set_xlabel('c (m/s)')
        axt.plot(ssp, z, 'w', linestyle=':')

    return fig

