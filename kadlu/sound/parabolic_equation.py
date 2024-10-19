""" This module contains methods for numerically solving the parabolic wave
    equation of Thomson and Chapman.
"""
import logging
import numpy as np
from numpy.lib import scimath
from kadlu.utils import toarray, deg2rad
from tqdm import tqdm
from kadlu.plot_util import plot_transm_loss_horiz, plot_transm_loss_vert


def thomson_starter(k0, kz, dz, zs, theta1):
    """ Compute Thomson starter field :math:`\psi (0, k_z)` as defined in
        Jensen Sec. 6.4.2.3.

        Args:
            k0: float
                Reference wavenumber in inverse meters
            kz: numpy.array
                Vertical wavenumber in inverse meters
            dz: float
                Vertical bin size in meters
            zs: array-like
                Source depth(s) in meters
            theta1: float
                Half-beamwidth in radians

        Returns:
            psi: numpy array
                Starter field, :math:`\psi (0, k_z)`.
                Has shape (len(zs), len(kz), 1)

        Example:
            >>> import numpy as np
            >>> from kadlu.sound.parabolic_equation import Grid, thomson_starter
            >>>
            >>> # Create a grid with depth of 500 m, range of 1 km,
            >>> # and grid spacings of 100 meters and 10 degrees
            >>> grid = Grid(100., 1000., 100., 500., 10.*np.pi/180.)
            >>>
            >>> # compute Thomson starter field with an aperture of 86 degrees,
            >>> # for a reference wavenumber of 0.04 m^-1 and a source depth of 9 meters,
            >>> psi = thomson_starter(k0=0.04, kz=grid.kz, dz=grid.dz, zs=9, theta1=86*np.pi/180.)
            >>> psi = np.round(psi, 4) # round to 4 decimals
            >>> print(psi[0])
            [[ 0.    +0.j    ]
             [ 0.0101-0.0101j]
             [ 0.0205-0.0205j]
             [ 0.0319-0.0319j]
             [ 0.0451-0.0451j]
             [ 0.    +0.j    ]
             [-0.0451+0.0451j]
             [-0.0319+0.0319j]
             [-0.0205+0.0205j]
             [-0.0101+0.0101j]]
    """
    nz = len(kz)
    kz = kz[np.newaxis, :]
    zs = toarray(zs)[:, np.newaxis]

    # compute [scimath.sqrt(scimath.sqrt(k0**2 - kz**2))]^-1
    idx = np.nonzero(np.abs(kz) >= np.abs(k0))
    a = np.empty(kz.shape)
    a[idx] = 0
    idx = np.nonzero(np.abs(kz) < np.abs(k0))

    a[idx] = 1. / scimath.sqrt(scimath.sqrt(k0**2 - kz[idx]**2))

    psi = np.exp(-1j * np.pi / 4.) * scimath.sqrt(8 * np.pi) * np.sin(
        kz * zs) * a
    psi = psi / dz  # normalize
    psi[:, int(nz / 2)] = 0  # set to 0 at the bottom of the ocean

    # taper the spectrum to obtain desired angle using Turkey window
    kcut1 = k0 * np.sin(theta1)
    kcut0 = k0 * np.sin(theta1 - 1.5 * deg2rad)
    W = 0.5 * (1 + np.cos(np.pi / (kcut1 - kcut0) * (np.abs(kz) - kcut0)))
    W[np.abs(kz) >= kcut1] = 0
    W[np.abs(kz) <= kcut0] = 1
    psi = psi * W
    psi = psi[:, :, np.newaxis]

    return psi


class Grid():
    """ Regular grid in cylindrical coordinates r,q,z, where

            * r: radial coordinate (distance) in meters
            * z: vertical coordinate (depth) in meters
            * q: azimuthal coordinate (angle) in radians

        Note: The sea surface is at z = 0. We use positive z values below
        the sea surface and negative values above.

        Args:
            dr: float
                Radial grid spacing in meters
            r_max: float
                Radial range in meters
            dz: float
                Vertical grid spacing in meters
            z_max: float
                Vertical range in meters
            dq: float
                Angular grid spacing in radians
            q_max: float
                Angular range in radians
            q: float, list or numpy.array
                Angular coordinates in radians. If specified, dq and q_max are ignored.

        Attrs:
            dr: float
                Radial grid spacing in meters
            r: numpy.array
                Radial coordinates in meters
            nr: int
                Number of radial grid points
            dz: float
                Vertical grid spacing in meters
            z: numpy.array
                Vertical coordinates in meters
            nz: int
                Number of vertical grid points
            dq: float
                Angular grid spacing in radians
            q: numpy.array
                Angular coordinates in radians
            nq: int
                Number of angular grid points
            kz: numpy.array
                Vertical wavenumber coordinates in inverse meters
            below: numpy.array
                Indices of z axis corresponding to values below (z>=0) the surface
            q_qz, z_qz: numpy.array
                Coordinates of q-z meshgrid
            below_qz: numpy.array
                Indices of q-z meshgrid corresponding to values below (z>=0) the surface
    """

    def __init__(self, dr, r_max, dz, z_max, dq=None, q_max=2 * np.pi, q=None):
        assert dq is not None or q is not None, "either dq or q must be specified"

        self.r, self.dr = self._radial_coords(dr, r_max)
        self.nr = len(self.r)

        self.dq = None
        if q is None: self.q, self.dq = self._azimuth_coords(dq, q_max)
        elif isinstance(q, list): self.q = np.array(q)
        elif isinstance(q, (int, float)): self.q = np.array([q])
        self.nq = len(self.q)

        self.z, self.dz, self.below, self._above, self._mirror = self._vertical_coords(
            dz, z_max)
        self.nz = len(self.z)

        self.kz = self.z * 2 * np.pi / (len(self.z) * self.dz**2)  #wave number

        self.q_qz, self.z_qz = np.meshgrid(self.q, self.z)  #q-z meshgrid
        self.below_qz = np.nonzero(self.z_qz >= 0)

    def mirror(self, x, z_axis=0):
        """ Replace values above sea surface with mirrored values from below surface.

            Args:
                x: numpy.array
                    Array containing the data to be mirrored
                z_axis: int
                    z axis of array

            Returns:
                x: numpy.array
                    Mirrored array
        """
        x = np.swapaxes(x, 0, z_axis)
        x[self._above] = x[self._mirror]
        x = np.swapaxes(x, 0, z_axis)
        return x

    def _radial_coords(self, dr, r_max):
        """ Compute radial coordinates"""
        N = int(round(r_max / dr)) + 1
        r = np.arange(N, dtype=float) * dr
        return r, dr

    def _azimuth_coords(self, dq, q_max):
        """ Compute azimuthal coordinates"""
        N = int(np.ceil(q_max / dq))
        N += N % 2  # ensure even number of angular bins
        q_pos = np.arange(start=0, stop=N / 2, step=1, dtype=float)
        q_neg = np.arange(start=-N / 2, stop=0, step=1, dtype=float)
        q = np.concatenate((q_pos, q_neg)) * dq
        return q, dq

    def _vertical_coords(self, dz, z_max):
        """ Compute vertical coordinates"""
        N = int(round(z_max / dz) * 2)  # ensure even number of vertical bins
        z_pos = np.arange(start=0, stop=N / 2 + 1, step=1, dtype=float)
        z_neg = np.arange(start=-N / 2 + 1, stop=0, step=1, dtype=float)
        z = np.concatenate((z_pos, z_neg)) * dz
        idx_pos = np.nonzero(z >= 0)[0]
        idx_neg = np.nonzero(z < 0)[0]
        idx_mirror = idx_pos[-2:0:-1]
        return z, dz, idx_pos, idx_neg, idx_mirror


def prop_defr(x, k0, kz, nq):
    """ Compute the defractive propagation matrix, :math:`U_D`.

        Args:
            x: float
                Step size in meters
            k0: float
                Reference wavenumber in inverse meters
            kz: array-like
                Vertical wavenumber in inverse meters
            nq: int
                Number of angular bins

        Returns:
            U: numpy.array
                Defractive propagation matrix, has shape (1,nz,nq)
    """
    U = np.exp(1j * (scimath.sqrt(k0**2 - kz**2) - k0) * x)
    U = U[:, np.newaxis] * np.ones(shape=(1, nq))
    U = U[np.newaxis, :, :]
    return U


def prop_refr(x, k0, n):
    """ Compute the refractive propagation matrix, :math:`U_R`.

        Args:
            x: float
                Step size in meters
            k0: float
                Reference wavenumber in inverse meters
            n: numpy.array
                Index of refraction, has shape (nz,nq)

        Returns:
            U: numpy.array
                Refractive propagation matrix, has shape (1,nz,nq)
    """
    U = np.exp(1j * x * k0 * (n - 1))
    U = U[np.newaxis, :, :]
    return U


def artificial_absorp(z, z_max, D, alpha):
    """ Compute complex term, to be added to the square of the index of
        refraction in order to simulate the effect of an artificial
        absorption layer at the bottom of the physical domain.

        Args:
            z: numpy.array
                Depth values in meters
            z_max: float
                Maximum depth in meters
            D: float
                Characteristic width of the absorption layer in meters
            alpha: float
                Attenuation coefficient

        Returns:
            : numpy.array
                Absorption term, to be added to the square of the physical
                index of refraction.
    """
    return 1j * alpha * np.exp(-(np.abs(z) - z_max)**2 / D**2)


def complex_sound_speed(c, alpha):
    """ Computes complex sound speed representing the effect
        of volume attenuation.

        Args:
            c: array-like
                Sound speed in m/s
            alpha: float
                Attenuation coefficient in dB/lambda

        Returns:
            ci: array-like
                Complex sound speed term
    """
    if alpha == 0: return 0

    beta = alpha / (40. * np.pi * np.log10(np.e) * c)

    if isinstance(c, (int, float)):
        ci = np.roots([beta, -1,
                       beta * c**2])  # roots of polynomial p[0]*x^n+...p[n]
        ci = ci[np.imag(ci) == 0]
        ci = ci[np.logical_and(ci >= 0, ci < c)]
        ci = -1j * ci[0]

    else:
        ci = -1j * beta * c**2

    return ci


def index_refr_sq(c0, c, alpha=0):
    """ Compute index of refraction.

        If alpha > 0, the index of refraction receives a complex term,
        which represents the effect of volume attenuation.

        Args:
            c0: float
                Reference sound speed in m/s
            c: array-like
                Sound speed in m/s
            alpha: float
                Attenuation coefficient in dB/lambda

        Returns:
            n2: array-like
                Index of refraction squared
    """
    c += complex_sound_speed(c, alpha)
    n2 = (c0 / c)**2
    return n2


def smooth_index_refr(z, zb, n2, n2b, L):
    """ Smoothen the index of refraction from the water column to the
        bottom layer.

        Args:
            z: numpy.array
                Depth values in meters
            zb: numpy.array
                Seafloor depth in meters
            n2: numpy.array
                Index of refraction squared in water
            n2b: numpy.array
                Index of refraction squared in bottom
            L: float
                Characteristic smoothing length in meters

        Returns:
            : numpy.array
                Index of refraction squared, with smooth transition from water to bottom
    """
    x = (np.abs(z) - zb) / L
    return n2 + 0.5 * (n2b - n2) * (1 + np.tanh(x))


def smooth_density(z, zb, dzb2, L, r, rb):
    """ Smoothen the density transition from the water column to the
        bottom layer and compute first and second derivatives.

        Args:
            z: numpy.array
                Depth values in meters
            zb: numpy.array
                Seafloor depth in meters
            dzb2: numpy.array
                Gradient of the seafloor depth squared
            L: float
                Characteristic smoothing length in meters
            r: numpy.array
                Water density in g/cm^3
            rb: numpy.array
                Bottom density in g/cm^3

        Returns:
            rs: numpy.array
                Density in g/cm^3, with smooth transition from water to bottom
            dr2: numpy.array
                Square of density gradient in (g/cm^3/m)^2
            d2r: numpy.array
                Divergence of density gradient (g/cm^3/m^2)
    """
    x = (np.abs(z) - zb) / L
    tanh = np.tanh(x)
    sech2 = np.empty(x.shape)
    idx = np.nonzero(np.abs(x) < 100)
    sech2[idx] = 1. / np.cosh(x[idx])**2
    idx = np.nonzero(np.abs(x) >= 100)
    sech2[idx] = 0
    rs = r + 0.5 * (rb - r) * (
        1 + tanh)  # density (rho), smooth transition from water to bottom
    dr2 = 1. / (4 * L**2) * (rb - r)**2 * sech2**2 * (1 + dzb2
                                                      )  # gradient squared
    d2r = -1. / L**2 * (rb - r) * sech2 * tanh * (1 + dzb2
                                                  )  # divergence of gradient
    return rs, dr2, d2r


def eff_index_refr_sq(z,
                      zb,
                      dzb2,
                      k0,
                      L,
                      n2,
                      n2b,
                      r,
                      rb,
                      return_density=False):
    """ Compute the effective index of refraction squared.

        Args:
            z: numpy.array
                Depth values in meters
            zb: numpy.array
                Seafloor depth in meters
            dzb2: numpy.array
                Gradient of the seafloor depth squared
            k0: float
                Reference wavenumber in inverse meters
            L: float
                Characteristic smoothing length in meters
            n2: numpy.array
                Water refractive index squared
            n2b: numpy.array
                Bottom refractive index squared
            r: numpy.array
                Water density in g/cm^3
            rb: numpy.array
                Bottom density in g/cm^3
            return_density: bool
                Return the smoothened density array. Default is False.

        Returns:
            n2e: numpy.array
                Effective index of refraction squared
            rs: numpy.array
                Smoothened density. Only returned if return_density is True.

    """
    n2s = smooth_index_refr(z, zb, n2, n2b, np.finfo(float).eps)
    rs, dr2, d2r = smooth_density(z, zb, dzb2, L, r, rb)
    n2e = n2s + 1 / (2 * k0**2) * (1 / rs * d2r - 3 / (2 * rs**2) * dr2)
    if return_density: return n2e, rs
    else: return n2e


class TransmissionLoss():
    """ Compute the transmission loss by solving the parabolic wave
        equation.

        The seafloor acoustic properties are specified in a dictionary
        with the following keys,

            * sound_speed (bottom sound speed in m/s)
            * density (bottom density in g/cm3)
            * attenuation (bottom attenuation in dB/lambda)

        Keyword arguments can be used to specify the computational
        grid :class:`kadlu.sound.parabolic_equation.Grid`.

        Args:
            freq: float
                Frequency in Hz
            bathy_func: function
                Bathymetry interpolation function in variables x,y
            bathy_deriv_func: function
                Bathymetry derivative interpolation function in variables x,y
            sound_speed_func: function
                Sound speed interpolation function in variables x,y,z
            bottom: dict
                Seafloor acoutic properties. Must contain the keys 'sound_speed',
                'density', and 'attenuation'.
            propagation_range: float
                Propagation range in km. Default is 50 km.
            angular_bin: float
                Angular bin size in degrees. Default is 10 degrees.
            c0: float
                Reference sound speed in m/s
            source_depth: array-like
                Source depths in meters.

        Attributes:

        Example:
    """

    def __init__(self,
                 freq,
                 bathy_func,
                 bathy_deriv_func,
                 sound_speed_func,
                 bottom,
                 propagation_range=50,
                 angular_bin=10,
                 c0=1500,
                 **kwargs):

        self.logger = logging.getLogger("kadlu")

        self.name = self.__class__.__name__

        debug_msg = f"[{self.name}] Initializing {self.name} with range = {propagation_range:.3f}km"
        self.logger.debug(debug_msg)

        self.c0 = c0
        self.k0 = 2 * np.pi / c0 * freq
        self.water_density = 1.0

        r_max = 1e3 * propagation_range
        dq = angular_bin * deg2rad

        #depth of physical domain
        max_depth = self._max_depth(bathy_func, r_max)

        if max_depth > 0.01:
            debug_msg = f"[{self.name}] Max depth in computational domain: {max_depth:.1f}m"
            self.logger.debug(debug_msg)

        else:
            err_msg = f"Max depth in computational domain should be stricly positive (i.e. underwater) but was found to be {max_depth:.1f}m"
            raise RuntimeError(err_msg)

        H = max_depth + 3 * c0 / freq
        
        z_max = 4. / 3 * H

        # default grid
        grid_kwargs = {
            'dz': c0 / freq / 2.,
            'dr': c0 / freq,
            'dq': dq,
            'q': None,
            'r_max': r_max,
            'z_max': z_max
        }

        # replace defaults with input args, if any
        for key in grid_kwargs:
            if key in kwargs.keys(): grid_kwargs[key] = kwargs[key]

        self.logger.debug(f"[{self.name}] Initializing computational grid with")
        for k,v in grid_kwargs.items():
            self.logger.debug(f"[{self.name}] - {k}: {v}")

        self.grid = Grid(**grid_kwargs)
        self.bottom = bottom

        self.costheta = np.cos(self.grid.q)
        self.sintheta = np.sin(self.grid.q)

        # bottom refractive index
        self.n2b = index_refr_sq(c0=c0, c=bottom['sound_speed'], alpha=bottom['attenuation'])

        # artificial bottom absorption layer
        alpha = 1. / np.log10(np.e) / np.pi
        z_max = np.max(self.grid.z)
        D = (z_max - H) / 3
        self.absorp = artificial_absorp(z=self.grid.z_qz,
                                        z_max=z_max,
                                        D=D,
                                        alpha=alpha)

        # interpolation functions
        self._bathy = bathy_func
        self._bathy_deriv = bathy_deriv_func
        self._sound_speed = sound_speed_func

        self._do_vertical = False  # compute transmission loss on vertical plane

        # source depth
        self._source_depth = kwargs[
            'source_depth'] if 'source_depth' in kwargs.keys() else None

    def calc(self,
             source_depth=None,
             rec_depth=[.1],
             vertical=False,
             aperture=86,
             progress_bar=True,
             nz_max=250,
             nr_max=250,
             size_limit=True,
             return_field=False):
        """ Calculate the transmission loss in the horizontal plane at
            the specified depth(s).

            Args:
                source_depth: array-like
                    Source depths in meters
                rec_depth: array-like
                    Receiver depths in meters
                vertical: array-like
                    Compute the transmission loss in the vertical plane.
                    Note: This will slow down the computation.
                aperture: float
                    Half-beamwidth of the starter field in degrees.
                progress_bar: bool
                    Show progress bar. Default is True
                nz_max: int
                    Maximum number of vertical bins for output arrays
                nr_max: int
                    Maximum number of radial bins for output arrays
                size_limit: bool
                    Limit of the size of the output arrays to the dimensions
                    specified in nz_max and nr_max. If False, nz_max and nr_max
                    have no effect
                return_field: bool
                    Instead of returning the transmission loss, TL, return
                    the complex pressure field, f, where TL = 20 * np.log10(np.abs(f)).
                    Default is False.

            Returns:
                tl_h: numpy.array
                    Transmission loss in dB in the horizontal plane; has shape
                    (len(source_depth), len(rec_depth), len(q), len(r)).
                ax_h: dict
                    Axes of the horizontal transmission loss array, (source_depth, rec_depth, q, r).
                    q and r are the azimuthal and radial coordinate axes.
                tl_v: numpy.array
                    Transmission loss in dB in the vertical plane; has shape
                    (len(source_depth), len(z), len(r), len(q)).
                    Only returned if vertical is True.
                ax_v: dict
                    Axes of the vertical transmission loss array, (source_depth, z, r, q).
                    z and r are the vertical and radial coordinate axes.
                    q is the azimuthal axes.
                    Only returned if vertical is True.
        """
        # source depth
        if source_depth is None: source_depth = self._source_depth
        assert source_depth is not None, 'source depth must be specified'

        self._do_vertical = vertical
        source_depth = toarray(source_depth)
        rec_depth = toarray(rec_depth)
        self._init_output(num_sources=len(source_depth),
                          rec_depth=rec_depth,
                          nz_max=nz_max,
                          nr_max=nr_max,
                          size_limit=size_limit)
        psi = self._solve_pe(source_depth=source_depth,
                             rec_depth=rec_depth,
                             aperture=aperture,
                             progress_bar=progress_bar)

        # transmission loss, horizontal plane
        field_h = np.fft.fftshift(self._field_horiz[:, :, :, 1:],
                                  axes=2)  #re-order q axis
        tl_h = -20 * np.log10(np.abs(field_h))
        q = self.grid.q
        if len(q) > 1: q = np.squeeze(q)
        q = np.fft.fftshift(q)
        ax_h = {
            'source_depth': source_depth,
            'receiver_depth': rec_depth,
            'azimuthal_axis': q,
            'radial_axis': self._field_r_ax[1:]
        }
        res_h = field_h if return_field else tl_h
        self.tl_h, self.ax_h = tl_h, ax_h

        if not self._do_vertical:
            return res_h, ax_h

        else:  # transmission loss, vertical plane
            field_v = np.fft.fftshift(self._field_vert[:, :, :, :],
                                      axes=3)  #re-order q axis
            tl_v = -20 * np.ma.log10(
                np.abs(field_v))  # OBS: this computation is rather slow
            ax_v = {
                'source_depth': source_depth,
                'vertical_axis': self._field_z_ax,
                'radial_axis': self._field_r_ax,
                'azimuthal_axis': q
            }
            self.tl_v, self.ax_v = tl_v, ax_v
            res_v = field_v if return_field else tl_v
            return res_h, ax_h, res_v, ax_v

    def plot_horiz(self, source_depth_idx=0, rec_depth_idx=0):
        """ Plot the transmission loss on a horizontal plane in polar coordinates.

            Args:
                source_depth_idx: int
                    Source depth index
                rec_depth_idx: int
                    Receive depth index

            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        tl = self.tl_h[source_depth_idx, rec_depth_idx]
        r = self.ax_h['radial_axis']
        q = self.ax_h['azimuthal_axis']
        fig = plot_transm_loss_horiz(tl, r, q)
        return fig

    def plot_vert(self,
                  angle=0,
                  source_depth_idx=0,
                  show_bathy=True,
                  max_depth=None,
                  show_ssp=False):
        """ Plot the transmission loss on a vertical plane in carthesian coordinates.

            Args:
                angle: float
                    Angle in degrees
                source_depth_idx: int
                    Source depth index
                show_bathy: bool
                    Superimpose bathymetry on plot
                show_ssp: bool
                    Superimpose sound speed profile on plot

            Returns:
                fig: matplotlib.figure.Figure
                    A figure object.
        """
        z = self.ax_v['vertical_axis']
        r = self.ax_v['radial_axis']
        q = self.ax_v['azimuthal_axis']

        # truncate z axis
        if max_depth is not None: z = z[z < max_depth]

        # find nearest angular bin
        if angle <= 180: q0 = angle * np.pi / 180
        else: q0 = (angle - 360) * np.pi / 180
        angle_idx = np.abs(q - q0).argmin()

        tl = self.tl_v[source_depth_idx, 1:len(z), :, angle_idx]
        z = z[:-1]  #drop last

        # bathymetry
        if show_bathy:
            def bathy_r(r):
                angle_rad = angle * deg2rad
                x = np.cos(angle_rad) * r
                y = np.sin(angle_rad) * r
                return self._bathy(x=x, y=y)
        else:
            bathy_r = None

        # sound speed profile
        if show_ssp:
            x = np.cos(angle * deg2rad) * np.linspace(0, np.max(r), 1000)
            y = np.sin(angle * deg2rad) * np.linspace(0, np.max(r), 1000)
            idx = np.argmax(self._bathy(x=x, y=y))
            x0, y0 = x[idx], y[idx]
            def ssp_z(z):
                return self._sound_speed(x=x0, y=y0, z=z, grid=True)
        else:
            ssp_z = None

        fig = plot_transm_loss_vert(tl, z, r, bathy_r, ssp_z)
        return fig

    def _solve_pe(self,
                  source_depth,
                  rec_depth,
                  aperture,
                  progress_bar,
                  return_field=False):
        """ Solve the parabolic wave equation of Thomson and Chapman by means
            of a split-step Fourier algorithm.

            The displacement field is computed on one or several horizontal planes
            at the specified receiver depth(s).

            Optionally, the field may also be computed on a vertical plane intersecting
            the source position.

            Args:
                source_depth: array-like
                    Source depths in meters
                rec_depth: array-like
                    Receiver depths in meters
                aperture: float
                    Half-beamwidth of the starter field in degrees.
                progress_bar: bool
                    Show progress bar. Default is True
                return_field: bool
                    Return the displacement field :math:`\psi (r, k_z)` after propagation to
                    :math:`r=r_{max}`. Primarily for diagnostics/debugging purposes.
                    Default is False.

            Returns:
                psi: numpy.array
                    Displacement field :math:`\psi (r, k_z)` at :math:`r=r_{max}`.
                    Only returned if return_field is True.
        """
        dr = self.grid.dr
        dz = self.grid.dz
        k0 = self.k0
        kz = self.grid.kz
        nq = self.grid.nq
        nr = self.grid.nr
        rec_depth = toarray(rec_depth)

        UD = prop_defr(x=dr / 2, k0=k0, kz=kz,
                       nq=nq)  #diffractive propagation matrix

        psi = thomson_starter(k0=k0,
                              kz=kz,
                              dz=dz,
                              zs=source_depth,
                              theta1=aperture * deg2rad)  # starter field
        psi = psi * np.ones((1, 1, nq))

        self._save_output(step_no=0,
                          r=0,
                          psi=psi,
                          sqrt_rho=0,
                          rec_depth=rec_depth)  #save output

        r = 0
        for i in tqdm(range(nr - 1), disable=not progress_bar):  # PE marching
            psi = UD * psi  # diffractive propagation, half-step
            n, sqrt_rho = self._update_env(
                r + dr / 2)  # update acoustic environment
            UR = prop_refr(x=dr, k0=k0, n=n)  # refractive propagation
            psi = np.fft.fft(UR * np.fft.ifft(psi, axis=1),
                             axis=1)  # refractive propagation, full step
            psi = UD * psi  # diffractive propagation, half-step
            r += dr  # increment distance
            self._save_output(step_no=i + 1,
                              r=r,
                              psi=psi,
                              sqrt_rho=sqrt_rho,
                              rec_depth=rec_depth)  # collect output

        if return_field: return psi

    def _max_depth(self, bathy, r_max, return_xy=False):
        """ Find the maximum depth in the computational domain.

            Args:
                bathy:
                    Bathymetry interpolation function in x,y variables
                r_max: float
                    Range in meters
                return_xy: bool
                    Return also the coordinates of the location of maximum depth

            Returns:
                b0: float
                    Maximum depth in meters
                x0,y0: float,float
                    x,y coordinates of location of max depth. Only return if return_xy is True.
        """
        x = np.linspace(-r_max, r_max, 2000)
        b = bathy(x, x, grid=True)
        b0 = np.max(b)
        if return_xy:
            idx = np.unravel_index(np.argmax(b, axis=None), b.shape)
            x0 = x[idx[0]]
            y0 = x[idx[1]]
            return b0, x0, y0
        else:
            return b0

    def _bathy_grad_sq(self, r):
        """ Compute seafloor gradient squared at a specified distance
            from the source in the radial direction.

            Args:
                r: float
                    Radial coordinate in meters

            Returns:
                : numpy.array
                    Seafloor gradient squared in all angular bin
        """
        x = self.costheta * r
        y = self.sintheta * r
        dzdx = self._bathy_deriv(x=x, y=y, axis='x')
        dzdy = self._bathy_deriv(x=x, y=y, axis='y')
        return (self.costheta * dzdx)**2 + (self.sintheta * dzdy)**2

    def _update_bathy(self, r):
        """ Update the bathymetry to reflect conditions at a
            specified distance from the source.

            Args:
                r: float
                    Radial coordinate in meters

            Returns:
                zb: numpy.array
                    Seafloor depth in meters
                dzb2: numpy.array
                    Seafloor gradient squared
        """
        x = self.costheta * r
        y = self.sintheta * r
        zb = self._bathy(x=x, y=y)  #bathymetry

        # temp fix to ensure that bathy returns array even when x and y have length 1
        if np.ndim(zb) == 0: zb = np.array([zb])

        dzb2 = self._bathy_grad_sq(r=r)  #gradient squared
        nz = self.grid.nz
        zb = np.ones((nz, 1)) * zb[np.newaxis, :]
        dzb2 = np.ones((nz, 1)) * dzb2[np.newaxis, :]
        return zb, dzb2

    def _update_sound_speed(self, r):
        """ Update the sound speed to reflect conditions at a
            specified distance from the source.

            Args:
                r: float
                    Radial coordinate in meters

            Returns:
                c: numpy.array
                    Sound speed in m/s
        """
        x = self.costheta * r
        y = self.sintheta * r
        z = self.grid.z[self.grid.below]
        x, _ = np.meshgrid(x, z)
        y, z = np.meshgrid(y, z)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        c = self._sound_speed(x=x, y=y, z=z)
        return c

    def _eff_index_refr(self, c, zb, dzb2):
        """ Compute the effective index of refraction,
            including the artificial bottom absorption term.

            Args:
                c: numpy.array
                    Sound speed in m/s
                zb: numpy.array
                    Seafloor depth in meters
                dzb2: numpy.array
                    Seafloor gradient squared

            Returns:
                : numpy.array
                    Effective index of refraction
                : numpy.array
                    Square root of density
        """
        # refractive index squared
        n2 = np.zeros((self.grid.nz, self.grid.nq))
        n2[self.grid.below_qz] = index_refr_sq(self.c0, c)
        n2 = self.grid.mirror(n2)

        # effective refr. index squared
        n2, rho = eff_index_refr_sq(z=self.grid.z_qz,
                                    zb=zb,
                                    dzb2=dzb2,
                                    k0=self.k0,
                                    L=np.pi / self.k0,
                                    n2=n2,
                                    n2b=self.n2b,
                                    r=self.water_density,
                                    rb=self.bottom['density'],
                                    return_density=True)

        # add absorption
        n2 += self.absorp

        return scimath.sqrt(n2), scimath.sqrt(rho)

    def _update_env(self, r):
        """ Update the acoustic environment to reflect
            the conditions at a specified distance from the source.

            Args:
                r: float
                    Radial coordinate in meters

            Returns:
                n: numpy.array
                    Refractive index
                sqrt_rho: numpy.array
                    Square root of density
        """
        zb, dzb2 = self._update_bathy(r)
        c = self._update_sound_speed(r)
        n, sqrt_rho = self._eff_index_refr(c, zb, dzb2)
        return n, sqrt_rho

    def _save_output(self, step_no, r, psi, sqrt_rho, rec_depth):
        """ Post-processe and save output data at specified distance from the source.

            Args:
                step_no: int
                    Step number
                r: float
                    Radial distance from the source in meters
                psi: numpy.array
                    Displacement field at the specified radial distance; has shape (ns,nz,nq).
                sqrt_rho: numpy.array
                    Square root of density
                rec_depth: array-like
                    Receiver depths in meters
        """
        if step_no % self._r_step != 0: return

        bin_no = int(step_no / self._r_step)

        if step_no > 0:
            dz = self.grid.dz
            idx = np.squeeze(np.round(rec_depth / dz).astype(int))
            if np.ndim(idx) == 0: idx = np.array([idx])
            F = np.matmul(self._ifft_kernel, psi)
            G = sqrt_rho[idx][np.newaxis, :, :]
            # inverse fourier transform and multiply by sqrt(density)
            self._field_horiz[:, :, :, bin_no] = F * G * np.exp(
                1j * self.k0 * r) / np.sqrt(r)

        if self._do_vertical:
            # inverse fourier transform and multiply by sqrt(density)
            psi_z = np.fft.ifft(psi, axis=1)
            if r > 0: psi_z *= np.exp(1j * self.k0 * r) / np.sqrt(r) * sqrt_rho
            # only save values below sea surface
            n = int(self.grid.nz / 2)
            self._field_vert[:, :, bin_no, :] = psi_z[:, :n:self._z_step, :]

    def _init_output(self,
                     num_sources,
                     rec_depth,
                     nz_max=250,
                     nr_max=250,
                     size_limit=False):
        """ Initialize output containers and compute inverse fourier transform kernel.

            Args:
                num_sources: int
                    Number of source depths
                rec_depth: array-like
                    Receiver depths in meters
                nz_max: int
                    Maximum number of vertical bins for output arrays
                nr_max: int
                    Maximum number of radial bins for output arrays
                size_limit: bool
                    Limit of the size of the output arrays to the dimensions
                    specified in nz_max and nr_max. If False, nz_max and nr_max
                    have no effect
        """
        nr = self.grid.nr
        nq = self.grid.nq
        nz = self.grid.nz
        ns = num_sources
        nd = len(rec_depth)
        kz = self.grid.kz
        rec_depth = rec_depth[:, np.newaxis]
        # ifft kernel
        self._ifft_kernel = np.exp(
            1j * np.matmul(rec_depth, kz[np.newaxis, :])) / len(kz)
        self._ifft_kernel = self._ifft_kernel[np.newaxis, :, :]
        # output arrays
        if size_limit:
            self._r_step = int(nr / nr_max) + 1
            self._z_step = int(nz / 2 / nz_max) + 1
        else:
            self._r_step = 1
            self._z_step = 1

        self._field_r_ax = self.grid.r[::self._r_step]  #output r axis
        self._field_z_ax = self.grid.z[:int(nz /
                                            2):self._z_step]  #output z axis
        nr = len(self._field_r_ax)
        nz = len(self._field_z_ax)
        self._field_horiz = np.empty(shape=(ns, nd, nq, nr), dtype=complex)
        self._field_vert = np.empty(shape=(ns, nz, nr, nq), dtype=complex)
