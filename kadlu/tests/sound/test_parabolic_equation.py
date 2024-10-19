""" Unit tests for the the 'sound.parabolic_equation' module"""
import pytest
import os
import numpy as np
import kadlu.sound.parabolic_equation as pe 
from kadlu.utils import deg2rad

def test_prop_defr():
    """ Check that defractive propagation matrix is computed correctly """
    dz = 50
    z = np.array([0,50,100,150,200,-150,-100,-50], dtype=float)
    kz = z * 2 * np.pi / (len(z) * dz**2) 
    k0 = 2 * np.pi * 10 / 1500
    UD = pe.prop_defr(x=5, k0=k0, kz=kz, nq=6)
    assert UD.shape == (1,8,6)
    answ = np.array([1.+0.j, 0.9998832 -0.01528329j, 0.99748706-0.07084889j,
                     0.87806263-0.18663797j, 0.77394444-0.16450697j, 0.87806263-0.18663797j,
                     0.99748706-0.07084889j, 0.9998832 -0.01528329j])
    assert np.all(np.abs(UD[0,:,0] - answ) < 1e-7)

def test_thomson_starter():
    """ Check that thomson starter field has expected shape and values"""
    grid = pe.Grid(100., 1000., 100., 500., 10.*np.pi/180.)
    psi = pe.thomson_starter(k0=0.04, kz=grid.kz, dz=grid.dz, zs=9, theta1=86*np.pi/180.)
    assert psi.shape == (1,grid.nz,1)
    psi = np.round(psi[0], 4)
    answ = np.array([[ 0.    +0.j    ],
            [ 0.0101-0.0101j],
            [ 0.0205-0.0205j],
            [ 0.0319-0.0319j],
            [ 0.0451-0.0451j],
            [ 0.    +0.j    ],
            [-0.0451+0.0451j],
            [-0.0319+0.0319j],
            [-0.0205+0.0205j],
            [-0.0101+0.0101j]])
    assert np.all(answ == psi)

def test_thomson_starter_multiple_depths():
    """ Check that thomson starter field has expected shape and values 
        for multiple depth values """
    grid = pe.Grid(100., 1000., 100., 500., 10.*np.pi/180.)
    psi = pe.thomson_starter(k0=0.04, kz=grid.kz, dz=grid.dz, zs=[9, 99], theta1=86*np.pi/180.)
    psi = np.round(psi, 4)
    answ0 = np.array([[0.+0.j],[0.0101-0.0101j],[0.0205-0.0205j],\
        [0.0319-0.0319j],[ 0.0451-0.0451j],[0.+0.j],[-0.0451+0.0451j],\
        [-0.0319+0.0319j],[-0.0205+0.0205j],[-0.0101+0.0101j]])
    assert np.all(psi[0,:,:] == answ0)
    answ1 = np.array([[0.+0.j],[0.1039-0.1039j],[0.1723-0.1723j],\
        [0.1806-0.1806j],[0.1222-0.1222j],[0.+0.j],[-0.1222+0.1222j],\
        [-0.1806+0.1806j],[-0.1723+0.1723j],[-0.1039+0.1039j]])
    assert np.all(psi[1,:,:] == answ1)

def test_create_grid():
    """ Check that the created grid is as expected"""
    g = pe.Grid(dr=100, r_max=1000, dq=45*np.pi/180, dz=100, z_max=500)
    # radial
    assert g.dr == 100
    assert g.nr == 11
    answ = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    assert np.all(g.r == answ)
    # azimuthal
    assert g.dq == 45*np.pi/180
    assert g.nq == 8
    answ = np.array([0, 45, 90, 135, -180, -135, -90, -45]) * deg2rad
    assert np.all(np.abs(g.q - answ) < 1E-9)
    # vertical
    assert g.dz == 100
    assert g.nz == 10
    answ = np.array([0., 100., 200., 300., 400., 500., -400., -300., -200., -100.])
    assert np.all(g.z == answ)
    # below/above indices
    answ = np.array([0, 1, 2, 3, 4, 5])
    assert np.all(g.below == answ)
    answ = np.array([6, 7, 8, 9])
    assert np.all(g._above == answ)
    answ = np.array([4, 3, 2, 1])
    assert np.all(g._mirror == answ)

def test_create_grid_selected_angles():
    """ Check that we can create a grid for selected angular slices"""
    g = pe.Grid(dr=100, r_max=1000, q=[45*deg2rad, 120*deg2rad], dz=100, z_max=500)
    assert g.dq == None
    assert g.nq == 2
    answ = np.array([45*deg2rad, 120*deg2rad])
    assert np.all(np.abs(g.q - answ) < 1E-9)

def test_mirror_grid():
    """ Check that the mirror method returns the expected array"""
    g = pe.Grid(dr=100, r_max=1000, dq=45*deg2rad, dz=100, z_max=500)
    # mirror 1d array
    a = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, np.nan, np.nan, np.nan, np.nan])
    answ = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    a = g.mirror(a)
    assert np.all(a == answ)
    # mirror 2d array
    a = np.array([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0], [np.nan], [np.nan], [np.nan], [np.nan]])
    a = np.swapaxes(a, 0, 1)
    answ = np.array([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    answ = np.swapaxes(answ, 0, 1)
    a = g.mirror(a, z_axis=1)
    assert np.all(a == answ)

def test_eff_index_refr_sq_flat_seafloor():
    """ Check that the effective index of refraction is computed 
        correctly in an environment with uniform seafloor depth """
    z    = np.array([[0,0],[50,50],[100,100],[150,150],[-100,-100],[-50,-50]])
    zb   = np.ones(z.shape) * 90
    dzb2 = np.zeros(z.shape)
    k0   = 0.041887902047863905
    L    = 75
    r    = 1.0
    rb   = 1.5
    n2   = np.ones(z.shape) * 1.0272096420745072
    n2b  = 0.7783506614961636+0.014264417521561993j
    n2_eff = pe.eff_index_refr_sq(z,zb,dzb2,k0,L,n2,n2b,r,rb)
    assert n2_eff.shape == z.shape
    answ = np.array([[1.03298621+0.j, 1.03298621+0.j],
                     [1.03339092+0.j, 1.03339092+0.j],
                     [0.77299581+0.01426442j, 0.77299581+0.01426442j],
                     [0.77096953+0.01426442j, 0.77096953+0.01426442j],
                     [0.77299581+0.01426442j, 0.77299581+0.01426442j],
                     [1.03339092+0.j, 1.03339092+0.j]])
    assert np.all(np.abs(n2_eff - answ) < 1e-7)

def test_eff_index_refr_sq_sloping_seafloor():
    """ Check that the effective index of refraction is computed 
        correctly in an environment with a sloping seafloor """
    z    = np.array([[0,0],[50,50],[100,100],[150,150],[-100,-100],[-50,-50]])
    zb   = np.ones(z.shape) * 90
    dzb  = np.ones(z.shape) * 0.45
    dzb2 = dzb**2
    k0   = 0.041887902047863905
    L    = 75
    r    = 1.0
    rb   = 1.5
    n2   = np.ones(z.shape) * 1.0272096420745072
    n2b  = 0.7783506614961636+0.014264417521561993j
    n2_eff = pe.eff_index_refr_sq(z,zb,dzb2,k0,L,n2,n2b,r,rb)
    assert n2_eff.shape == z.shape
    answ = np.array([[1.03415596+0.j, 1.03415596+0.j],
                    [1.03464263+0.j, 1.03464263+0.j],
                    [0.77191145+0.01426442j, 0.77191145+0.01426442j],
                    [0.76947486+0.01426442j, 0.76947486+0.01426442j],
                    [0.77191145+0.01426442j, 0.77191145+0.01426442j],
                    [1.03464263+0.j, 1.03464263+0.j]])
    assert np.all(np.abs(n2_eff - answ) < 1e-7)

def test_index_refr_sq():
    """ Check that the index of refraction squared is computed correctly 
        including complex attenuation term"""
    n2 = pe.index_refr_sq(c0=1500, c=1700, alpha=0.5)
    assert np.abs(n2 - (0.7783506614961636+0.01426441752156199j)) < 1e-9

def test_init_transm_loss():
    """ Check that we can initialize and instance of the PESolver
        with a flat seafloor and uniform sound speed """
    freq = 100
    prop_range = 40
    angular_bin = 12
    bottom = {'sound_speed':1700,'density':1.5,'attenuation':0.5}
    def bathy_func(x,y,grid=None): return 430*np.ones(x.shape)
    def bathy_deriv_func(x,y,axis): return np.zeros(x.shape)
    def sound_speed_func(x,y,z): return 1480*np.ones(x.shape)   
    transm_loss = pe.TransmissionLoss(freq=freq, bathy_func=bathy_func, 
        bathy_deriv_func=bathy_deriv_func, sound_speed_func=sound_speed_func, 
        bottom=bottom, propagation_range=prop_range, angular_bin=angular_bin)
    assert transm_loss.c0 == 1500
    assert transm_loss.k0 == 2 * np.pi / transm_loss.c0 * freq
    assert transm_loss.grid.dr == transm_loss.c0 / freq
    assert transm_loss.grid.dz == transm_loss.c0 / freq / 2.
    assert transm_loss.grid.dq == pytest.approx(angular_bin * np.pi/180, abs=1e-6)
    assert np.abs(np.max(transm_loss.grid.r) - 1e3*prop_range) < transm_loss.grid.dr
    z_max = 4/3 * (430 + 3 * transm_loss.c0 / freq)
    assert np.abs(np.max(transm_loss.grid.z) - z_max) < transm_loss.grid.dz
    # now, try with kwargs
    transm_loss = pe.TransmissionLoss(freq=freq, bathy_func=bathy_func, 
        bathy_deriv_func=bathy_deriv_func, sound_speed_func=sound_speed_func, 
        bottom=bottom, propagation_range=prop_range, angular_bin=angular_bin,
        dq=24*np.pi/180)
    assert transm_loss.grid.dq == pytest.approx(24 * np.pi/180, abs=1e-6)

def test_transm_loss_solve_pe():
    freq = 100
    prop_range = 0.2
    angular_bin = 180
    grid_kwargs = {'dr':50, 'dz':50, 'z_max':150}
    bottom = {'sound_speed':1700,'density':1.5,'attenuation':0.5}
    source_depth = 16
    def bathy_func(x,y,grid=None): return 90*np.ones(x.shape)
    def bathy_deriv_func(x,y,axis): return np.zeros(x.shape)
    def sound_speed_func(x,y,z): return 1480*np.ones(x.shape)   
    transm_loss = pe.TransmissionLoss(freq=freq, bathy_func=bathy_func, 
        bathy_deriv_func=bathy_deriv_func, sound_speed_func=sound_speed_func, 
        bottom=bottom, propagation_range=prop_range, angular_bin=angular_bin,
        **grid_kwargs)
    transm_loss._init_output(num_sources=1, rec_depth=np.array([.1]))        
    psi = transm_loss._solve_pe(source_depth=source_depth, rec_depth=[.1], return_field=True, progress_bar=False, aperture=86)
    answ = np.array([[[0+0j, 0+0j],
             [7.25511049e-02-5.22233517e-03j, 7.25511049e-02-5.22233517e-03j],
             [7.41434583e-02+1.77258038e-02j, 7.41434583e-02+1.77258038e-02j],
             [0+0j, 0+0j],
             [-7.41434583e-02-1.77258038e-02j, -7.41434583e-02-1.77258038e-02j],
             [-7.25511049e-02+5.22233517e-03j, -7.25511049e-02+5.22233517e-03j]]])
    assert np.all(np.abs(psi - answ) < 1e-9*np.max(np.abs(answ)))

def test_compute_transm_loss_flat_seafloor_uniform_ssp():
    """ Check that PESolver returns expected transmission loss for 
        flat bathymetry and uniform sound speed profile"""
    freq = 10
    prop_range = 10
    angular_bin = 45
    grid_kwargs = {'dr':1000, 'dz':1000}
    bottom = {'sound_speed':1700,'density':1.5,'attenuation':0.5}
    source_depth = 9900
    def bathy_func(x,y,grid=None): return 10000*np.ones(x.shape)
    def bathy_deriv_func(x,y,axis): return np.zeros(x.shape)
    def sound_speed_func(x,y,z): return 1500*np.ones(x.shape)   
    transm_loss = pe.TransmissionLoss(freq=freq, bathy_func=bathy_func, 
        bathy_deriv_func=bathy_deriv_func, sound_speed_func=sound_speed_func, 
        bottom=bottom, propagation_range=prop_range, angular_bin=angular_bin,
        **grid_kwargs)
    tl, axes = transm_loss.calc(source_depth=source_depth, rec_depth=[.1], aperture=88, progress_bar=False)
    expected = np.array([[-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532],
                         [-164.6146, -170.7267, -176.8872, -172.0097, -182.3044, -176.6344, -176.9115, -183.7996, -177.9538, -181.3532]])
    np.testing.assert_array_almost_equal(-tl[0,0], expected, decimal=3) 

def test_compute_transm_loss_multiple_sources():
    """ Check that PESolver can handle multiple source depths"""
    freq = 10
    prop_range = 10
    angular_bin = 45
    grid_kwargs = {'dr':1000, 'dz':1000}
    bottom = {'sound_speed':1700,'density':1.5,'attenuation':0.5}
    def bathy_func(x,y,grid=None): return 10000*np.ones(x.shape)
    def bathy_deriv_func(x,y,axis): return np.zeros(x.shape)
    def sound_speed_func(x,y,z): return 1500*np.ones(x.shape)   
    transm_loss = pe.TransmissionLoss(freq=freq, bathy_func=bathy_func, 
        bathy_deriv_func=bathy_deriv_func, sound_speed_func=sound_speed_func, 
        bottom=bottom, propagation_range=prop_range, angular_bin=angular_bin,
        **grid_kwargs)
    tl0, ax  = transm_loss.calc(source_depth=9900, rec_depth=[.1], aperture=88, progress_bar=False)
    tl1, ax  = transm_loss.calc(source_depth=8800, rec_depth=[.1], aperture=88, progress_bar=False)
    tl01, ax = transm_loss.calc(source_depth=[9900,8800], rec_depth=[.1], aperture=88, progress_bar=False)
    np.testing.assert_array_almost_equal(tl01[0,0], tl0[0,0], decimal=3) 
    np.testing.assert_array_almost_equal(tl01[1,0], tl1[0,0], decimal=3) 
