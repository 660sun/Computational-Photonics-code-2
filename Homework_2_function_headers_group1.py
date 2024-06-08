'''Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


def waveguide(xa, xb, Nx, n_cladding, n_core):
    '''Generates the refractive index distribution of a slab waveguide
    with step profile centered around the origin of the coordinate
    system with a refractive index of n_core in the waveguide region
    and n_cladding in the surrounding cladding area.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        xb : float
            Width of waveguide
        Nx : int
            Number of grid points
        n_cladding : float
            Refractive index of cladding
        n_core : float
            Refractive index of core

    Returns
    -------
        n : 1d-array
            Generated refractive index distribution
        x : 1d-array
            Generated coordinate vector
    '''
    
    x = np.linspace(-xa/2, xa/2, Nx)
    n = np.zeros(Nx, dtype=float)
    # index distribution
    for i in range(Nx):
        if abs(x[i]) <= xb/2:
            n[i] = n_core
        else:
            n[i] = n_cladding
    # for i, xi in enumerate(x):
    #     if abs(xi) <= xb/2:
    #         n[i] = n_core
    #     else:
    #         n[i] = n_cladding

    return n, x




def gauss(xa, Nx, w):
    '''Generates a Gaussian field distribution v = exp(-x^2/w^2) centered
    around the origin of the coordinate system and having a width of w.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        Nx : int
            Number of grid points
        w  : float
            Width of Gaussian field

    Returns
    -------
        v : 1d-array
            Generated field distribution
        x : 1d-array
            Generated coordinate vector
    '''
    
    # v = np.zeros(Nx, dtype=float)
    x = np.linspace(-xa/2, xa/2, Nx)
    v = np.exp(-x**2/w**2)

    return v, x




def beamprop_CN(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit-implicit
    Crank-Nicolson scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''
    
    # Basic parameters - wavenumbers
    k0 = 2*np.pi/lam
    kd = nd*k0
    k1 = n*k0
    k2 = np.ones(len(k1)) * kd

    # Construction of the operator matrix L1
    ## Diagonal elements
    diagonals_1 = np.zeros((3, len(n)))
    diagonals_1[0] = np.ones(len(n)) * (-2)
    diagonals_1[1] = np.ones(len(n)) * 1
    diagonals_1[-1] = np.ones(len(n)) * 1
    diag_position_1 = [0, 1, -1]
    ## Sparse matrix construction
    L1 = sps.diags(diagonals_1, diag_position_1)
    L1 = (-1j/(2*kd*dx**2)) * L1

    # Construction of the operator matrix L2
    ## Diagonal elements
    diagonals_2 = np.zeros((1, len(n)))
    diagonals_2[0] = (k1**2 - k2**2) / (2*kd)
    diag_position_2 = [0]
    ## Sparse matrix construction
    L2 = sps.diags(diagonals_2, diag_position_2)
    L2 = -1j*L2

    # Construction of the operator matrix L
    L = L1 + L2

    ## Construction of the operator matrix M1        
    M1 = sps.eye(len(n)) - (dz/2) * L
    ## Construction of the operator matrix M2
    M2 = sps.eye(len(n)) + (dz/2) * L

    # Crank-Nicolson scheme
    # Consider output_step with z = linspace(0, z_end, int(z_end/dz) + 1)
    # z = np.linspace(0, z_end, int(z_end/dz) + 1)
    # v_out = np.zeros((len(z), len(n)), dtype=complex)
    # v_out[0,:] = v_in
    # for i in range(len(z) - 1):
    #     # Solution of the slowly varying envelope along the propagation direction
    #     v_out[i + 1 ,:] = sps.linalg.spsolve(M1, M2.dot(v_out[i,:]))
    #     i += 1


    # Consider output_step with z = z[i] + dz
    z = []
    z.append(0)
    v_out = []
    v_out.append(v_in)
    counter = 1
    i = 0
    for i in range(int(z_end/dz) + 1):
        if z[i] > z_end:
            break
        v_out.append(sps.linalg.spsolve(M1, M2.dot(v_out[counter - 1][:])))
        z.append(z[i] + dz)
        i += 1
        counter += 1
    
    # selection using output_step
    v_out = v_out[::output_step]
    z = z[::output_step]

    return v_out, z




def beamprop_FN(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''
    # Basic parameters - wavenumbers
    k0 = 2*np.pi/lam
    kd = nd*k0
    k1 = n*k0
    k2 = np.ones(len(k1)) * kd

    # Construction of the operator matrix L1
    ## Diagonal elements
    diagonals_1 = np.zeros((3, len(n)))
    diagonals_1[0] = np.ones(len(n)) * (-2)
    diagonals_1[1] = np.ones(len(n)) * 1
    diagonals_1[-1] = np.ones(len(n)) * 1
    diag_position_1 = [0, 1, -1]
    ## Sparse matrix construction
    L1 = sps.diags(diagonals_1, diag_position_1)
    L1 = (-1j/(2*kd*dx**2)) * L1

    # Construction of the operator matrix L2
    ## Diagonal elements
    diagonals_2 = np.zeros((1, len(n)))
    diagonals_2[0] = (k1**2 - k2**2) / (2*kd)
    diag_position_2 = [0]
    ## Sparse matrix construction
    L2 = sps.diags(diagonals_2, diag_position_2)
    L2 = -1j*L2

    # Construction of the operator matrix L
    L = L1 + L2

    ## Construction of the operator matrix M2
    M2 = sps.eye(len(n)) + (dz/2) * L

    # Explicit scheme
    # Consider output_step with z = linspace(0, z_end, int(z_end/dz) + 1)
    # z = np.linspace(0, z_end, int(z_end/dz) + 1)
    # v_out = np.zeros((len(z), len(n)), dtype=complex)
    # v_out[0,:] = v_in
    # for i in range(len(z) - 1):
    #     # Solution of the slowly varying envelope along the propagation direction
    #     v_out[i + 1,:] = M2.dot(v_out[i,:])
    #     i += 1

    # Consider output_step with z = z[i] + dz
    z = []
    z.append(0)
    v_out = []
    v_out.append(v_in)
    counter = 1
    i = 0
    for i in range(int(z_end/dz) + 1):
        if z[i] > z_end:
            break
        M2 = sps.eye(len(n)) + (dz) * L
        v_out.append(M2.dot(v_out[counter - 1][:]))
        z.append(z[i] + dz)
        i += 1
        counter += 1

    # selection using output_step
    v_out = v_out[::output_step]
    z = z[::output_step]


    return v_out, z





def beamprop_BN(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the implicit scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''
    
    # Basic parameters - wavenumbers
    k0 = 2*np.pi/lam
    kd = nd*k0
    k1 = n*k0
    k2 = np.ones(len(k1)) * kd

    # Construction of the operator matrix L1
    ## Diagonal elements
    diagonals_1 = np.zeros((3, len(n)))
    diagonals_1[0] = np.ones(len(n)) * (-2)
    diagonals_1[1] = np.ones(len(n)) * 1
    diagonals_1[-1] = np.ones(len(n)) * 1
    diag_position_1 = [0, 1, -1]
    ## Sparse matrix construction
    L1 = sps.diags(diagonals_1, diag_position_1)
    L1 = (-1j/(2*kd*dx**2)) * L1

    # Construction of the operator matrix L2
    ## Diagonal elements
    diagonals_2 = np.zeros((1, len(n)))
    diagonals_2[0] = (k1**2 - k2**2) / (2*kd)
    diag_position_2 = [0]
    ## Sparse matrix construction
    L2 = sps.diags(diagonals_2, diag_position_2)
    L2 = -1j*L2

    # Construction of the operator matrix L
    L = L1 + L2

    ## Construction of the operator matrix M1
    M1 = sps.eye(len(n)) - (dz) * L

    # Implicit scheme
    # Consider output_step with z = linspace(0, z_end, int(z_end/dz) + 1)
    # z = np.linspace(0, z_end, int(z_end/dz) + 1)
    # v_out = np.zeros((len(z), len(n)), dtype=complex)
    # v_out[0,:] = v_in
    # for i in range(len(z) - 1):
    #     # Solution of the slowly varying envelope along the propagation direction
    #     v_out[i + 1,:] = sps.linalg.spsolve(M1, v_out[i,:])
    #     i += 1


    # Consider output_step with z = z[i] + dz
    z = []
    z.append(0)
    v_out = []
    v_out.append(v_in)
    counter = 1
    i = 0
    for i in range(int(z_end/dz) + 1):
        if z[i] > z_end:
            break
        v_out.append(sps.linalg.spsolve(M1, v_out[counter - 1][:]))
        z.append(z[i] + dz)
        i += 1
        counter += 1

    # selection using output_step
    v_out = v_out[::output_step]
    z = z[::output_step]

    return v_out, z

