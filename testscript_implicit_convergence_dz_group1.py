""" Tests the convergence when varying the discretization.
"""

import time
import numpy as np
from Homework_2_function_headers_group1 import waveguide, gauss, beamprop_BN
from matplotlib import pyplot as plt

# dark bluered colormap, registers automatically with matplotlib on import
import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.925,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')

#%%
# computational parameters
z_end   = 100       # propagation distance
lam     = 1         # wavelength
nd      = 1.455     # reference index
xa      = 50        # size of computational window
Nx      = 251       # number of transverse points
dx      = xa/(Nx-1) # transverse step size
output_step = 1     # output step size

# waveguide parameters
xb      = 2.0       # size of waveguide
n_cladding  = 1.45      # cladding index
n_core  = 1.46      # core refr. index

# create index distribution
n, x = waveguide(xa, xb, Nx, n_cladding, n_core)

# source width
w       = 5.0       # Gaussian beam width

# create initial field
v_in, x     = gauss(xa, Nx, w)
v_in        = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity

# propagation step size - points between 1 and 0.01
# dz = [0.01, 0.0125, 0.016, 0.02, 0.025, 0.04, 0.05, 0.08, 0.1, 0.125, 0.16, 0.2, 0.25, 0.4, 0.5, 0.8, 1.0]
# dz = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.0625, 0.078125, 0.1, 0.125, 0.2, 0.25, 0.4, 0.5, 0.625, 0.78125, 1.0]
dz = [0.01, 0.0125, 0.016, 0.02, 0.025, 0.04, 0.05, 0.0625, 0.08, 0.1, 0.125, 0.16, 0.2, 0.25, 0.4, 0.5, 0.625, 0.78125, 0.8, 1.0]

operation_time = np.zeros(len(dz))
field_end = np.zeros((len(dz), Nx))
counter = 0

plt.figure()

for i, dzi in enumerate(dz):
    start = time.time()
    v_out, z = beamprop_BN(v_in, lam, dx, n, nd,  z_end, dzi, output_step)
    stop = time.time()
    operation_time[i] = stop - start
    # Plot results - z direction at x = 0
    plt.plot(z, [np.abs(v[Nx//2])**2 for v in v_out], label=f'dz={dzi:.6f}')
    print("dz = %6.3f, time = %gs" % (dzi, stop - start))
    field_end[counter][:] = np.abs(v_out[-1][:])**2
    counter += 1

plt.xlabel('z [µm]')
plt.ylabel('intensity')
plt.title('Field intensity distribution at x = 0 \n Implicit scheme')
plt.legend()
plt.show()

# calculate relative error to the value obtained at highest resolution
real_error = []
for i in range(field_end.shape[0]):
    real_error.append(np.linalg.norm(field_end[0][:] - field_end[i][:]) / np.linalg.norm(field_end[0][:]))


# Plot of operation time
plt.figure()
plt.plot(dz, operation_time, 'o-')
plt.xlabel('dz [µm]')
plt.ylabel('operation time [s]')
plt.title('Operation time for different dz \n Implicit scheme')
plt.show()

# Plot results - x direction
plt.figure()
for i in range(len(dz)):
    plt.plot(x, field_end[i], label='dz = %.6f' % dz[i])
plt.axvline(x=-xb/2, color='r', linestyle='--')
plt.axvline(x=xb/2, color='r', linestyle='--')
plt.xlabel('x [µm]')
plt.ylabel('intensity')
plt.title('Field intensity distribution at far end for different dz \n Implicit scheme')
plt.legend()
plt.show()


# Plot of relative error
plt.figure()
plt.plot(dz, real_error, 'o-')
plt.xlabel('dz [µm]')
plt.ylabel('relative error')
plt.title('Relative error for different dz \n Implicit scheme')
plt.yscale('log')
plt.show()
