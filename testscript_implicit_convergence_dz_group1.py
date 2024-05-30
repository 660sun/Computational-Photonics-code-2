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

# propagation step size - 31 points logarithmically spaced between 10 and 0.053
dz = np.logspace(np.log10(1), np.log10(0.05), 31)
operation_time = np.zeros(dz.size)
field_end = np.zeros((dz.size, Nx))
dz_num = []
counter = 0
for i, dzi in enumerate(dz):
    output_step = round(1.0/dzi)
    start = time.time()
    v_out, z = beamprop_BN(v_in, lam, dx, n, nd,  z_end, dzi, output_step)
    stop = time.time()
    operation_time[i] = stop - start
    print("dz = %6.3f, time = %gs" % (dzi, stop - start))
    z = z[::output_step]
    if abs(z[-1] - z_end) < 1e-6:
        field_end[counter, :] = np.abs(v_out[-1, :])**2
        dz_num.append(dzi)
        counter += 1

# calculate relative error to the value obtained at highest resolution
field_end = field_end[np.any(field_end != 0, axis=1)]
real_error = []
for i in range(field_end.shape[0]):
    real_error.append(np.linalg.norm(field_end[-1, :] - field_end[i, :]) / np.linalg.norm(field_end[-1, :]))


# Plot of operation time
plt.figure()
plt.plot(dz, operation_time, 'o-')
plt.xlabel('dz [µm]')
plt.ylabel('operation time [s]')
plt.title('Operation time for different dz')
# plt.xscale('log')
# plt.yscale('log')
plt.show()

# Plot of relative error
plt.figure()
plt.plot(dz_num, real_error, 'o-')
plt.xlabel('dz [µm]')
plt.ylabel('relative error')
plt.title('Relative error for different dz')
# plt.xscale('log')
# plt.yscale('log')
plt.show()
