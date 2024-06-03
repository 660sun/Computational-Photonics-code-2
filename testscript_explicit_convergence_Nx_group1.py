""" Tests the convergence when varying the discretization Nx.
"""

import time
import numpy as np
from Homework_2_function_headers_group1 import waveguide, gauss, beamprop_FN
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

# propagation step size
dz = 0.5
output_step = round(1.0/dz)

# waveguide parameters
xb      = 2.0       # size of waveguide
n_cladding  = 1.45      # cladding index
n_core  = 1.46      # core refr. index

# source width
w       = 5.0       # Gaussian beam width

Nx = np.linspace(201, 301, 10, dtype=int)
operation_time = np.zeros(len(Nx))
field_end = []
for i, Nxi in enumerate(Nx):
    dx      = xa/(Nxi-1) # transverse step size
    # create index distribution
    n, x = waveguide(xa, xb, Nxi, n_cladding, n_core)
    # create initial field
    v_in, x     = gauss(xa, Nxi, w)
    v_in        = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity
    # propagate field
    start = time.time()
    v_out, z = beamprop_FN(v_in, lam, dx, n, nd,  z_end, dz, output_step)
    stop = time.time()
    operation_time[i] = stop - start
    field_end.append(v_out[-1][:])
    print("Nx = %s, time = %gs" % (Nxi, stop - start))

# calculate relative error to the value obtained at highest resolution
real_error = np.zeros(len(Nx))
for i in range(len(Nx)):
    real_error[i] = 1 - np.linalg.norm(field_end[i])/np.linalg.norm(field_end[-1])



# Plot of operation time
plt.figure()
plt.plot(Nx, operation_time, 'o-')
plt.xlabel('Nx')
plt.ylabel('operation time [s]')
plt.title('Operation time for different Nx')
# plt.xscale('log')
# plt.yscale('log')
plt.show()

# Plot of relative error
plt.figure()
plt.plot(Nx, real_error, 'o-')
plt.xlabel('Nx')
plt.ylabel('relative error')
plt.title('Relative error for different Nx')
# plt.xscale('log')
# plt.yscale('log')
plt.show()