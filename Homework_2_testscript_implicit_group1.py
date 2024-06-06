'''Homework 2, Computational Photonics, SS 2024:  Beam propagation method.
'''

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

# %%
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

# source width
w       = 5.0       # Gaussian beam width

# propagation step size
dz = 0.5
output_step = round(1.0/dz)

# create index distribution
n, x = waveguide(xa, xb, Nx, n_cladding, n_core)

# create initial field
v_in, x     = gauss(xa, Nx, w)
v_in        = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity

# calculation
v_out, z = beamprop_BN(v_in, lam, dx, n, nd,  z_end, dz, output_step)
for i in range(len(z)):
    v_out[i] = v_out[i]/np.sqrt(np.sum(np.abs(v_out[i])**2)) # normalize power to unity

# Plot results - x-z plane
plt.figure()
plt.pcolormesh(x, z, np.abs(v_out)**2, cmap='bluered_dark')
plt.axvline(x=-xb/2, color='gray', linestyle='--')
plt.axvline(x=xb/2, color='gray', linestyle='--')
plt.xlabel('x [µm]')
plt.ylabel('z [µm]')
plt.title('Field intensity distribution in the x-z plane of the waveguide \n Implicit scheme')
plt.gca().set_aspect('equal')
cb = plt.colorbar()
cb.set_label('intensity')
plt.show()

# Plot results - x direction
plt.figure()
for i in range(0, len(z), 6):
    plt.plot(x, np.abs(v_out[i])**2, label='z = %d' % z[i])
plt.axvline(x=-xb/2, color='r', linestyle='--')
plt.axvline(x=xb/2, color='r', linestyle='--')
plt.xlabel('x [µm]')
plt.ylabel('intensity')
plt.title('Field intensity distribution in the x direction at different z values \n Implicit scheme')
plt.legend()
plt.show()