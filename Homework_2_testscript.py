'''Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''

import numpy as np
from Homework_2_function_headers_group1 import *
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
output_step = round(4.0/dz)

# create index distribution
n, x = waveguide(xa, xb, Nx, n_cladding, n_core)

## plot index distribution
plt.plot(x, n, label = 'n')
plt.axvline(x=-xb/2, color='r', linestyle='--', label = 'core width')
plt.axvline(x=xb/2, color='r', linestyle='--')
plt.xlabel('x [µm]')
plt.xlim(-xa/2,xa/2)
plt.ylabel('n')
plt.title('Refractive index distribution of the slab waveguide')
plt.grid(True)
plt.legend()
plt.show()


# create initial field
v_in, x     = gauss(xa, Nx, w)
v_in        = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity

## plot of initial field distribution
plt.plot(x, v_in, label = ' intensity distribution')
plt.axvline(x=-xb/2, color='r', linestyle='--', label = 'core width')
plt.axvline(x=xb/2, color='r', linestyle='--')
plt.xlabel('x [µm]')
plt.xlim(-xa/2,xa/2)
plt.ylabel('Normalized field intensity')
plt.title('Normalized initial field initensity $v_{in}$ in slab waveguide')
plt.grid(True)
plt.legend()
plt.show()

# calculation
v_out, z = beamprop_CN(v_in, lam, dx, n, nd,  z_end, dz, output_step)            # explicit-implicit scheme
v_out_ex, z_ex = beamprop_FN(v_in, lam, dx, n, nd,  z_end, dz, output_step)      # explicit scheme
v_out_im, z_im = beamprop_BN(v_in, lam, dx, n, nd,  z_end, dz, output_step)      # implicit scheme

### plots of each scheme



# %%
