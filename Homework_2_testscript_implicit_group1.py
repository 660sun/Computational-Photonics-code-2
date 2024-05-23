'''Homework 2, Computational Photonics, SS 2024:  Beam propagation method.
'''

import numpy as np
from Homework_2_function_headers import waveguide, gauss, beamprop_BN
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