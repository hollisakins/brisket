import h5py
import numpy as np

with h5py.File('brisket/models/grids/bc03_miles_chabrier.hdf5','r') as f:
    axes = [a.decode('utf-8') for a in list(f['axes'])]
    axes_nowav = [a for a in axes if a!='wavs']
    x = [np.array(f[axis]) for axis in axes_nowav]
    y = np.array(f['grid'])

# print(y[3,10])
from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator(x, y)
print(interp((0.9, 2)))


