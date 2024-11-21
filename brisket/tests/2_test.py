# import h5py
import numpy as np

# with h5py.File('brisket/models/grids/bc03_miles_chabrier.hdf5','r') as f:
#     axes = [a.decode('utf-8') for a in list(f['axes'])]
#     axes_nowav = [a for a in axes if a!='wavs']
#     x = [np.array(f[axis]) for axis in axes_nowav]
#     y = np.array(f['grid'])

# # print(y[3,10])
# from scipy.interpolate import RegularGridInterpolator
# interp = RegularGridInterpolator(x, y)
# print(interp((0.9, 2)))

import brisket
brisket.config.params_print_tree = True
brisket.config.params_print_summary = False

class NullModel:
    type = 'reprocessor'
    order = 100
    def __init__(self, params):
        self.params = params
    def _resample(self, wavs):
        self.wavelengths = wavs

# create a params object
params = brisket.Params()
params['redshift'] = 10

params.add_source('galaxy', model=brisket.models.GriddedStellarModel)
params['galaxy']['logMstar'] = 10
params['galaxy']['zmet'] = 1
params['galaxy'].add_sfh('constant', model=brisket.models.ConstantSFH)
params['galaxy']['constant']['age_min'] = 0.1
params['galaxy']['constant']['age_max'] = 0.2

print(params)


wavelengths = np.linspace(100, 8000, 5000)
components = params.components
for comp_name, comp_params in components.items(): 
    comp_params.model = comp_params.model(comp_params) # initialize the model
    comp_params.model._resample(wavelengths) # resample the model

    subcomps = comp_params.components
    for subcomp_name, subcomp_params in subcomps.items():
        # print(subcomp_params.parent.model.grid_ages)
        # break
        subcomp_params.model = subcomp_params.model(subcomp_params)
        subcomp_params.model._resample(wavelengths)

# mod = params['galaxy'].model
# sfh = mod.sfh_components['constant']
# sfh.update(params)
# print(.sfh)
sfh = subcomp_params.model
sfh.update(subcomp_params)
print(sfh.sfh)
# 
import matplotlib.pyplot as plt
plt.plot(sfh.ages, sfh.sfh)
plt.show()
# fig, ax = plt.subplots(figsize=(8,4), dpi=130, constrained_layout=True)

# mod = params['galaxy'].model
# i, j = 0, 0
# ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# i, j = 2, 110
# ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# i, j = 3, 150
# ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# ax.legend()
# plt.show()