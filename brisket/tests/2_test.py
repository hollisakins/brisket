# import h5py
import numpy as np
import brisket

# create a params object
params = brisket.Params()
params['redshift'] = 7

params.add_source('galaxy', model=brisket.models.GriddedStellarModel)
params['galaxy']['logMstar'] = 10 # 10^{10} Msun
params['galaxy']['zmet'] = 1 # solar metallicity
params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
params['galaxy'].add_sfh('constant1', model=brisket.models.ConstantSFH)
params['galaxy']['constant1']['age_min'] = 0.001 # from 1 Myr
params['galaxy']['constant1']['age_max'] = 0.05 # to 10 Myr
params['galaxy']['constant1']['logweight'] = 0 # low weight
params['galaxy'].add_sfh('constant2', model=brisket.models.ConstantSFH)
params['galaxy']['constant2']['age_min'] = 0.3
params['galaxy']['constant2']['age_max'] = 0.5
params['galaxy']['constant2']['logweight'] = 2

params.add_igm()
params['igm']['xhi'] = 0.9

params.add_calibration()
params['calibration']['R_curve'] = 'PRISM'
params['calibration']['oversample'] = 10
params['calibration']['f_LSF'] = 1.0

params.print_tree()

wavelengths = np.linspace(0.6e4, 5.3e4, 2000)
mod = brisket.ModelGalaxy(params, spec_wavs=wavelengths, verbose=False)


import matplotlib.pyplot as plt
# plt.style.use('brisket/brisket.mplstyle')

# fig, ax = plt.subplots()
fig, ax = mod.sed.plot(x='wav_obs', y='flam')#, ylim=(1e-21, 1e-17))#, xlim=(0.5e4, 50e4))
mod.spectrum.plot(ax=ax, x='wav_obs', y='flam', step=True)
ax.set_xlim(0.6, 5.3)
plt.show()
# # plt.stairs(sfh.ceh.grid[0,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[1,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[2,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[3,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[4,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[5,:], subcomp_params.parent.model.grid_age_bins)
# # plt.stairs(sfh.ceh.grid[6,:], subcomp_params.parent.model.grid_age_bins)
# # plt.step(sfh.ages, sfh.sfh*1e8/3, where='mid')
# # # plt.xlim(10, 20)
# # plt.show()
# # fig, ax = plt.subplots(figsize=(8,4), dpi=130, constrained_layout=True)

# # mod = params['galaxy'].model
# # i, j = 0, 0
# # ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# # i, j = 2, 110
# # ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# # i, j = 3, 150
# # ax.loglog(wavelengths, mod.grid._y[i][j], label=f"Z={mod.grid_metallicities[i]}, age={mod.grid_ages[j]/1e9:.3f} Gyr")
# # ax.legend()
# # plt.show()