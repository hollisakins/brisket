import brisket
import numpy as np

params = {}
params['redshift'] = 8.9
params['galaxy'] = {'logMstar':8, 'metallicity':0.1,'sfh':'constant', 'age_min':0, 'age_max':0.1}
# params['galaxy']['nebular'] = {'type':'cloudy', 'logU':-2}
params['igm'] = {'xhi': 0.5}

gal = brisket.ModelGalaxy(params, spec_wavs=np.linspace(1, 5, 1000), sed_units='ergscma')

import matplotlib.pyplot as plt
plt.style.use('hba_default')

fig, ax = plt.subplots(figsize=(6,3))

params['igm'] = {}; gal.update(params)
ax.plot(gal.wav_obs, gal.sed, label='No IGM damping')

params['igm']['xhi'] = 0; gal.update(params)
ax.plot(gal.wav_obs, gal.sed, label='X=0', linestyle='--')

params['igm']['xhi'] = 0.5; gal.update(params)
ax.plot(gal.wav_obs, gal.sed, label='X=0.5')

params['igm']['xhi'] = 1.0; gal.update(params)
ax.plot(gal.wav_obs, gal.sed, label='X=1.0')

ax.set_xlim(1.1, 1.5)
# ax[0].set_yscale('log')

# params['igm']['xhi'] = 0.5; gal.update(params)
# plt.plot(gal.spec_wavs, gal.spectrum, label='X=0.5')
# params['igm']['xhi'] = 1.0; gal.update(params)
# plt.plot(gal.spec_wavs, gal.spectrum, label='X=1.0')

ax.set_ylabel(r'$f_\lambda$')
ax.set_xlabel(r'Observed wavelength [$\mu$m]')

ax.legend()

plt.show()
