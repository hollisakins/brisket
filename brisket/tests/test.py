import brisket
# import bagpipes
import numpy as np
import astropy.units as u

# brisket.parse_toml_paramfile('tests/param_fit_test.toml')


# gal1 = brisket.model_galaxy('example_param.toml', wav_units=u.um, sed_units=u.uJy, filt_list=['jwst_nircam_f277w'])

# # mc = {'delayed':{'massformed':8, 'metallicity':0.2, 'age': 0.1, 'tau': 0.1}, 'nebular':{'logU':-2}, 'redshift': 5}
# # gal2 = bagpipes.model_galaxy(mc, filt_list=['jwst_nircam_f277w'])

from astropy.io import fits
f = fits.getdata('brisket/tests/brisket/test2/test2.fits')

import matplotlib.pyplot as plt
plt.style.use('hba_default')

fig, ax = plt.subplots()
ax.semilogx(f['wav_obs'], -2.5*np.log10(f['flux']/3631e6), label='BRISKET')
# from astropy.constants import c
# ax.semilogx(gal2.wavelengths/1e4*6, -2.5*np.log10((gal2.spectrum_full *u.erg/u.s/u.cm**2/u.angstrom * (gal2.wavelengths*6*u.angstrom)**2 / c).to(u.uJy).value /3631e6), label='BAGPIPES')

ax.set_xlim(0.4, 50)
ax.set_ylim(30, 24)
ax.legend()
plt.show()

