from astropy.io import fits 
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
plt.style.use('hba_sans')







fig, ax = plt.subplots()

# import brisket
# mc = {'galaxy':
#         {'model':'BC03','logMstar':10, 'metallicity':0.1, 'sfh':'constant', 'age_min': 0., 'age_max': 0.1, 
#         'nebular':{'logU':-2,'fesc':0.5},
#         'dust_atten':{'type':'Calzetti','Av':3,'logfscat':-2}}, 
#       'redshift': 7}
# gal = brisket.model_galaxy(mc, filt_list=['f115w','f150w','f277w','f444w'])
# ax.semilogx(gal.wav_obs, -2.5*np.log10(gal.spectrum_full/3631e6))

# mc['redshift'] = 5.5
# mc['galaxy']['dust_atten']['Av'] = 1.5
# gal.update(mc)
# ax.semilogx(gal.wav_obs, -2.5*np.log10(gal.spectrum_full/3631e6))
f = fits.getdata('brisket/tests/brisket/posterior/test7/756434_brisket_results.fits', extname='SED_MED')
ax.loglog(f['wav_rest']*8, f['f_lam_50'])

f = fits.getdata('brisket/tests/brisket/posterior/test10/756434_brisket_results.fits', extname='SED_MED')
ax.loglog(f['wav_rest']*8, f['f_lam_50'])

f = fits.getdata('brisket/tests/brisket/posterior/test7/756434_brisket_results.fits', extname='PHOT')
print(f['filter'])
ax.errorbar(f['wav']/1e4, f['flux'], yerr=f['flux_err'], linewidth=0, elinewidth=1, marker='o')

# ax.semilogx(f['wav']/1e4, -2.5*np.log10(gal.photometry/3631e6))

ax.set_xlim(0.7, 10)
ax.set_ylim(7e-3, 2)
plt.show()