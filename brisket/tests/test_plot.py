from astropy.io import fits 
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
plt.style.use('hba_sans')

import brisket
p = brisket.parameters.Params('basic')
### OR 
# p = brisket.parameters.Params('tests/param_fit_test.toml')

# gal = brisket.ModelGalaxy(p, filt_list=['f115w','f150w','f277w','f444w'], logger=brisket.utils.basicLogger)



def load_phot(ID):
    flux = 3631e6*np.power(10., -0.4*np.array([23.8, 23.9, 23.5, 22.9]))
    err = 0.1*flux
    return np.array([flux, err]).T

# p = {'redshift': 7}
# p['galaxy'] = {
#     'stellar_model': 'BC03',
#     'logMstar': {'low':10,'high':12},
#     'metallicity': {'low':0.001, 'high':1.0, 'prior':'log_10'},
#     'sfh': 'constant',
#     'age_min': 0,
#     'age_max': 1,
# }
# p = brisket.parameters.Params(p)

gal = brisket.Galaxy(1, load_phot=load_phot, filt_list=['f115w','f150w','f277w','f444w'], logger=brisket.utils.basicLogger)
fitter = brisket.Fitter(gal, p, run='test828', logger=brisket.utils.basicLogger)
fitter.fit(verbose=True, n_live=100, sampler='multinest')


fig, ax = plt.subplots()

# import brisket
# mc = {'galaxy':
#         {'model':'BC03','logMstar':10, 'metallicity':0.1, 'sfh':'constant', 'age_min': 0., 'age_max': 0.1, 
#         'nebular':{'logU':-2,'fesc':0.5},
#         'dust_atten':{'type':'Calzetti','Av':3,'logfscat':-2}}, 
#       'redshift': 7}
# gal = brisket.model_galaxy(mc, filt_list=['f115w','f150w','f277w','f444w'])


f = fits.getdata('brisket/posterior/test828/1_brisket_results.fits', extname='SED_MED')
ax.semilogx(f['wav_rest']*8, -2.5*np.log10(f['f_lam_50']/3631e6))
f = fits.getdata('brisket/posterior/test828/1_brisket_results.fits', extname='PHOT')
print(f['flux'])
ax.scatter(f['wav']/1e4, -2.5*np.log10(f['flux']/3631e6))

# ax.semilogx(gal.wav_obs, -2.5*np.log10(gal.sed/3631e6))

# mc['redshift'] = 5.5
# mc['galaxy']['dust_atten']['Av'] = 1.5
# gal.update(mc)
# ax.semilogx(gal.wav_obs, -2.5*np.log10(gal.spectrum_full/3631e6))
# f = fits.getdata('brisket/tests/brisket/posterior/test7/756434_brisket_results.fits', extname='SED_MED')
# ax.loglog(f['wav_rest']*8, f['f_lam_50'])

# f = fits.getdata('brisket/tests/brisket/posterior/test10/756434_brisket_results.fits', extname='SED_MED')
# ax.loglog(f['wav_rest']*8, f['f_lam_50'])

# f = fits.getdata('brisket/tests/brisket/posterior/test7/756434_brisket_results.fits', extname='PHOT')
# print(f['filter'])
# ax.errorbar(f['wav']/1e4, f['flux'], yerr=f['flux_err'], linewidth=0, elinewidth=1, marker='o')

# ax.semilogx(f['wav']/1e4, -2.5*np.log10(gal.photometry/3631e6))

ax.set_xlim(0.5, 10)
# ax.set_ylim(7e-3, 2)
ax.set_ylim(28, 20)
plt.show()