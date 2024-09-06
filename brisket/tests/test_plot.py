from astropy.io import fits 
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
plt.style.use('hba_sans')

import brisket


p = {'redshift': 6}
# p['galaxy'] = {
#     'stellar_model': 'BPASS',
#     'logMstar': 10,
#     'metallicity': 0.001,
#     'sfh': 'constant',
#     'age_min': 0,
#     'age_max': 0.3,
# }
p['nebular'] = {
    'type': 'flex',
    'f_Ha_broad': 1e-19, 
    'f_Ha_narrow': 1e-19, 
    'fwhm_broad': 5000,
    'fwhm_narrow': 300,
    'cont_type': 'dblplaw',
    'cont_beta1': -2.5,
    'cont_beta2': -2,
    'cont_break':3800,
    'f5100': 5e-21,
}
p['calib'] = {
    'R_curve': 'JWST_NIRSpec_PRISM'
}

spec_wavs = fits.getdata('/data/DD6585/final_cal2/jw06585004001_s66964_x1d.fits')['WAVELENGTH']*1e4
gal = brisket.ModelGalaxy(p, filt_list=['f115w','f150w','f277w','f444w'], spec_wavs=spec_wavs, logger=brisket.utils.basicLogger, spec_units=u.uJy)

fig, ax = plt.subplots()
ax.loglog(gal.wav_obs, gal.sed)
ax.step(spec_wavs/1e4, gal.spectrum[:,1], where='mid')
ax.set_xlim(0.7, 9)
ax.set_ylim(0.04, 2)
ax.loglog()
plt.show()



quit()


p = brisket.parameters.Params('basic')
### OR 
# p = brisket.parameters.Params('tests/param_fit_test.toml')

# gal = brisket.ModelGalaxy(p, filt_list=['f115w','f150w','f277w','f444w'], logger=brisket.utils.basicLogger)



def load_phot(ID):
    flux = 3631e6*np.power(10., -0.4*np.array([23.8, 23.9, 23.5, 22.9]))
    err = 0.1*flux
    return np.array([flux, err]).T

p = {'redshift': 7}
p['galaxy'] = {
    'stellar_model': 'BC03',
    'logMstar': {'low':10,'high':12},
    'metallicity': {'low':0.001, 'high':1.0, 'prior':'log_10'},
    'sfh': 'constant',
    'age_min': 0,
    'age_max': 1,
}
# p['nebular'] = {
#     'type': 'flex',
#     'cont_type': 'flat',
#     'f5100': 1e-18, 
#     # case 1: no broad+narrow, all lines have fixed FWHM
#     'fwhm': {'low':100, 'high': 1000},
#     'f_Ha': {'low': 1e-20, 'high': 1e-19}, # prior?
#     'f_Hb': {'low': 1e-20, 'high': 1e-19},
#     # case 2: broad+narrow lines, but tied together in FWHM
#     'fwhm_broad': {'low':500, 'high':5000}, 
#     'fwhm_narrow': {'low':50, 'high':500}, 
#     'f_Ha_broad': {'low': 1e-20, 'high': 1e-19}, # prior?
#     'f_Hb_broad': {'low': 1e-20, 'high': 1e-19},
#     'f_Ha_narrow': {'low': 1e-20, 'high': 1e-19},
#     'f_Hb_narrow': {'low': 1e-20, 'high': 1e-19},
#     # case 3: each line has its own fwhm, 
#     'f_Ha': {'low':1e-20, 'high':1e-19}, 
#     'fwhm_Ha': {'low':100, 'high':500},
#     'dv_Ha': {'low':-1000, 'high':1000},

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