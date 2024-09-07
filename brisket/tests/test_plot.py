from astropy.io import fits 
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c
import numpy as np
plt.style.use('hba_sans')

import brisket


p = {'redshift': 7.05}
# p['galaxy'] = {
#     'stellar_model': 'BC03',
#     'logMstar': 7.4,
#     'metallicity': 0.05,
#     'sfh': 'constant',
#     'age_min': 0,
#     'age_max': 0.01,
# }
# p['galaxy']['nebular'] = {
#     'type': 'cloudy',
#     'logU': -1.0, 
#     # 'metallicity': 0.001
# }
p['nebular'] = {
    'type': 'flex',
    'fwhm_broad': 3883.3553194999695, 'fwhm_narrow': 28.086774945259094, 'f_Ha_broad': 8.57437789440155e-18, 'f_Ha_narrow': 1.9943654537200928e-18,
    ###
    'cont_type': 'plaw',
    'cont_beta': 0,
    ###
    # 'f_Lya_narrow': 7e-19,
    # 'dv_Lya': 2800, 
    # 'f_CIII_narrow': 1.44e-19,
    # 'f_CIV_narrow': 2.296e-19,
    #'f_Ha_broad': 7.188e-19, 
    # 'f_Ha_narrow': 4.939e-19, 
    #'f_Hb_broad': 1e-20, 
    # 'f_Hb_narrow': 1.149e-19, 
    # 'f_OIII4959_narrow': 1.065e-19, 
    # 'f_OIII5007_narrow': 4.079e-19, 
    'f5100': 1.104e-21,
}
p['calib'] = { # 'calib' handles instrumental calibration, e.g. matching spectral resolution, or polynomial correction factors to account for slit loss/flux calibration
    'R_curve': 'JWST_NIRSpec_PRISM',
    'f_LSF': 1.3,
}

# spec_wavs = brisket.utils.prism_wavs
spec_wavs = fits.getdata('/data/DD6585/final_cal2/jw06585004001_s66964_x1d.fits')['WAVELENGTH']
flux = fits.getdata('/data/DD6585/final_cal2/jw06585004001_s66964_x1d.fits')['FLUX']
flux = (flux*u.Jy*c/(spec_wavs*u.micron)**2).to(u.erg/u.s/u.cm**2/u.angstrom).value

gal = brisket.ModelGalaxy(p, filt_list=['f115w','f150w','f277w','f444w'], spec_wavs=spec_wavs, 
                          wav_units='um', sed_units='ergscma', spec_units='ergscma', logger=brisket.utils.basicLogger)

fig, ax = plt.subplots(figsize=(10,4.5), dpi=200, constrained_layout=True)
ax.step(spec_wavs, flux, where='mid')
ax.step(spec_wavs, gal.spectrum, where='mid')
print(any(np.isnan(gal.spectrum)))

ax.plot(gal.wav_obs, gal.sed)
ax.set_xlim(0.7, 5.35)
ax.set_ylim(-1.7e-21, 4.16e-20)
ax.set_xlabel('Observed Wavelength [µm]')
# ax.set_ylabel(r'$f_{\lambda}$ [erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}]')
ax.set_ylabel('Flux Density [erg/s/cm/cm/A]')
# ax.loglog()
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