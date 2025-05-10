from brisket.models.sfzh import BurstSFH, BaseZHModel
from brisket.models.stellar import BaseStellarModel

zh = BaseZHModel(logZ=0.0)
sfh = BurstSFH(age=0.1)
stars = BaseStellarModel(logMstar=10.0, sfh=sfh, zh=zh, grid='bpass-2.2.1-bin_chabrier03-0.1,300.0')
print(stars)
print(stars.params)

# from brisket.models.dust import BaseDustAttenuationModel
# dust = BaseDustAttenuationModel(Av=0.5)
# dust2 = BaseDustAttenuationModel(Av=0.5)

# zh2 = BaseZHModel(logZ=0.0)
# sfh2 = BurstSFH(age=0.1)
# stars2 = BaseStellarModel(logMstar=10.0, sfh=sfh2, zh=zh2, grid='bpass-2.2.1-bin_chabrier03-0.1,300.0')

formula = stars #* dust + stars2 * dust2

print(formula)
print(formula.params)

from brisket import Model
from brisket.fitting.priors import *
redshift = Uniform(0, 10)
model = Model(redshift, formula, verbose=True)

# from brisket.models.base import *

# emitter1 = EmitterModel(testparam=1)
# emitter2 = EmitterModel(test2=3)
# absorber = AbsorberModel()
# reprocessor = ReprocessorModel(hello='world')
# formula = emitter1 * absorber+ emitter2 + emitter1 % reprocessor * absorber

# print(formula)

# print(formula.params)


quit()

# IDEA: instead of constructing the model using a "fit_instructions" dictionary, 
# build the model using python operations, e.g. +, *, and callable objects ()
# This makes the model construction fully modular, provided I define the 
# necessary methods in the model classes.

#### Simple example
# sfh = ... (star formation history)
# zh = ... (metallicity history)
redshift = Uniform(0, 10) # or some other prior
stars = StellarModel(sfh=sfh, zh=zh, grid='...')
dust_atten = CalzettiAttenuationModel(Av=priors.Uniform(0,1))
agn = AGNModel(...)
igm = Inoue14IGMModel()
model = brisket.Model(redshift, ((stars * dust_atten) + agn) * igm)
# + operation (adding the models together) tells the code to just sum the SEDs 
# * operation (multiplying the models together), tells the code that we're multiplying an SED by some transmission function
# model() (calling the model directly) tells the code that we're reprocessing the SED in some more complicated manner, 
# e.g. by absorbing light <912A and reprocessing as emission lines (nebular model)

#### More complex example, including "reprocessing" models
stars = StellarModel(sfh=sfh, zh=zh, grid='...')
dust_atten = CalzettiAttenuationModel(Av=priors.Uniform(0,1))
dust_emission = MBBDustEmissionModel(energy_balance=True, Tdust=priors.Normal(50, 10))
dust = dust_atten + dust_emission
nebular = CloudyNebularModel(logU=priors.Uniform(-4, 0))
# the model now passes stars through the nebular model, then through the dust model
# then the whole thing gets passed through the IGM model 
# formula = dust(nebular(stars)) * igm
formula = (stars % nebular) % dust * igm




model = brisket.Model(redshift, formula, verbose=True)


obs = Spectrum()
obs += Photometry()

model.predict() 

model.predict(obs, param=param)
# or
model.predict(obs) # (use current params) 



fitter = Fitter(model, obs)
fitter.fit(sampler="ultranest")
fitter.save_results()



# import matplotlib.pyplot as plt
# plt.style.use('hba_sans')
# import numpy as np
# import harmonizer

# # params = harmonizer.Params()
# # # params['redshift'] = np.random.normal(1, 0.1, size=100)
# # params['redshift'] = 1 #harmonizer.priors.Uniform(0, 10)
# # stars = params.add_stars()
# # stars['grid'] = 'test_grid'
# # # stars['logZ'] = 0
# # stars['logMstar'] = 10
# # sfh = stars.add_sfh('constant')
# # sfh['age_min'] = 0.0
# # sfh['age_max'] = 0.1

# # zh = stars.add_zh('delta')
# # zh['zmet'] = 0.005


# from harmonizer.fitting.priors import Uniform
# from harmonizer.models.stellar import CompositeStellarPopulationModel
# from harmonizer.models.sfzh import ConstantSFH, BaseZHModel
# from harmonizer.models.igm import Inoue2014IGMModel


# stars = CompositeStellarPopulationModel(
#     verbose=True, 
#     grid='bpass-2.2.1-bin_chabrier03-0.1,300.0', 
#     logMstar=10)
# sfh = ConstantSFH(age_min=0, age_max=0.1)
# stars.add_sfh(constant=sfh)
# zh = BaseZHModel(logZ=-0.5)
# stars.add_zh(delta=zh)

# igm = Inoue2014IGMModel()
# redshift = 9
# model = harmonizer.Model(
#     redshift, 
#     stars=stars, 
#     igm=igm,
#     verbose=True
# )

# from astropy.io import fits
# f = fits.open('/data/capers/CAPERS_EGS_v0.2/Spectra/CAPERS_EGS_P5/CAPERS_EGS_P5_s000025297_x1d_optext.fits')
# wav_obs = f[1].data['WAVELENGTH']*1e4

# obs = harmonizer.Observation(
#     spec = harmonizer.Spectrum(
#         wavs=wav_obs, 
#         resolution = SpectralResolutionModel(curve='prism', f_LSF=Uniform(0.7, 1.5)), 
#         calibration = SpectralCalibrationModel()
#     )
# )
