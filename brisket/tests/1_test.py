import brisket


# create a params object
params = brisket.Params()
params.add_source('galaxy')
params['galaxy']['grids'] = 'bc03'
params['galaxy']['logMstar'] = brisket.FreeParam(low=5, high=12)
params['galaxy']['zmet'] = brisket.FreeParam(low=0.001, high=2.5, prior='log_uniform')

params['galaxy'].add_sfh('continuity')#, model=brisket.models.ContinuitySFH)
# if 'continuity' in name:
#     self.model = brisket.models.sfh.ContinuitySFH
params['galaxy']['continuity']['bin_edges'] = [0, 10, 30, 100]
params['galaxy']['continuity']['n_bins'] = 7
params['galaxy']['continuity']['z_max'] = 20

params['galaxy'].add_nebular()
params['galaxy']['nebular']['logU'] = brisket.FreeParam(low=-4, high=-1)

params['galaxy'].add_dust()#model=brisket.models.CalzettiDustAttenuationModel)
params['galaxy']['dust']['Av'] = brisket.FreeParam(low=0.001, high=5, prior='log_uniform')

# print(params['galaxy'].sources)
# print(params['galaxy']['nebular'].all_params)

print(params)
# print(params.free_param_names)
# print(params.free_param_priors)
# you can also import a specific template, e.g.
# params = brisket.Params(template='Akins24a') # for fitting LRD models
# table = params.__repr__()

# console.print(table)
quit()


params.add_source('galaxy')
params['galaxy']['grids'] = 'bc03'
params['galaxy']['logMstar'] = brisket.FreeParam(low=5, high=12)
params['galaxy']['zmet'] = brisket.FreeParam(low=0, high=2)

params['galaxy'].add_sfh('continuity')
params['galaxy']['continuity']['bin_edges'] = [0, 10, 30, 100]
params['galaxy']['continuity']['n_bins'] = brisket.FixedParam(7)
params['galaxy']['continuity']['z_max'] = 20

params['galaxy'].add_sfh('burst')
params['galaxy']['burst']['age'] = brisket.FreeParam(low=0, high=1)

params['galaxy'].add_nebular()
params['galaxy']['nebular']['logU'] = brisket.FreeParam(low=-4, high=-1)

params['galaxy'].add_dust()
params['galaxy']['dust']['Av'] = brisket.FreeParam(low=0, high=5)
params['galaxy']['dust']['delta'] = brisket.FixedParam(0)

# params.add_source('agn')
# params['agn']['gas']['f_Ha'] = brisket.FreeParam(low=0, high=1e-17)
# params['agn']['gas']['f_Hb'] = brisket.LinkedParam(params['agn']['gas']['f_Ha'], scale=1/2.86)

# More advanced users might want to add custom sources into BRISKET. This is made possible via 
# the `model` keyword, which tells brisket which model class to use for a given source. Parameters 
# defined under this source must be applicable to the arbitrary source
from my_code import CustomArbitrarySourceModel
params.add_source('arbitrary_name', model=CustomArbitrarySourceModel)
# For example, to fit a spectrum alone, which maybe only has nebular lines, you can add a flexible 
# nebular model
params.add_source('nebular', model=brisket.models.FlexibleNebularModel)
params['nebular']['fwhm'] = brisket.FreeParam(low=100, high=500)

params.add_source('dust', model=brisket.models.MBBPowerLawDustEmissionModel)
params['nebular']['Tdust'] = brisket.FreeParam(low=20, high=60)
params['nebular']['lum'] = brisket.FreeParam(low=20, high=60)




# params.validate() # <- this will check that all required parameters are defined, 
#                        warn you if the code is using defaults, and define several 
#                        internally-used variables. automatically run when a params 
#                        object is passed to ModelGalaxy or Fitter
# print(params.flatten())
# print(params)

# from brisket.models import 
# StellarModel = stellar models  CloudyNebularModel, FlexibleNebularModel






# `incident` spectra are the spectra that serve as an input to the photoionisation modelling. 
# In the context of stellar population synthesis these are the spectra that are produced by 
# these codes and equivalent to the “pure stellar” spectra.

# `transmitted` spectra is the incident spectra that is transmitted through the gas in the 
# photoionisation modelling. Functionally the main difference between transmitted and incident
# is that the transmitted has little flux below the Lyman-limit, since this has been absorbed 
# by the gas. This depends on fesc.

# `nebular` is the nebular continuum and line emission predicted by the photoionisation model. 
# This depends on fesc.

# `reprocessed` is the emission which has been reprocessed by the gas. This is the sum of nebular 
# and transmitted emission.

# `escaped` is the incident emission that escapes reprocessing by gas. This is fesc * incident. 
# This is not subsequently affected by dust.

# `intrinsic` is the sum of the escaped and reprocessed emission, essentially the emission before 
# dust attenuation.

# `attenuated` is the reprocessed emission with attenuation by dust.

# `dust` is the thermal dust emission calculated using an energy balance approach, 
# and assuming a dust emission model.

# `total` is the sum of attenuated and dust, i.e. 
# it includes both the effect of dust attenuation and dust emission.


# Most users will only be interested in plotting `total` SEDs, but each stage is 
# stored in the resulting `ModelGalaxy` object. Each component also stores SEDs for
# each stage, i.e. ModelGalaxy['agn']['incident']

# maybe define a `ModelComponent` class, which includes all of this infrastructure? 