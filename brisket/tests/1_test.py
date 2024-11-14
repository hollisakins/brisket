from brisket.models import StellarModel, CloudyNebularModel, FlexibleNebularModel

# create a params object
params = brisket.Params(template=None)
# you can also import a specific template, e.g.
params = brisket.Params(template='Akins24a') # for fitting LRD models



params.add_source('galaxy', StellarModel)
params['galaxy'].add_absorber('gas', CloudyNebularModel)
params['galaxy'].add_dust()
params['galaxy']['dust']['Av'] = brisket.FreeParam(low=0, high=5)
params['galaxy']['dust']['delta'] = brisket.FixedParam(0)

params.add_source('agn')
params['agn'].add_absorber('gas')
params['agn'].add_absorber('dust')


params['agn']['gas']['f_Ha'] = brisket.FreeParam(low=0, high=1e-17)
params['agn']['gas']['f_Hb'] = brisket.LinkedParam(params['agn']['gas']['f_Ha'], scale=1/2.86)

params.add_absorber()
params.add_attenuator()


# To create a more complicated model, for example, fitting multiple stellar grids 
custom_stellar_model = StellarModel(grids='bc03') + StellarModel(grids='bpass') + StellarModel(grids='yggdrasil')
# or a simpler alias 
custom_stellar_model = StellarModel(grids=['bc03','bpass','yggdrasil'])
params.add_source('galaxy', custom_stellar_model)


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