"""
Example of how users would interact with the brisket API
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpyro
numpyro.enable_x64()
import matplotlib.pyplot as plt
from brisket import Photometry

obs = Photometry(
    name='jwst',
    filters=['f115w', 'f200w', 'f444w'],
    fnu = [4e7, 2e8, 1e8], 
    fnu_err = [1e7, 1e7, 1e7],
)

from brisket.models.stellar import StellarModel
from brisket.models.spectrum import EmissionLine
from brisket.parameters import Uniform

# Create model components (expensive preprocessing happens here)
model = StellarModel(
    log_stellar_mass = Uniform('mass', 8, 12), 
    age = 1.0, 
    metallicity = 1.0,
    stellar_library='bc03', 
    redshift = 5, 
)
model += StellarModel(
    log_stellar_mass = Uniform('mass2', 8, 12), 
    age = 1.0, 
    metallicity = 1.0,
    stellar_library='bc03', 
)

print(model.parameters)
print(model.parameter_manager)

# Advanced usage
from brisket import Fitter
fitter = Fitter(model, obs, sampler='nuts', num_warmup=2000, num_samples=2000, num_chains=1)
results = fitter.run(rng_key=42)

print(results.summary())

import matplotlib.pyplot as plt
plt.scatter(results.samples['mass'], results.samples['mass2'])  # Use Parameter's name
plt.show()