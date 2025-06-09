"""
Example of how users would interact with the brisket API
"""
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

log_stellar_mass = 10  # Example fixed value
age = 1.0  # Example fixed value in Gyr
metallicity = 1.0  # Example fixed value

# Create model components (expensive preprocessing happens here)
model = StellarModel(
    log_stellar_mass= Uniform('m1', 8, 12), 
    age=age, 
    metallicity=metallicity,
    stellar_library='bc03', 
    redshift = 5, 
)
# model += StellarModel(
#     log_stellar_mass= Uniform('m2', 8, 12), 
#     age=age, 
#     metallicity=metallicity,
#     stellar_library='bc03', 
# )

# Advanced usage
from brisket import Fitter
fitter = Fitter(model, obs, sampler='nuts', num_warmup=2000, num_samples=2000, num_chains=4)
results = fitter.run(rng_key=42)

print(results.summary())

import matplotlib.pyplot as plt
plt.hist(results.samples['m1'], bins=30)
plt.show()