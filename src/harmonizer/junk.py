'''This module makes it possible to fit synthesizer parametric models to 
   observations using a Bayesian framework.
'''

import numpy as np
from unyt import Myr

from synthesizer.parametric import SFH, Stars, ZDist


metal_dist = ZDist.DeltaConstant(log10metallicity=-2.5)
print(metal_dist)


# Define a constant SFH
sfh = SFH.Constant(100 * Myr)
print(sfh)

# Create the Stars object
const_stars = Stars(
    grid.log10age,
    grid.metallicity,
    sf_hist=sfh,
    metal_dist=metal_dist,
    initial_mass=10**9 * Msun,
)

# And create the galaxy
galaxy = Galaxy(
    stars=stars,
    black_holes=blackholes,
    redshift=1,
)


# write class FreeParam that can be interpreted as a unyt quantity so as to not break the code? 
# essentially, something that can scrape through the __init__ methods for the emitters 
# can be replaced with a unyt quantity at the emissionmodel stage

params = {}
params['redshift'] = Uniform(0, 10)
params['stars'] = {
    'grid': 'grid_name',
    'sfh': 'sfh_name',
    'zdist': 'metal_dist_name',
}

from synthesizer.fitter import Galaxy, Stars, SFH
from synthesizer.fitter.priors import Uniform, LogUniform, Gaussian


from synthesizer.parametric import Stars as ParametricStars
class Stars(ParametricStars):
    '''Wrapper around synthesizer.parametric.Stars to specify the 
       stellar grids by file name rather than the specific ages/metallicities
    '''

    @accepts(initial_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        grid_file,
        initial_mass=None,
        morphology=None,
        sfzh=None,
        sf_hist=None,
        metal_dist=None,
        fesc=None,
        fesc_ly_alpha=None,
        **kwargs):

    # initial_mass, fesc, fesc_ly_alpha all need to be able to be taken as priors

    # Instantiate the parent
    super().__init__(
        log10ages,
        metallicities,
        initial_mass=initial_mass,
        morphology=morphology,
        sfzh=sfzh,
        sf_hist=sf_hist,
        metal_dist=metal_dist,
        fesc=fesc,
        fesc_ly_alpha=fesc_ly_alpha,
        **kwargs,
    )


stars = Stars(
    grid_file = 'grid_name',
    sf_hist = SFH.Constant(
        age_min = 0*Myr, 
        age_max = Uniform(low=100*Myr, high=500*Myr),
    ),
    metal_dist = ZDist.DeltaConstant(
        log10metallicity = Uniform(low=-2.5, high=-1.5),
    ),
    initial_mass = LogUniform(low=1e9*Msun, high=1e11*Msun),
)

galaxy = Galaxy(
    redshift = Uniform(0, 10),
    stars = stars
)



from synthesizer.fitter import Fitter
f = Fitter(galaxy, observation)

