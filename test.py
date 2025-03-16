import numpy as np
from unyt import Mpc, Msun, Myr, kelvin, yr

from synthesizer.emission_models import (
    AttenuatedEmission,
    BimodalPacmanEmission,
    DustEmission,
    EmissionModel,
    UnifiedAGN,
)
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emission_models.dust.emission import Blackbody, Greybody
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars, Galaxy
# from synthesizer.particle import BlackHoles, Galaxy

# # Get the grids which we'll need for extraction
# grid_dir = "../../../tests/test_grid"
# grid_name = "test_grid"
# grid = Grid(grid_name, grid_dir=grid_dir)
# nlr_grid = Grid("test_grid_agn-nlr", grid_dir="../../../tests/test_grid")
# blr_grid = Grid("test_grid_agn-blr", grid_dir="../../../tests/test_grid")

# # Get the stellar pacman model
# pc_model = BimodalPacmanEmission(
#     grid=grid,
#     tau_v_ism=1.0,
#     tau_v_birth=0.7,
#     dust_curve_ism=PowerLaw(slope=-1.3),
#     dust_curve_birth=PowerLaw(slope=-0.7),
#     dust_emission_ism=Blackbody(temperature=100 * kelvin),
#     dust_emission_birth=Blackbody(temperature=30 * kelvin),
#     fesc=0.2,
#     fesc_ly_alpha=0.9,
#     label="stellar_total",
# )


# # Get the UnifiedAGN model
# uni_model = UnifiedAGN(
#     nlr_grid,
#     blr_grid,
#     covering_fraction_nlr=0.1,
#     covering_fraction_blr=0.1,
#     torus_emission_model=Blackbody(1000 * kelvin),
#     label="agn_intrinsic",
#     ionisation_parameter=0.1,
#     hydrogen_density=1e5,
# )

# # Define an emission model to attenuate the intrinsic AGN emission
# att_uni_model = AttenuatedEmission(
#     dust_curve=PowerLaw(slope=-1.0),
#     apply_to=uni_model,
#     tau_v=0.7,
#     emitter="blackhole",
#     label="agn_attenuated",
# )

# gal_intrinsic = EmissionModel(
#     label="total_intrinsic",
#     combine=(uni_model, pc_model["intrinsic"]),
#     emitter="galaxy",
# )

# gal_attenuated = EmissionModel(
#     label="total_attenuated",
#     combine=(att_uni_model, pc_model["attenuated"]),
#     related_models=(gal_intrinsic,),
#     emitter="galaxy",
# )

# # And now include the dust emission
# gal_dust = DustEmission(
#     dust_emission_model=Greybody(30 * kelvin, 1.2),
#     dust_lum_intrinsic=gal_intrinsic,
#     dust_lum_attenuated=gal_attenuated,
#     emitter="galaxy",
#     label="dust_emission",
# )

# gal_total = EmissionModel(
#     label="total",
#     combine=(gal_attenuated, gal_dust),
#     related_models=(gal_intrinsic,),
#     emitter="galaxy",
# )
from datetime import datetime

# Get the grids which we'll need for extraction
grid_dir = "/Users/hba423/codes/synthesizer/tests/test_grid"
grid_name = "test_grid"
grid = Grid(grid_name, grid_dir=grid_dir)

start = datetime.now()
# Define the metallicity history
zh = ZDist.DeltaConstant(metallicity=0.01)

# Define the star formation history
sfh_p = {"max_age": 100 * Myr}
sfh = SFH.Constant(**sfh_p)

# Initialise the parametric Stars object
param_stars = Stars(
    grid.log10age,
    grid.metallicity,
    sf_hist=sfh,
    metal_dist=zh,
    initial_mass=10**9 * Msun,
)
# And create the galaxy
galaxy = Galaxy(
    stars=param_stars,
    black_holes=None,
    redshift=1,
)

end = datetime.now()
print(end-start)

from synthesizer.emission_models import IntrinsicEmission
model = IntrinsicEmission(grid, fesc=0.1)
print(dir(model))

# print(dir(galaxy.redshift))


# from synthesizer.grid import Grid

# # Get the grid


# from synthesizer.pipeline import Pipeline

# pipeline = Pipeline(
#     emission_model=model,
#     # instruments=instruments,
#     nthreads=1,
#     verbose=1,
# )
# # spectra = galaxy.get_spectra()
# # from astropy.cosmology import Planck18 as cosmo
# # galaxy.get_observed_spectra(cosmo)


# synthesizer distinguishes between the "emitter" and the "emission model"
# e.g., it separates parameters defined for the source of emission (stars, black holes, etc), 
# which are predicted by the simulation, from parameters that describe how that 
# emission propagates (e.g., dust attenuation, covering fraction, etc).


# pre-defined instrument models 


from unyt import kelvin

from synthesizer.emission_models import (
    AttenuatedEmission,
    BimodalPacmanEmission,
    DustEmission,
    EmissionModel,
    UnifiedAGN,
)
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emission_models.dust.emission import Blackbody, Greybody
from synthesizer.grid import Grid

params = {}
params['emission_models'] = {
    'BimodalPacmanEmission': {
        'grid': grid, 
        'fesc': Uniform(0.0, 0.2), 
        'dust_curve_ism': {
            'PowerLaw': {'slope': Uniform(-2.0, -1.0)}
            },
        'dust_curve_birth': {
            'PowerLaw': {'slope': -0.7}
            },
        }
}
params.add_emitter(Stars, 
)

params.add_emission_model(BimodalPacmanEmission, 
    grid = grid, 
    fesc = Uniform(0.0, 0.2),
    dust_curve_ism = PowerLaw(slope=Uniform(-2.0, -1.0)),

def ModelA():
    # Get the grids which we'll need for extraction
    grid_dir = "../../../tests/test_grid"
    grid_name = "test_grid"
    grid = Grid(grid_name, grid_dir=grid_dir)
    nlr_grid = Grid("test_grid_agn-nlr", grid_dir="../../../tests/test_grid")
    blr_grid = Grid("test_grid_agn-blr", grid_dir="../../../tests/test_grid")

    # Get the stellar pacman model
    pc_model = BimodalPacmanEmission(
        grid=grid,
        tau_v_ism=1.0,
        tau_v_birth=0.7,
        dust_curve_ism=PowerLaw(slope=-1.3),
        dust_curve_birth=PowerLaw(slope=-0.7),
        dust_emission_ism=Blackbody(temperature=100 * kelvin),
        dust_emission_birth=Blackbody(temperature=30 * kelvin),
        fesc=0.2,
        fesc_ly_alpha=0.9,
        label="stellar_total",
    )
    pc_model.plot_emission_tree(fontsize=5)

    # Get the UnifiedAGN model
    uni_model = UnifiedAGN(
        nlr_grid,
        blr_grid,
        covering_fraction_nlr=0.1,
        covering_fraction_blr=0.1,
        torus_emission_model=Blackbody(1000 * kelvin),
        label="agn_intrinsic",
    )

    # Define an emission model to attenuate the intrinsic AGN emission
    att_uni_model = AttenuatedEmission(
        dust_curve=PowerLaw(slope=-1.0),
        apply_to=uni_model,
        tau_v=0.7,
        emitter="blackhole",
        label="agn_attenuated",
    )

    # And now include the dust emission
    dust_uni_model = DustEmission(
        dust_emission_model=Greybody(30 * kelvin, 1.2),
        dust_lum_intrinsic=uni_model,
        dust_lum_attenuated=att_uni_model,
        emitter="blackhole",
        label="agn_dust_emission",
    )
    dust_uni_model.plot_emission_tree(fontsize=7)