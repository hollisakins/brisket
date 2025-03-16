
import numpy as np
from unyt import Msun, Myr
from synthesizer import Grid
from synthesizer.emission_models import IncidentEmission
from synthesizer.parametric import SFH, Stars, ZDist

grid = Grid("bpass-2.3-bin_chabrier03-0.1,300.0", grid_dir="/Users/hba423/Downloads/", read_lines=False)
print(grid.spectra['incident'].shape)
a = 'alpha_enhancement'
x = 0.35
v = grid._axes_values[a]

if np.min(np.abs(v - x)) < 1e-5:
    # x is (essentially) already in the grid, so just collapse to that value
    pass

else:
    i0 = np.argmax(v[v <= x])
    i1 = i0 + 1
    x0 = v[i0]
    x1 = v[i1]
    print(x0, x1)
    c0 = (x1 - x)/(x1 - x0)
    c1 = (x - x0)/(x1 - x0)
    print(c0, c1)


print(grid.spectra['incident'])


quit()


# stars = Stars(
#     grid.log10age,
#     grid.metallicity,
#     sf_hist = SFH.Constant(max_age = 100 * Myr),
#     metal_dist = ZDist.Normal(mean = 0.01, sigma = 0.05),
#     initial_mass = 1e11 * Msun,
# )
# emission_model = IncidentEmission(grid=grid)
# spectra = stars.get_spectra(emission_model)



def get_spectra(mass, max_age, metallicity, logU): 
    grid = Grid("hypothetical_grid")
    grid.interpolate_axis('logU', logU) # interpolate the entire grid to the desired logU

    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist = SFH.Constant(max_age = max_age),
        metal_dist = ZDist.DeltaConstant(metallicity=metallicity),
        initial_mass = mass,
    )
    emission_model = IncidentEmission(grid=grid)
    return stars.get_spectra(emission_model)
from synthesizer import Grid
from synthesizer.emission_models import UnifiedAGN
from synthesizer.emission_models.dust.emission import Greybody
from synthesizer.parametric import BlackHole
import numpy as np
from unyt import K, Mpc, Msun, deg, yr

import matplotlib.pyplot as plt
plt.style.use('hba_agg')



# Get the NLR and BLR grids
nlr_grid = Grid("test_grid_agn-nlr", grid_dir="/Users/hba423/codes/synthesizer/tests/test_grid")
blr_grid = Grid("test_grid_agn-blr", grid_dir="/Users/hba423/codes/synthesizer/tests/test_grid")

fig, ax = plt.subplots()



blackhole = BlackHole(
    mass = 1e8 * Msun,
    inclination = 60 * deg,
    accretion_rate = 1 * Msun / yr,
    metallicity = 0.01)
uniagn = UnifiedAGN(
    nlr_grid,
    blr_grid,
    covering_fraction_nlr=0.1,
    covering_fraction_blr=0.1,
    torus_emission_model=Greybody(1000 * K, 1.5),
    ionisation_parameter=0.01,
    hydrogen_density=1e5)

spectra1 = blackhole.get_spectra(uniagn)

ax.loglog(spectra1.lam, spectra1.lnu, label='ionization_parameter = 0.01')

blackhole = BlackHole(
    mass = 0.95e8 * Msun,
    inclination = 60 * deg,
    accretion_rate = 1 * Msun / yr,
    metallicity = 0.01)
uniagn = UnifiedAGN(
    nlr_grid,
    blr_grid,
    covering_fraction_nlr=0.1,
    covering_fraction_blr=0.1,
    torus_emission_model=Greybody(1000 * K, 1.5),
    ionisation_parameter=0.07,
    hydrogen_density=1e5)

spectra2 = blackhole.get_spectra(uniagn)

ax.loglog(spectra2.lam, spectra2.lnu, label='ionization_parameter = 0.07')

blackhole = BlackHole(
    mass = 0.95e8 * Msun,
    inclination = 60 * deg,
    accretion_rate = 1 * Msun / yr,
    metallicity = 0.01)
uniagn = UnifiedAGN(
    nlr_grid,
    blr_grid,
    covering_fraction_nlr=0.1,
    covering_fraction_blr=0.1,
    torus_emission_model=Greybody(1000 * K, 1.5),
    ionisation_parameter=0.1,
    hydrogen_density=1e5)

spectra3 = blackhole.get_spectra(uniagn)

ax.loglog(spectra3.lam, spectra3.lnu, label='ionization_parameter = 0.1')


x = 0.07
x0 = 0.01
x1 = 0.1
y0 = spectra1.lnu 
y1 = spectra3.lnu
y = (y0*(x1-x) + y1*(x-x0))/(x1-x0)

ax.loglog(spectra3.lam, y, label='linearly interpolated')



ax.set_xlim(400, 30000)
ax.set_ylim(5e28, 4e31)
ax.legend()
plt.show()


# intrinsic = spectra['intrinsic']
# print(intrinsic)

# params = harmonizer.Params()
# params['redshift'] = 7


# stars = params.add_stars()
# stars['grid'] = 'bc03'
# stars['logMstar'] = 10 # Uniform(8, 11)
# stars['zmet'] = 1
# stars.add_nebular()
# dust = stars.add_dust_attenuation()
# dust['dust_curve'] = 'Calzetti'
# dust['dust_curve'] = Powerlaw
# stars.add_dust_emission(dust_emission_model=synthesizer.emission_models.dust.emission.Blackbody)
# sfh = stars.add_sfh('constant', sfh_function=synthesizer.parametric.SFH.Constant)
# sfh['age_min'] = 0.1 # Uniform(0.05, 0.15)
# sfh['age_max'] = 0.4 # Uniform(0.4, 0.6)


# params.add_emitter('agn', ) # UnifiedAGN
# params.stars['nlr_grid'] = ...
# params.stars['blr_grid'] = ...
# params.stars.add_photoionization() # whether to include BLR/NLR emission
# params.stars.add_dust_attenuation() # -> separate attenuation emissionmodel
# params.stars.add_dust_emission() # -> torus, +reprocessing of attenuating dust


# import numpy as np
# from unyt import Mpc, Msun, Myr, kelvin, yr

# from synthesizer.emission_models import (
#     AttenuatedEmission,
#     BimodalPacmanEmission,
#     DustEmission,
#     EmissionModel,
#     UnifiedAGN,
# )
# from synthesizer.emission_models.attenuation import PowerLaw
# from synthesizer.emission_models.dust.emission import Blackbody, Greybody
# from synthesizer.grid import Grid
# from synthesizer.parametric import SFH, ZDist
# from synthesizer.parametric import Stars, Galaxy
# # from synthesizer.particle import BlackHoles, Galaxy

# # # Get the grids which we'll need for extraction
# # grid_dir = "../../../tests/test_grid"
# # grid_name = "test_grid"
# # grid = Grid(grid_name, grid_dir=grid_dir)
# # nlr_grid = Grid("test_grid_agn-nlr", grid_dir="../../../tests/test_grid")
# # blr_grid = Grid("test_grid_agn-blr", grid_dir="../../../tests/test_grid")

# # # Get the stellar pacman model
# # pc_model = BimodalPacmanEmission(
# #     grid=grid,
# #     tau_v_ism=1.0,
# #     tau_v_birth=0.7,
# #     dust_curve_ism=PowerLaw(slope=-1.3),
# #     dust_curve_birth=PowerLaw(slope=-0.7),
# #     dust_emission_ism=Blackbody(temperature=100 * kelvin),
# #     dust_emission_birth=Blackbody(temperature=30 * kelvin),
# #     fesc=0.2,
# #     fesc_ly_alpha=0.9,
# #     label="stellar_total",
# # )


# # # Get the UnifiedAGN model
# # uni_model = UnifiedAGN(
# #     nlr_grid,
# #     blr_grid,
# #     covering_fraction_nlr=0.1,
# #     covering_fraction_blr=0.1,
# #     torus_emission_model=Blackbody(1000 * kelvin),
# #     label="agn_intrinsic",
# #     ionisation_parameter=0.1,
# #     hydrogen_density=1e5,
# # )

# # # Define an emission model to attenuate the intrinsic AGN emission
# # att_uni_model = AttenuatedEmission(
# #     dust_curve=PowerLaw(slope=-1.0),
# #     apply_to=uni_model,
# #     tau_v=0.7,
# #     emitter="blackhole",
# #     label="agn_attenuated",
# # )

# # gal_intrinsic = EmissionModel(
# #     label="total_intrinsic",
# #     combine=(uni_model, pc_model["intrinsic"]),
# #     emitter="galaxy",
# # )

# # gal_attenuated = EmissionModel(
# #     label="total_attenuated",
# #     combine=(att_uni_model, pc_model["attenuated"]),
# #     related_models=(gal_intrinsic,),
# #     emitter="galaxy",
# # )

# # # And now include the dust emission
# # gal_dust = DustEmission(
# #     dust_emission_model=Greybody(30 * kelvin, 1.2),
# #     dust_lum_intrinsic=gal_intrinsic,
# #     dust_lum_attenuated=gal_attenuated,
# #     emitter="galaxy",
# #     label="dust_emission",
# # )

# # gal_total = EmissionModel(
# #     label="total",
# #     combine=(gal_attenuated, gal_dust),
# #     related_models=(gal_intrinsic,),
# #     emitter="galaxy",
# # )
# from datetime import datetime

# # Get the grids which we'll need for extraction
# grid_dir = "/Users/hba423/codes/synthesizer/tests/test_grid"
# grid_name = "test_grid"
# grid = Grid(grid_name, grid_dir=grid_dir)

# start = datetime.now()
# # Define the metallicity history
# zh = ZDist.DeltaConstant(metallicity=0.01)

# # Define the star formation history
# sfh_p = {"max_age": 100 * Myr}
# sfh = SFH.Constant(**sfh_p)

# # Initialise the parametric Stars object
# param_stars = Stars(
#     grid.log10age,
#     grid.metallicity,
#     sf_hist=sfh,
#     metal_dist=zh,
#     initial_mass=10**9 * Msun,
# )
# # And create the galaxy
# galaxy = Galaxy(
#     stars=param_stars,
#     black_holes=None,
#     redshift=1,
# )

# end = datetime.now()
# print(end-start)

# from synthesizer.emission_models import IntrinsicEmission
# model = IntrinsicEmission(grid, fesc=0.1)
# print(dir(model))

# # print(dir(galaxy.redshift))


# # from synthesizer.grid import Grid

# # # Get the grid


# # from synthesizer.pipeline import Pipeline

# # pipeline = Pipeline(
# #     emission_model=model,
# #     # instruments=instruments,
# #     nthreads=1,
# #     verbose=1,
# # )
# # # spectra = galaxy.get_spectra()
# # # from astropy.cosmology import Planck18 as cosmo
# # # galaxy.get_observed_spectra(cosmo)


# # synthesizer distinguishes between the "emitter" and the "emission model"
# # e.g., it separates parameters defined for the source of emission (stars, black holes, etc), 
# # which are predicted by the simulation, from parameters that describe how that 
# # emission propagates (e.g., dust attenuation, covering fraction, etc).


# # pre-defined instrument models 


# from unyt import kelvin

# from synthesizer.emission_models import (
#     AttenuatedEmission,
#     BimodalPacmanEmission,
#     DustEmission,
#     EmissionModel,
#     UnifiedAGN,
# )
# from synthesizer.emission_models.attenuation import PowerLaw
# from synthesizer.emission_models.dust.emission import Blackbody, Greybody
# from synthesizer.grid import Grid

# params = {}
# params['emission_models'] = {
#     'BimodalPacmanEmission': {
#         'grid': grid, 
#         'fesc': Uniform(0.0, 0.2), 
#         'dust_curve_ism': {
#             'PowerLaw': {'slope': Uniform(-2.0, -1.0)}
#             },
#         'dust_curve_birth': {
#             'PowerLaw': {'slope': -0.7}
#             },
#         }
# }
# params.add_emitter(Stars, 
# )

# params.add_emission_model(BimodalPacmanEmission, 
#     grid = grid, 
#     fesc = Uniform(0.0, 0.2),
#     dust_curve_ism = PowerLaw(slope=Uniform(-2.0, -1.0)),

# def ModelA():
#     # Get the grids which we'll need for extraction
#     grid_dir = "../../../tests/test_grid"
#     grid_name = "test_grid"
#     grid = Grid(grid_name, grid_dir=grid_dir)
#     nlr_grid = Grid("test_grid_agn-nlr", grid_dir="../../../tests/test_grid")
#     blr_grid = Grid("test_grid_agn-blr", grid_dir="../../../tests/test_grid")

#     # Get the stellar pacman model
#     pc_model = BimodalPacmanEmission(
#         grid=grid,
#         tau_v_ism=1.0,
#         tau_v_birth=0.7,
#         dust_curve_ism=PowerLaw(slope=-1.3),
#         dust_curve_birth=PowerLaw(slope=-0.7),
#         dust_emission_ism=Blackbody(temperature=100 * kelvin),
#         dust_emission_birth=Blackbody(temperature=30 * kelvin),
#         fesc=0.2,
#         fesc_ly_alpha=0.9,
#         label="stellar_total",
#     )
#     pc_model.plot_emission_tree(fontsize=5)

#     # Get the UnifiedAGN model
#     uni_model = UnifiedAGN(
#         nlr_grid,
#         blr_grid,
#         covering_fraction_nlr=0.1,
#         covering_fraction_blr=0.1,
#         torus_emission_model=Blackbody(1000 * kelvin),
#         label="agn_intrinsic",
#     )

#     # Define an emission model to attenuate the intrinsic AGN emission
#     att_uni_model = AttenuatedEmission(
#         dust_curve=PowerLaw(slope=-1.0),
#         apply_to=uni_model,
#         tau_v=0.7,
#         emitter="blackhole",
#         label="agn_attenuated",
#     )

#     # And now include the dust emission
#     dust_uni_model = DustEmission(
#         dust_emission_model=Greybody(30 * kelvin, 1.2),
#         dust_lum_intrinsic=uni_model,
#         dust_lum_attenuated=att_uni_model,
#         emitter="blackhole",
#         label="agn_dust_emission",
#     )
#     dust_uni_model.plot_emission_tree(fontsize=7)