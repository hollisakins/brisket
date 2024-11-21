'''
This module contains all of the configuration variables for brisket.
This includes specifying the cosmology, loading different grids of 
models into the code, and the specifying the way the spectral sampling 
for models is constructed. For the latter, the default values are a 
trade-off between speed and accuracy, you may find changing them 
negatively affects one or both of these things.
'''

import os
import numpy as np

from astropy.io import fits
from astropy.units import cm

########################## Set up directory structure ##########################
install_dir = os.path.dirname(os.path.realpath(__file__))
'''Stores the install directory for easy reference'''

grid_dir = install_dir + "/data"
'''Stores the path to the model grid directory for easy reference'''

res_dir = install_dir + "/models/res"
'''Stores the path to the resolution curve directory for easy reference'''

filter_directory = install_dir + '/filters/filter_files/filter_directory.toml'
'''Stores the path to the filter directory for easy reference'''


ascii = False
'''Whether to use ascii characters only in printing fit outputs. Defaults to false (i.e., use fancier unicode characters)'''

from astropy.cosmology import Planck18 as cosmo
'''The cosmology used in the code. Defaults to Planck 2018 cosmology, but can be changed to any astropy cosmology object.'''

# These variables control the wavelength sampling for models.
R_spec = 1000.
'''Sets the R = lambda/dlambda value for spectroscopic outputs.'''
R_phot = 100.
'''Sets the R value for photometric regions.'''
R_other = 50.
'''Sets the R value for other spectral regions.'''

R_curves = {}
r = fits.getdata(os.path.join(res_dir,'jwst_nirspec_prism_disp.fits'))
R_curves['JWST_NIRSpec_PRISM'] = np.array([r['WAVELENGTH']*1e4, r['R']]).T
r = fits.getdata(os.path.join(res_dir,'jwst_nirspec_g395m_disp.fits'))
R_curves['JWST_NIRSpec_G395M'] = np.array([r['WAVELENGTH']*1e4, r['R']]).T
#TODO add the rest of the JWST resolution curves


fwhm = 500
'''Default FWHM of nebular lines, in km/s (TODO: is this implemented?)'''

max_wavelength = 1 * cm
'''Maximum wavelength at which the full SED models are computed.'''

max_redshift = 20.
'''Sets the maximum redshift the code is set up to calculate models for.'''


import shutil
cols = shutil.get_terminal_size((80, 20)).columns

if ascii:
    value_chars = ' ░▒▓█'
    border_chars = '═║╔╦╗╠╬╣╚╩╝'
else:
    value_chars = ' ▁▂▃▄▅▆▇█'
    border_chars = '═║╔╦╗╠╬╣╚╩╝'

### units
import astropy.units as u
default_wavelength_unit=u.angstrom
default_frequency_unit=u.GHz
default_energy_unit=u.keV
default_fnu_unit=u.uJy
default_flam_unit=u.erg/u.s/u.cm**2/u.angstrom
default_Llam_unit = u.Lsun/u.angstrom
default_Lnu_unit = u.Lsun/u.Hz
default_L_unit = u.Lsun

params_print_summary = True
params_print_tree = False

sfh_age_log_sampling=0.0025

# # These variables tell the code where to find the raw stellar emission
# # models, as well as some of their basic properties.
# stellar_models = {}
# '''Dictionary storing the stellar model grids.'''

# try:
#     # Name of the fits file storing the stellar models
#     stellar_file = 'bc03_miles_stellar_grids.fits'
    
#     stellar_models['BC03'] = {}
#     # The metallicities of the stellar grids in units of Z_Solar
#     stellar_models['BC03']['metallicities'] = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])
#     # The wavelengths of the grid points in Angstroms
#     stellar_models['BC03']['wavelengths'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-1)
#     # The ages of the grid points in Gyr
#     stellar_models['BC03']['raw_stellar_ages'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-2)
#     # The fraction of stellar mass still living (1 - return fraction).
#     # Axis 0 runs over metallicity, axis 1 runs over age.
#     stellar_models['BC03']['live_frac'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-3)[:, 1:]
#     # The raw stellar grids, stored as a FITS HDUList.
#     # The different HDUs are the grids at different metallicities.
#     # Axis 0 of each grid runs over wavelength, axis 1 over age.
#     stellar_models['BC03']['raw_stellar_grid'] = [fits.getdata(os.path.join(grid_dir,stellar_file),ext=i,memmap=False) for i in range(1,8)]
#     # Set up edge positions for metallicity bins for stellar models.
#     metallicity_bins = make_bins(stellar_models['BC03']['metallicities'], make_rhs=True)[0]
#     metallicity_bins[0] = 0.
#     metallicity_bins[-1] = 10.
#     stellar_models['BC03']['metallicity_bins'] = metallicity_bins
# except IOError:
#     print("Failed to load BC03 stellar grids, these should be placed in the brisket/models/grids/ directory.")

# # try:
#     # Name of the fits file storing the stellar models
#     stellar_file = 'bpass_2.2.1_bin_imf135_300_stellar_grids.fits'
    
#     stellar_models['BPASS'] = {}
#     # The metallicities of the stellar grids in units of Z_Solar
#     stellar_models['BPASS']['metallicities'] = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3,
#                               6e-3, 8e-3, 1e-2, 0.014, 0.020, 0.030,
#                               0.040])/0.02
#     # The wavelengths of the grid points in Angstroms
#     stellar_models['BPASS']['wavelengths'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-1)
#     # The ages of the grid points in Gyr
#     stellar_models['BPASS']['raw_stellar_ages'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-2)
#     # The fraction of stellar mass still living (1 - return fraction).
#     # Axis 0 runs over metallicity, axis 1 runs over age.
#     stellar_models['BPASS']['live_frac'] = fits.getdata(os.path.join(grid_dir,stellar_file),ext=-3)
#     # The raw stellar grids, stored as a FITS HDUList.
#     # The different HDUs are the grids at different metallicities.
#     # Axis 0 of each grid runs over wavelength, axis 1 over age.
#     stellar_models['BPASS']['raw_stellar_grid'] = [fits.getdata(os.path.join(grid_dir,stellar_file),ext=i,memmap=False) for i in range(1,14)]
#     # Set up edge positions for metallicity bins for stellar models.
#     metallicity_bins = make_bins(stellar_models['BPASS']['metallicities'], make_rhs=True)[0]
#     metallicity_bins[0] = 0.
#     metallicity_bins[-1] = 10.
#     stellar_models['BPASS']['metallicity_bins'] = metallicity_bins
# except IOError:
#     print("Failed to load BPASS stellar grids, these should be placed in the brisket/models/grids/ directory.")


# #These variables tell the code where to find the raw nebular emission
# # models, as well as some of their basic properties. 
# # Note that the nebular emission models are tied to the stellar templates used

# try:
#     # Names for the emission features to be tracked.
#     line_names = np.loadtxt(grid_dir + "/cloudy_lines.txt",
#                             dtype="str", delimiter="}")

#     # Wavelengths of these emission features in Angstroms.
#     line_wavs = np.loadtxt(grid_dir + "/cloudy_linewavs.txt")
# except IOError:
#     print('Failed to load cloudy_lines.txt and cloudy_linewavs.txt')

# nebular_models = {}
# '''Dictionary storing the nebular model grids.'''

# try:
#     # Names of files containing the nebular grids.
#     neb_cont_file = "bc03_miles_nebular_cont_grids.fits"
#     neb_line_file = "bc03_miles_nebular_line_grids.fits"
#     nebular_models['BC03'] = {}
#     # Ages for the nebular emission grids.
#     nebular_models['BC03']['neb_ages'] = fits.open(os.path.join(grid_dir,neb_line_file))[1].data[1:, 0]
#     # Wavelengths for the nebular continuum grids.
#     nebular_models['BC03']['neb_wavs'] = fits.open(os.path.join(grid_dir, neb_cont_file))[1].data[0, 1:]
#     # LogU values for the nebular emission grids.
#     nebular_models['BC03']['logU'] = np.arange(-4., -0.99, 0.5)
#     # Grid of line fluxes.
#     nebular_models['BC03']['line_grid'] = [fits.getdata(os.path.join(grid_dir,neb_line_file),ext=i,memmap=False) for i in range(1,len(stellar_models['BC03']['metallicities']) * len(nebular_models['BC03']['logU']) + 1)]
#     # Grid of nebular continuum fluxes.
#     nebular_models['BC03']['cont_grid'] = [fits.getdata(os.path.join(grid_dir,neb_cont_file),ext=i,memmap=False) for i in range(1,len(stellar_models['BC03']['metallicities']) * len(nebular_models['BC03']['logU']) + 1)]
    

# except IOError:
#     print('Failed to load BC03 nebular grids, these should be placed in the brisket/models/grids/ directory.')

# try:
#     # Names of files containing the nebular grids.
#     neb_cont_file = "bpass_2.2.1_bin_imf135_300_nebular_cont_grids.fits"
#     neb_line_file = "bpass_2.2.1_bin_imf135_300_nebular_line_grids.fits"
#     nebular_models['BPASS'] = {}
#     # Ages for the nebular emission grids.
#     nebular_models['BPASS']['neb_ages'] = fits.open(os.path.join(grid_dir,neb_line_file))[1].data[1:, 0]
#     # Wavelengths for the nebular continuum grids.
#     nebular_models['BPASS']['neb_wavs'] = fits.open(os.path.join(grid_dir, neb_cont_file))[1].data[0, 1:]
#     # LogU values for the nebular emission grids.
#     nebular_models['BPASS']['logU'] = np.arange(-4., -0.99, 0.5)
#     # Grid of line fluxes.
#     nebular_models['BPASS']['line_grid'] = [fits.getdata(os.path.join(grid_dir,neb_line_file),ext=i,memmap=False) for i in range(1,len(stellar_models['BPASS']['metallicities']) * len(nebular_models['BPASS']['logU']) + 1)]
#     # Grid of nebular continuum fluxes.
#     nebular_models['BPASS']['cont_grid'] = [fits.getdata(os.path.join(grid_dir,neb_cont_file),ext=i,memmap=False) for i in range(1,len(stellar_models['BPASS']['metallicities']) * len(nebular_models['BPASS']['logU']) + 1)]

# except IOError:
#     print('Failed to load BPASS nebular grids, these should be placed in the brisket/models/grids/ directory.')


# # These variables tell the code where to find the raw dust emission
# # models, as well as some of their basic properties.

# try:
#     # Values of Umin for each of the Draine + Li (2007) dust emission grids.
#     umin_vals = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00,
#                           1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00,
#                           10.0, 12.0, 15.0, 20.0, 25.0])

#     # Values of qpah for each of the Draine + Li (2007) dust emission grids.
#     qpah_vals = np.array([0.10, 0.47, 0.75, 1.12, 1.49, 1.77,
#                           2.37, 2.50, 3.19, 3.90, 4.58])

#     # Draine + Li (2007) dust emission grids, stored as a FITS HDUList.
#     dust_grid_umin_only = fits.open(grid_dir + "/dl07_grids_umin_only.fits")

#     dust_grid_umin_umax = fits.open(grid_dir + "/dl07_grids_umin_umax.fits")

# except IOError:
#     print("Failed to load dust emission grids, these should be placed in the"
#           + " bagpipes/models/grids/ directory.")

# # These variables tell the code where to find the raw IGM attenuation
# # models, as well as some of their basic properties.

# # Redshift points for the IGM grid.
# igm_redshifts = np.arange(0.0, max_redshift + 0.01, 0.01)

# # Wavelength points for the IGM grid.
# igm_wavelengths = np.arange(1.0, 1225.01, 1.0)

# try:
#     # If the IGM grid has not yet been calculated, calculate it now.
#     if not os.path.exists(grid_dir + "/d_igm_grid_inoue14.fits"):
#         igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

#     else:
#         # Check that the wavelengths and redshifts in the igm file are right
#         igm_file = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")

#         if len(igm_file) != 4:
#             igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

#         else:
#             wav_check = np.min(igm_file[2].data == igm_wavelengths)
#             z_check = np.min(igm_file[3].data == igm_redshifts)

#             if not wav_check or not z_check:
#                 igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

#     # 2D numpy array containing the IGM attenuation grid.
#     raw_igm_grid = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")[1].data
# except:
#     pass




# try:
#     qsogen_wnrm = 5500.
#     qsogen_ext_curve = f'{grid_dir}/pl_ext_comp_03.sph'

#     qsogen_blr = np.loadtxt(os.path.join(grid_dir, 't21_blr_adj_all.txt'), usecols=1)
#     qsogen_blr_lya = np.loadtxt(os.path.join(grid_dir, 't21_blr_adj_lya_only.txt'), usecols=1)
#     qsogen_nlr = np.loadtxt(os.path.join(grid_dir, 't21_nlr_adj_all.txt'), usecols=1)
#     qsogen_nlr_oiii = np.loadtxt(os.path.join(grid_dir, 't21_nlr_adj_oiii_only.txt'), usecols=1)
#     qsogen_conval = np.loadtxt(os.path.join(grid_dir, 't21_cont.txt'), usecols=1)
#     qsogen_wavelengths = np.loadtxt(os.path.join(grid_dir, 't21_cont.txt'), usecols=0)
# except:
#     print('Failed to load QSOGEN BLR/NLR models, these should be placed in the brisket/models/grids/ directory.')


