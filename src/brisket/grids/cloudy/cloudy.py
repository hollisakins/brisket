import numpy as np
import os
from astropy.io import fits
from ..console import setup_logger
from .grids import Grid
from .. import config

if "CLOUDY_DATA_PATH" in list(os.environ):
    cloudy_data_path = os.environ["CLOUDY_DATA_PATH"]
    cloudy_sed_dir = os.path.join(cloudy_data_path, 'SED')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1

def mpi_split_array(array):
    """ Distributes array elements to cores when using mpi. """
    if size > 1: # If running on more than one core

        n_per_core = array.shape[0]//size

        # How many are left over after division between cores
        remainder = array.shape[0]%size

        if rank == 0:
            if remainder == 0:
                core_array = array[:n_per_core, ...]

            else:
                core_array = array[:n_per_core+1, ...]

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                comm.send(array[start:stop, ...], dest=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                comm.send(array[start:stop, ...], dest=i)

        if rank != 0:
            core_array = comm.recv(source=0)

    else:
        core_array = array

    return core_array


def mpi_combine_array(core_array, total_len):
    """ Combines array sections from different cores. """
    if size > 1: # If running on more than one core

        n_per_core = total_len//size

        # How many are left over after division between cores
        remainder = total_len%size

        if rank != 0:
            comm.send(core_array, dest=0)
            array = None

        if rank == 0:
            array = np.zeros([total_len] + list(core_array.shape[1:]))
            array[:core_array.shape[0], ...] = core_array

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                array[start:stop, ...] = comm.recv(source=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                array[start:stop, ...] = comm.recv(source=i)

        array = comm.bcast(array, root=0)

    else:
        array = core_array

    return array

def logQ_from_logU(logU, lognH, logr):
    '''Compute logQ from logU, log(nH/cm^-3), and log(radius/pc). Osterbrok & Ferland eq. 14.7.'''
    U = np.power(10., logU)
    nH = np.power(10., lognH)
    r = np.power(10., logr) * 3.086e18  # pc -> cm
    c = 2.99e10  # cm s^-1
    return np.log10(4*np.pi * r**2 * c * nH * U)

def make_cloudy_input_file(dir, filename: str, params: dict):
    """Generates in input parameter file for cloudy. Much of this code is adapted from synthesizer"""

    # # Copy file with emission line names to the correct directory
    # if not os.path.exists(cloudy_data_path + "/brisket_lines.txt"):
    #     os.system("cp " + utils.install_dir + "/models/grids/cloudy_lines.txt "
    #               + cloudy_data_path + "/pipes_cloudy_lines.txt")

    f = open(f"{dir}/{filename}.in", "w+")
    # input ionizing spectrum
    f.write(f"table SED \"{filename}.sed\"\n")

    zmet = params['zmet'] # metallicity in solar units
    CO = params['CO'] # C/O ratio, relative to solar
    xid = params['xid'] # dust-to-gas ratio 
    lognH = params['lognH'] # log10 of hydrogen density in cm^-3
    logU = params['logU'] # log10 of ionization parameter

    
    if params['geometry'] == 'spherical':
        radius = params['radius']
        logr = np.log10(radius)
        logQ = logQ_from_logU(logU, lognH, logr)
        f.write("sphere\n")
        f.write(f"radius {logr:.3f} log parsecs\n")
    else:
        raise NotImplementedError("Only spherical geometry is currently supported")

    if params["cosmic_rays"] is not None:
        f.write("cosmic rays background\n")

    if params["CMB"] is not None:
        f.write(f'CMB {params["z"]}\n')

    f.write(f"hden {lognH:.3f} log\n")
    f.write(f"Q(H) = {logQ:.3f} log\n")
    
    # # constant density flag
    # if params["constant_density"] is not None:
    #     cinput.append("constant density\n")

    # # constant pressure flag
    # if params["constant_pressure"] is not None:
    #     cinput.append("constant pressure\n")

    # if (params["constant_density"] is not None) and (
    #     params["constant_pressure"] is not None
    # ):
    #     raise InconsistentArguments(
    #         """Cannot specify both constant pressure and density"""
    #     )

    # # covering factor
    # if params["covering_factor"] is not None:
    #     cinput.append(f'covering factor {params["covering_factor"]} linear\n')



    #######################################################################################################################################
    # Chemical composition ################################################################################################################
    #######################################################################################################################################
    if not 'abundances' in params:
        # set the default abundance model
        params['abundances'] = {'model': 'Gutkin16'}

    if params['abundances']['model'] == 'Gutkin16':
        zsol = 0.01508
        numbers = np.arange(1, 31)
        masses = np.array([1.0080, 4.00260,7.0,9.012183,10.81,12.011,14.007,15.999,18.99840316,20.180,22.9897693,24.305,26.981538,28.085,30.97376200,32.07,35.45,39.9,39.0983,40.08,44.95591,47.867,50.9415,51.996,54.93804,55.84,58.93319,58.693,63.55,65.4])
        elements = np.array(['hydrogen', 'helium','lithium','beryllium','boron','carbon','nitrogen','oxygen','fluorine','neon','sodium','magnesium','aluminium','silicon','phosphorus','sulphur','chlorine','argon','potassium','calcium','scandium','titanium','vanadium','chromium','manganese','iron','cobalt','nickel','copper','zinc'])
        abundances = np.array([0, -1.01, -10.99, -10.63, -9.47, -3.53, -4.32, -3.17, -7.47, -4.01, -5.70, -4.45, -5.56, -4.48, -6.57, -4.87, -6.53, -5.63, -6.92, -5.67, -8.86, -7.01, -8.03, -6.36, -6.64, -4.51, -7.11, -5.78, -7.82, -7.43])
        fdpl = np.array([0, 0, 0.84, 0.4, 0.87, 0.5, 0, 0.3, 0.7, 0, 0.75, 0.8, 0.98, 0.9, 0.75, 0, 0.5, 0, 0.7, 0.997, 0.995, 0.992, 0.994, 0.994, 0.95, 0.99, 0.99, 0.96, 0.9, 0.75])
        abundances[numbers>=3] += np.log10(zmet)
        # z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        
        # primary+secondary nitrogren abundance prescription from Gutkin+16
        i_N = np.where(elements == 'nitrogen')[0][0]
        i_O = np.where(elements == 'oxygen')[0][0]
        abundances[i_N] = np.log10(0.41 * np.power(10., abundances[i_O]) * (np.power(10., -1.6) + np.power(10., 2.33+abundances[i_O])))
        z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        abundances[numbers>=3] += np.log10(zmet/z_ism)

        # variable C/O prescription from Gutkin+16
        i_C = np.where(elements == 'carbon')[0][0]
        # CO_sol = np.power(10, abundances[i_C])/np.power(10., abundances[i_O])
        abundances[i_C] += np.log10(CO)
        z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        abundances[numbers>=3] += np.log10(zmet/z_ism)

        # He abundance scaling w/ metallicity from Gutkin+16, following Bressan+12
        i_He = np.where(elements == 'helium')[0][0]
        abundances[i_He] = np.log10(np.power(10., abundances[i_He]) + 1.7756*zmet*zsol) 

        # adjust the depletion factors based on the dust-to-metals mass ratio
        xid0 = 0.36
        for i in range(len(fdpl)):
            if not fdpl[i] == 0:
                fdpl[i] = np.interp(xid, [0, xid0, 1], [0, fdpl[i], 1])

        abundances_depleted = np.log10(np.power(10., abundances * 1-fdpl))

        for i in range(len(elements)):
            # print(f'element abundance {elements[i]} {abundances_depleted[i]:.2f} no grains')
            f.write(f'element abundance {elements[i]} {abundances_depleted[i]:.2f} no grains\n')
    else:
        raise ValueError('Unknown abundance model, currently only Gutkin16 is supported')
    

    #######################################################################################################################################
    # Processing commands #################################################################################################################
    #######################################################################################################################################
    if params["iterate_to_convergence"] is not None:
        f.write("iterate to convergence\n")

    # if params["T_floor"] is not None:
    #     f.write(f'set temperature floor {params["T_floor"]} linear\n')

    # if params["stop_T"] is not None:
    #     f.write(f'stop temperature {params["stop_T"]}K\n')

    # if params["stop_efrac"] is not None:
    #     f.write(f'stop efrac {params["stop_efrac"]}\n')

    # if params["stop_column_density"] is not None:
    #     f.write(f'stop column density {params["stop_column_density"]}\n')
    #     # For some horrible reason the above is ignored in favour of a
    #     # built in temperature stop (4000K) unless that is turned off.
    #     f.write("stop temperature off\n")



    # # --- output commands
    # # cinput.append(f'print line vacuum\n')  # output vacuum wavelengths
    # cinput.append(
    #     f'set continuum resolution {params["resolution"]}\n'
    # )  # set the continuum resolution
    # cinput.append(f'save overview  "{model_name}.ovr" last\n')


    #######################################################################################################################################
    # Output commands #####################################################################################################################
    #######################################################################################################################################
    f.write(f'save last outward continuum "{filename}.cont" units Angstroms\n')
    f.write(f'save last line list intrinsic absolute column "{filename}.lines" "brisket_cloudy_lines.txt" \n')

    # f.write(f'save line list column absolute last units angstroms "{filename}.intrinsic_elin" "linelist.dat"\n')
    # f.write(f'save line list emergent column absolute last units angstroms "{filename}.emergent_elin" "linelist.dat"\n')
    
    
    # # save input file
    # if output_dir is not None:
    #     print(f"created input file: {output_dir}/{model_name}.in")
    #     open(f"{output_dir}/{model_name}.in", "w").writelines(cinput)

    # f.write("##### Output continuum and lines #####\n")
    # f.write("set save prefix \"" + "%.5f" % age + "\"\n")
    # f.write("save last outward continuum \".econ\" units microns\n")
    # f.write("save last line list intrinsic absolute column"
    #         + " \".lines\" \"pipes_cloudy_lines.txt\"\n")

    # f.write("########################################")

    f.close()


# def run_cloudy_model(age, zmet, logU, path):
#     """ Run an individual cloudy model. """

#     make_cloudy_sed_file(age, zmet)
#     make_cloudy_input_file(age, zmet, logU, path)
#     os.chdir(path + "/cloudy_temp_files/"
#              + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet)

#     os.system(os.environ["CLOUDY_EXE"] + " -r " + "%.5f" % age)
#     os.chdir("../../..")


# def extract_cloudy_results(age, zmet, logU, path):
#     """ Loads individual cloudy results from the output files and converts the
#     units to L_sol/A for continuum, L_sol for lines. """

#     cloudy_lines = np.loadtxt(path + "/cloudy_temp_files/"
#                               + "logU_" + "%.1f" % logU
#                               + "_zmet_" + "%.3f" % zmet + "/" + "%.5f" % age
#                               + ".lines", usecols=(1),
#                               delimiter="\t", skiprows=2)

#     cloudy_cont = np.loadtxt(path + "/cloudy_temp_files/"
#                              + "logU_" + "%.1f" % logU + "_zmet_"
#                              + "%.3f" % zmet + "/" + "%.5f" % age + ".econ",
#                              usecols=(0, 3, 8))[::-1, :]

#     # wavelengths from microns to angstroms
#     cloudy_cont[:, 0] *= 10**4

#     # subtract lines from nebular continuum model
#     cloudy_cont[:, 1] -= cloudy_cont[:, 2]

#     # continuum from erg/s to erg/s/A.
#     cloudy_cont[:, 1] /= cloudy_cont[:, 0]

#     # Get bagpipes input spectrum: angstroms, erg/s/A
#     input_spectrum = get_bagpipes_spectrum(age, zmet)

#     # Total ionizing flux in the bagpipes model in erg/s
#     ionizing_spec = input_spectrum[(input_spectrum[:, 0] <= 911.8), 1]
#     ionizing_wavs = input_spectrum[(input_spectrum[:, 0] <= 911.8), 0]
#     pipes_ionizing_flux = np.trapz(ionizing_spec, x=ionizing_wavs)

#     # Total ionizing flux in the cloudy outputs in erg/s
#     cloudy_ionizing_flux = np.sum(cloudy_lines) + np.trapz(cloudy_cont[:, 1],
#                                                            x=cloudy_cont[:, 0])

#     # Normalise cloudy fluxes to the level of the input bagpipes model
#     cloudy_lines *= pipes_ionizing_flux/cloudy_ionizing_flux
#     cloudy_cont[:, 1] *= pipes_ionizing_flux/cloudy_ionizing_flux

#     # Convert cloudy fluxes from erg/s/A to L_sol/A
#     cloudy_lines /= 3.826*10**33
#     cloudy_cont[:, 1] /= 3.826*10**33

#     nlines = config.wavelengths.shape[0]
#     cloudy_cont_resampled = np.zeros((nlines, 2))

#     # Resample the nebular continuum onto wavelengths of stellar models
#     cloudy_cont_resampled[:, 0] = config.wavelengths
#     cloudy_cont_resampled[:, 1] = np.interp(cloudy_cont_resampled[:, 0],
#                                             cloudy_cont[:, 0],
#                                             cloudy_cont[:, 1])

#     return cloudy_cont_resampled[:, 1], cloudy_lines


# def compile_cloudy_grid(path):

#     line_wavs = np.loadtxt(utils.install_dir
#                            + "/models/grids/cloudy_linewavs.txt")

#     for logU in config.logU:
#         for zmet in config.metallicities:

#             print("logU: " + str(np.round(logU, 1))
#                   + ", zmet: " + str(np.round(zmet, 4)))

#             mask = (config.age_sampling < age_lim)
#             contgrid = np.zeros((config.age_sampling[mask].shape[0]+1,
#                                  config.wavelengths.shape[0]+1))

#             contgrid[0, 1:] = config.wavelengths
#             contgrid[1:, 0] = config.age_sampling[config.age_sampling < age_lim]

#             linegrid = np.zeros((config.age_sampling[mask].shape[0]+1,
#                                 line_wavs.shape[0]+1))

#             linegrid[0, 1:] = line_wavs
#             linegrid[1:, 0] = config.age_sampling[mask]

#             for i in range(config.age_sampling[mask].shape[0]):
#                 age = config.age_sampling[mask][i]
#                 cont_fluxes, line_fluxes = extract_cloudy_results(age*10**-9,
#                                                                   zmet, logU,
#                                                                   path)

#                 contgrid[i+1, 1:] = cont_fluxes
#                 linegrid[i+1, 1:] = line_fluxes

#             if not os.path.exists(path + "/cloudy_temp_files/grids"):
#                 os.mkdir(path + "/cloudy_temp_files/grids")

#             np.savetxt(path + "/cloudy_temp_files/grids/"
#                        + "zmet_" + str(zmet) + "_logU_" + str(logU)
#                        + ".neb_lines", linegrid)

#             np.savetxt(path + "/cloudy_temp_files/grids/"
#                        + "zmet_" + str(zmet) + "_logU_" + str(logU)
#                        + ".neb_cont", contgrid)

#     # Nebular grids
#     list_of_hdus_lines = [fits.PrimaryHDU()]
#     list_of_hdus_cont = [fits.PrimaryHDU()]

#     for logU in config.logU:
#         for zmet in config.metallicities:

#             line_data = np.loadtxt(path + "/cloudy_temp_files/"
#                                    + "grids/zmet_" + str(zmet)
#                                    + "_logU_" + str(logU) + ".neb_lines")

#             hdu_line = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
#                                      + "%.1f" % logU, data=line_data)

#             cont_data = np.loadtxt(path + "/cloudy_temp_files/"
#                                    + "grids/zmet_" + str(zmet)
#                                    + "_logU_" + str(logU) + ".neb_cont")

#             hdu_cont = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
#                                      + "%.1f" % logU, data=cont_data)

#             list_of_hdus_lines.append(hdu_line)
#             list_of_hdus_cont.append(hdu_cont)

#     hdulist_lines = fits.HDUList(hdus=list_of_hdus_lines)
#     hdulist_cont = fits.HDUList(hdus=list_of_hdus_cont)

#     hdulist_lines.writeto(path + "/cloudy_temp_files"
#                           + "/grids/bagpipes_nebular_line_grids.fits",
#                           overwrite=True)

#     hdulist_cont.writeto(path + "/cloudy_temp_files"
#                          + "/grids/bagpipes_nebular_cont_grids.fits",
#                          overwrite=True)


# def run_cloudy_grid(path=None):
#     """ Generate the whole grid of cloudy models and save to file. """

#     if path is None:
#         path = utils.working_dir

#     if rank == 0 and not os.path.exists(path + "/cloudy_temp_files"):
#         os.mkdir(path + "/cloudy_temp_files")

#     ages = config.age_sampling[config.age_sampling < age_lim]

#     n_models = config.logU.shape[0]*ages.shape[0]*config.metallicities.shape[0]

#     params = np.zeros((n_models, 3))

#     n = 0
#     for i in range(config.logU.shape[0]):
#         for j in range(config.metallicities.shape[0]):

#             # Make directory to store cloudy inputs/outputs
#             if rank == 0:
#                 if not os.path.exists(path + "/cloudy_temp_files/"
#                                       + "logU_" + "%.1f" % config.logU[i]
#                                       + "_zmet_" + "%.3f" % config.metallicities[j]):

#                     os.mkdir(path + "/cloudy_temp_files/"
#                              + "logU_" + "%.1f" % config.logU[i]
#                              + "_zmet_" + "%.3f" % config.metallicities[j])

#             # Populate array of parameter values
#             for k in range(ages.shape[0]):

#                 params[n, 0] = ages[k]
#                 params[n, 1] = config.metallicities[j]
#                 params[n, 2] = config.logU[i]
#                 n += 1

#     # Assign models to cores
#     thread_nos = mpi_split_array(np.arange(n_models))

#     # Run models assigned to this core
#     for n in thread_nos:
#         age = params[n, 0]
#         zmet = params[n, 1]
#         logU = params[n, 2]

#         print("logU: " + str(np.round(logU, 1)) + ", zmet: "
#               + str(np.round(zmet, 4)) + ", age: "
#               + str(np.round(age*10**-9, 5)))

#         run_cloudy_model(age*10**-9, zmet, logU, path)

#     # Combine arrays of models assigned to cores, checks all is finished
#     mpi_combine_array(thread_nos, n_models)

#     # Put the final grid fits files together
#     if rank == 0:
#         compile_cloudy_grid(path)

def make_cloudy_sed_file(output_dir, filename, wavs, fluxes):
    energy = 911.8/wavs
    nu = 2.998e8/(wavs*1e-10) # in Hz
    fluxes = fluxes * 3.826e33 / nu # in erg/s/Hz
    fluxes[fluxes <= 0] = np.power(10.,-99)
    energy = np.flip(energy)
    fluxes = np.flip(fluxes)
    np.savetxt(f"{output_dir}/{filename}.sed", 
               np.array([energy, fluxes]).T, 
               header="Energy units: Rydbergs, Flux units: erg/s/Hz",)

def extract_cloudy_results(dir, filename, input_wav, input_flux):
    """ Loads individual cloudy results from the output files and converts the
    units to L_sol/A for continuum, L_sol for lines. """

    lines = np.loadtxt(f"{dir}/{filename}.lines", usecols=(1), delimiter="\t", skiprows=2)
    cont_wav, cont_incident, cont_flux, cont_lineflux = np.loadtxt(f"{dir}/{filename}.cont", usecols=(0, 1, 3, 8)).T
    cont_wav = np.flip(cont_wav)
    cont_flux = np.flip(cont_flux)
    cont_lineflux = np.flip(cont_lineflux)

    # Convert cloudy fluxes from erg/s to Lsun
    lines /= 3.826e33
    cont_flux /= 3.826e33
    cont_lineflux /= 3.826e33

    # subtract lines from nebular continuum model
    cont_flux -= cont_lineflux

    # continuum from Lsun to Lsun/A.
    cont_flux /= cont_wav

    cont_flux[cont_wav < 911.8] = 0

    # # Total ionizing flux in the input model
    ionizing_spec = input_flux[input_wav <= 911.8]
    ionizing_wavs = input_flux[input_wav <= 911.8]
    input_ionizing_flux = np.trapezoid(ionizing_spec, x=ionizing_wavs)

    # # Total ionizing flux in the cloudy outputs
    cloudy_ionizing_flux = np.sum(lines) + np.trapezoid(cont_flux, x=cont_wav)

    # # Normalise cloudy fluxes to the level of the input model
    lines *= input_ionizing_flux/cloudy_ionizing_flux
    cont_flux *= input_ionizing_flux/cloudy_ionizing_flux

    # Resample the nebular continuum onto wavelengths of stellar models
    cont_flux = np.interp(input_wav, cont_wav, cont_flux)
    return cont_flux, lines




default_cloudy_params = {
        "no_grain_scaling": False,
        "ionisation_parameter": None, # ionisation parameter
        "radius": None, # radius in log10 parsecs, only important for spherical geometry
        "covering_factor": None, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
        "stop_T": None, # K, if not provided the command is not used
        "stop_efrac": None, # if not provided the command is not used
        "stop_column_density": None, # log10(N_H/cm^2), if not provided the command is not used
        "T_floor": None, # K, if not provided the command is not used
        "hydrogen_density": None, # Hydrogen density
        "z": 0.0, # redshift, only necessary if CMB heating included
        "CMB": None, # include CMB heating
        "cosmic_rays": None, # include cosmic rays
        "metals": True, # include metals
        "grains": None, # include dust grains
        "geometry": None, # the geometry

        "constant_density": None, # constant density flag
        "constant_pressure": None, # constant pressure flag # need one of these two

        "resolution": 1.0, # relative resolution the saved continuum spectra
        "output_abundances": None, # output abundances
        "output_cont": None, # output continuum
        "output_lines": None, # output full list of all available lines
        "output_linelist": None, # output linelist
    }

# to run: python -m brisket.grids.cloudy <params>.toml
import argparse, h5py, tqdm, toml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Run cloudy on a given input grid.')
    parser.add_argument('param', help='Path to TOML parameter file', type=str)
    args = parser.parse_args()
    params = toml.load(args.param)
    params = default_cloudy_params | params

    logger = setup_logger(__name__, level='DEBUG')
    if rank==0: logger.info(f"Running cloudy with parameters from: [bold]{args.param}[/bold]")

    if rank==0: logger.info(f'Running on input grid: [bold]{params["grid"]}[/bold]')
    grid = Grid(params['grid'])
    wavs = grid.wavelengths
    if rank==0: logger.info(f'Detected input grid axes: {list(grid.axes)}')


    if 'age' in grid.axes: # TODO FIX NAMING OF AGES IN GRID FILES
        if rank==0: logger.info(f'Shrinking grid to only include ages less than age_lim = {params["age_lim"]} Myr')
        age_axis = grid.array_axes.index('age')
        indices = np.where(grid.age <= params['age_lim']*1e6)[0]
        grid.data = np.take(grid.data, indices, axis=age_axis)
        grid.age = grid.age[grid.age <= params['age_lim']*1e6]
        if rank==0: logger.info(f'Reduced grid shape: {grid.shape}')
    else:
        if rank==0: logger.info(f'Detected input grid shape: {grid.shape}')

    # copy over the line list into a file cloudy can read
    # these are the lines we want to track in the cloudy models
    if rank==0: logger.info(f"Exporting cloudy line list to {cloudy_data_path}/brisket_cloudy_lines.txt")
    with open(f"{cloudy_data_path}/brisket_cloudy_lines.txt", 'w') as f:
        f.writelines([l + '\n' for l in linelist.cloudy_labels])
    
    # create a temporary directory to store the CLOUDY input/output files
    if rank==0: logger.info(f"Creating cloudy_temp/brisket_{grid.name}/ directory")
    os.chdir(config.grid_dir)
    base_dir = f'./cloudy_temp/brisket_{grid.name}/'
    if not os.path.exists(base_dir): 
        os.makedirs(base_dir)

    if rank==0: logger.info("[bold]Detecting CLOUDY parameter input axes")
    # handle the input parameters
    # things that can be free: logU, lognH, xid, CO
    default_vals = {'logU':-2, 'lognH':2, 'xid':0.36, 'CO':1.0}
    cloudy_axes = []
    cloudy_axes_vals = []
    for k,v in default_vals.items():
        if k in params:
            # then we have specified this parameter, whether free or fixed
            fixed = False
            if 'fixed' in params[k]:
                fixed = params[k]['fixed']
            elif 'isfixed' in params[k]:
                fixed = params[k]['isfixed']
            elif 'free' in params[k]:
                fixed = not params[k]['free']
            elif 'isfree' in params[k]:
                fixed = not params[k]['isfree']
            else:
                raise ValueError(f"Parameter {k} must have a key specifying whether it is free or fixed: 'fixed', 'free', 'isfree', isfixed'")

            if fixed:
                if rank==0: logger.info(f"Parameter {k} fixed at {params[k]['value']}")
                params[k] = params[k]['value']
            else:
                if rank==0: logger.info(f"Parameter {k} free from {params[k]['low']} to {params[k]['high']} in steps of {params[k]['step']}")
                cloudy_axes.append(k)
                cloudy_axes_vals.append(np.arange(params[k]['low'], params[k]['high'], params[k]['step']))
        else:
            logger.warning(f'Parameter {k} not specified, fixing at default value {v}')
            params[k] = v

    cloudy_axes_shape = [len(vals) for vals in cloudy_axes_vals]

    final_grid_axes = list([str(s) for s in grid.axes]) + cloudy_axes
    final_grid_shape = grid.shape + tuple(cloudy_axes_shape)
    n_runs = np.prod(final_grid_shape)

    if rank==0: logger.info(f"Final grid will have axes: {final_grid_axes}")
    if rank==0: logger.info(f"Final grid will have shape: {final_grid_shape}")
    if rank==0: logger.info(f"Preparing to run cloudy {n_runs} times!")

    if size > 1 and not params['MPI']:
        if rank == 0: logger.error("MPI is not enabled, but more than one core is available. Please enable MPI in the parameter file.")
        quit()
    elif size > 1:
        # Assign models to cores
        if rank==0: logger.info(f"Splitting up into {size} MPI threads")
        threads = mpi_split_array(np.arange(n_runs))
        indices = np.array(list(np.ndindex(final_grid_shape)))[threads]
        indices = [tuple(i) for i in indices]
    else:
        threads = np.arange(n_runs)
        indices = list(np.ndindex(final_grid_shape))
    n_threads = len(threads)

    n_current = 1
    for index in indices:
        filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
        logger.info(f"[bold green]Thread {rank}[/bold green]: running model {n_current}/{n_threads} ({filename})")

        if len(cloudy_axes) > 0:
            input_grid_index = index[:-len(cloudy_axes)]
            cloudy_grid_index = index[-len(cloudy_axes):] 
            cloudy_params = {cloudy_axes[i] : cloudy_axes_vals[i][cloudy_grid_index[i]] for i in range(len(cloudy_axes))}
            params = params | cloudy_params
        else:
            input_grid_index = index



        if 'zmet' in grid.axes: 
            zmet_index = grid.axes.index('zmet')
            zmet = grid.zmet[input_grid_index[zmet_index]]
            params['zmet'] = zmet
        else:
            raise Exception('grid needs to have metallicity (for now)')

        # export the SED to a file that cloudy can read
        make_cloudy_sed_file(cloudy_sed_dir, filename, wavs, grid.data[input_grid_index])

        # create the cloudy input file, using the params
        make_cloudy_input_file(base_dir, filename=filename, params=params)

        os.chdir(base_dir)

        # remove any existing cloudy output files
        if os.path.exists(f"{filename}.cont"):
            os.remove(f"{filename}.cont")
        if os.path.exists(f"{filename}.lines"):
            os.remove(f"{filename}.lines")
        if os.path.exists(f"{filename}.out"):
            os.remove(f"{filename}.out")

        # run cloudy!
        cloudy_exe = os.environ["CLOUDY_EXE"]
        os.system(f'{cloudy_exe} -r {filename}')
        
        # remove the SED file, since we don't need it anymore (and we don't want to clog the filesystem)
        os.remove(f"{filename}.in")
        os.remove(f"{filename}.out")
        os.remove(f"{cloudy_sed_dir}/{filename}.sed")
        os.chdir(config.grid_dir)
        n_current += 1

    
    if params['MPI']:
        # Combine arrays of models assigned to cores, checks all is finished
        # if rank==0: logger.info(f"Combining cloudy runs from {size} MPI threads
        threads = mpi_combine_array(threads, n_runs)

    if rank == 0:
        final_grid_cont_data = np.zeros(final_grid_shape + (len(wavs),))
        final_grid_line_data = np.zeros(final_grid_shape + (len(linelist),))

        for index in (pbar := tqdm.tqdm(np.ndindex(final_grid_shape), total=n_runs)):
            filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
            pbar.set_description(filename.removeprefix('brisket_'))

            if len(cloudy_axes) > 0:
                input_grid_index = index[:-len(cloudy_axes)]
            else:
                input_grid_index = index

            cont, lines = extract_cloudy_results(base_dir, filename, wavs, grid[input_grid_index])

            final_grid_cont_data[index] = cont
            final_grid_line_data[index] = lines


        outfilepath = os.path.join(config.grid_dir, params['grid']+'_cloudy_lines.hdf5')
        with h5py.File(outfilepath, 'w') as hf:
            hf.create_dataset('wavs', data=linelist.wavs)
            hf.create_dataset('names', data=list(linelist.names), dtype=h5py.string_dtype())
            hf.create_dataset('labels', data=list(linelist.labels),  dtype=h5py.string_dtype())
            hf.create_dataset('cloudy_labels', data=list(linelist.cloudy_labels),  dtype=h5py.string_dtype())
            
            hf.create_dataset('axes', data=final_grid_axes + ['wavs'])
            input_params = {axis : getattr(grid, axis) for axis in grid.axes}
            for axis in final_grid_axes:
                if axis in cloudy_axes:
                    hf.create_dataset(axis, data=cloudy_axes_vals[cloudy_axes.index(axis)])
                else:
                    hf.create_dataset(axis, data=getattr(grid, axis))

            hf.create_dataset('grid', data=final_grid_line_data)

        outfilepath = os.path.join(config.grid_dir, params['grid']+'_cloudy_cont.hdf5')
        with h5py.File(outfilepath, 'w') as hf:
            hf.create_dataset('wavs', data=wavs)
            
            hf.create_dataset('axes', data=final_grid_axes + ['wavs'])
            input_params = {axis : getattr(grid, axis) for axis in grid.axes}
            for axis in final_grid_axes:
                if axis in cloudy_axes:
                    hf.create_dataset(axis, data=cloudy_axes_vals[cloudy_axes.index(axis)])
                else:
                    hf.create_dataset(axis, data=getattr(grid, axis))

            hf.create_dataset('grid', data=final_grid_cont_data)

        for index in (pbar := tqdm.tqdm(np.ndindex(final_grid_shape), total=n_runs)):
            filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
            os.remove(f"{base_dir}/{filename}.lines")
            os.remove(f"{base_dir}/{filename}.cont")

