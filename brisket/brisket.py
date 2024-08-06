#####################################################################################
############################ main command line interface ############################
#####################################################################################
from copy import copy
import astropy.units as u
import argparse, logging, toml
import sys, os, shutil
from . import config
from . import utils
from .filters import filter_set
import numpy as np

def parse_toml_paramfile(toml_file, logger=utils.NullLogger):
    param = toml.load(toml_file)
    kwargs = {}
    if 'mod' in param:
        # if 'spec_wavs' in param['mod']:
        #     kwargs['spec_wavs'] = spec_wavs
        #     logger.debug('Using spectral wavelengths from param file:', kwargs['spec_wavs'])
        # else:
        #     logger.debug('No spectroscopic model output requested')
        
        if 'filt_list' in param['mod']:
            logger.debug(f'Parsing filt_list from param file')
            fs = filter_set(param['mod']['filt_list'], logger=logger)
            kwargs['filt_list'] = fs
        else:
            logger.debug('No photometric model output requested')

        del param['mod']

    if 'fit' in param:
        from astropy.io import fits
        kwargs = {}
        load_phot = None
        if 'photometry' in param['fit']:
            ID = param['fit']['ID']
            catalog = fits.getdata(param['fit']['photometry']['file'])
            catalog = catalog[catalog['ID']==ID]
            phot = np.array([catalog[col][0] for col in param['fit']['photometry']['phot_columns']])
            phot_err = np.array([catalog[col][0] for col in param['fit']['photometry']['phot_err_columns']])
            load_phot = lambda ID: np.array([phot, phot_err]).T 
            kwargs['filt_list'] = param['fit']['photometry']['filt_list']
            kwargs['phot_units'] = utils.unit_parser(param['fit']['photometry']['phot_units'])

        load_spec = None
        if 'spectroscopy' in param['fit']:
            spectrum = fits.getdata(param['fit']['spectroscopy']['file'])
            spec_wav = spectrum[wav_column]
            spec = spectrum[spec_column]
            spec_err = spectrum[spec_err_column]
            load_spec = lambda ID: np.array([spec_wav, spec, spec_err]).T
            kwargs['wav_units'] = utils.unit_parser(param['fit']['spectroscopy']['wav_units'])
            kwargs['spec_units'] = utils.unit_parser(param['fit']['spectroscopy']['spec_units'])

        ### parse spec_wavs and filt_list out of param file
        ### parse units out of param file
        del param['fit']

        def reword_priors(name, data):
            out = {}
            out[name] = (data['low'], data['high'])
            if 'prior' in data:
                if data['prior'] != 'Uniform':
                    out[name + '_prior'] = data['prior']
                    for p in data:
                        if not p in ['prior','low','high']:
                            out[name + f'_prior_{p}'] = data[p]
            return out

            
        print(param)

        fit_instructions = {}
        for group in param:
            fit_instructions[group] = {}
            for key in param[group]:
                # If the value is a number or string, just adopt the fixed value directly 
                if type(param[group][key]) in [str,int,float,list]: 
                    fit_instructions[group][key] = param[group][key]
                # If the value is a dict, that means we have a sub-group, e.g. nebular model within galaxy model
                elif type(param[group][key]) == dict: 
                    subgroup = key
                    fit_instructions[group][subgroup] = {}
                    for key in param[group][subgroup]:
                        if type(param[group][subgroup][key]) in [str,int,float]:
                            fit_instructions[group][subgroup][key] = param[group][subgroup][key]
                        else:
                            fit_instructions[group][subgroup].update(reword_priors(key, param[group][subgroup][key]))
                # The only other option is the TOML inline table data type, which is similar to a dict but not exactly a dict
                # This we use for the specification of priors for free parameters 
                else: # TOML inline table, specification of free parameter
                    fit_instructions[group].update(reword_priors(key, param[group][key]))

        fit_instructions.update(fit_instructions['base'])
        del fit_instructions['base']
        param = fit_instructions

    kwargs['logger'] = logger
    return param, kwargs






def setup_logger(outdir, run, verbose):
    logger = logging.getLogger(run)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(outdir, f'{run}.log'), mode='w') # file handler
    ch = logging.StreamHandler(sys.stdout) # console handler
    if verbose: 
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', "%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def mod():
    if '--gen-param' in sys.argv:
        print("Saving default parameter file to 'param_mod_default.toml'")
        defaults_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'defaults')
        shutil.copy2(os.path.join(defaults_dir,'param_mod_default.toml'), '.')
        print('Exiting...')
        sys.exit(0)

    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
    parser = argparse.ArgumentParser(
                        prog='brisket-mod',
                        description='brisket-mod: generating galaxy/AGN SED models from the command line.',
                        formatter_class=formatter)
                        # epilog='More help text')
    parser.add_argument('-p', '--param', help='Path to TOML parameter file', metavar='{param}.toml', type=str)
    parser.add_argument('-r', '--run', help='Run name', metavar='{run}', type=str)
    parser.add_argument('-o', '--outdir', metavar='{dir}/', help='Output directory (defaults to brisket/)', type=str) 
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (y/n)')  # on/off flag
    args = parser.parse_args()
    
    param, run = args.param, args.run
    if args.outdir is None:
        outdir = f'brisket/{run}/'
    else:
        assert os.path.exists(args.outdir)
        outdir = os.path.join(args.outdir, f'{run}')
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(param, outdir)

    logger = setup_logger(outdir, run, args.verbose)
    logger.debug(f'args {args}')
    logger.info(f'Beginning model generation for run {run} from param file {param}')

    logger.debug(f'Importing BRISKET')
    import brisket 

    param, kwargs = parse_toml_paramfile(param, logger=logger)

    gal = brisket.model_galaxy(param, **kwargs)
    gal.save_output(os.path.join(outdir, f'{run}.fits'), overwrite=True)



def fit():

    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
    parser = argparse.ArgumentParser(
                        prog='brisket-fit',
                        description='brisket-fit: command line tool for fitting galaxy/AGN SED models to photometric/spectroscopic data',
                        formatter_class=formatter)
    
    parser.add_argument('-p', '--param', help='Path to TOML parameter file', metavar='{param}.toml', type=str)
    parser.add_argument('-r', '--run', help='Run name', metavar='{run}', type=str)
    # parser.add_argument('-m', '--mpi', help='MPI?', metavar='{run}', type=str)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (y/n)')  # on/off flag
    args = parser.parse_args()
    
    param, run = args.param, args.run
    outdir = f'brisket/{run}/'
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(param, outdir)
    logger = setup_logger(outdir, run, args.verbose)

    ID = param['fit']['ID']

    import brisket
    gal = brisket.galaxy(ID, 
                 load_phot=load_phot, # instead of a catalog file
                 load_spec=load_spec,
                 **kwargs)
    print(gal.photometry)   
    
    
    fit_params = copy(param['fit'])
    del param['fit']
    fit = brisket.fit(gal, param, run=run, n_posterior=fit_params['n_posterior'])

    print(fit.fit_instructions)
    # fit.fit(verbose=args.verbose, n_live=fit_params['n_live'])



def plot():
    pass



def filters():
    import h5py

    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
    parser = argparse.ArgumentParser(
                        prog='brisket-filters',
                        description='brisket-filters: command-line tool for adding/editing the database of filter curves',
                        formatter_class=formatter)
                        # epilog='More help text')
    
    parser.add_argument('action', action='store', choices=['add','edit','remove'], help='Action to perform: add, edit, or remove a filter curve from the database')
    parser.add_argument('filter', action='store', help='Filter file to add or filter name to edit/remove')

    ## arguments used if action==add
    if 'add' in sys.argv:
        parser.add_argument('-w', '--wav-units', action='store', choices=['angstrom', 'nm', 'um', 'mm', 'm'], help="Wavelength units in the filter file (default='angstrom')", default='angstrom', type=str)
        parser.add_argument('-d', '--description', action='store', help="Short description (e.g., relevant telescope, data source)", default='', type=str)
        parser.add_argument('-n', '--nickname', action='extend', nargs='+', help="Nickname that you can reference the filter as (e.g. f277w -> jwst_nircam_f277w). Can specify multiple ", type=str, default=[])
        parser.add_argument('-o', '--overwrite', action='store_true', help='Whether to overwrite existing filter curve of the same name')
    ### ex brisket-filters add jwst_nircam_f277w.dat -d "JWST NIRCam F277W from SVO filter profile service" -n f277w F277W

    args = parser.parse_args()
            

    if args.action=='add':
        filter_name = os.path.splitext(os.path.basename(args.filter))[0]

        # load filter file 
        filt = np.genfromtxt(args.filter, usecols=(0,1))

        # if wav-units is not angstroms, convert to angstroms

        # truncate any zeros from the file
        while filt[0, 1] == 0: filt = filt[1:, :]
        while filt[-1, 1] == 0: filt = filt[:-1, :]

        with h5py.File(config.filter_db, mode='a') as db:
            if filter_name in db:
                if args.overwrite:
                    del db[filter_name]
                else:
                    print(f'Filter {filter_name} already exists in database, either remove it or toggle overwriting (-o --overwrite)')
                    sys.exit(0)

            ds = db.create_dataset(filter_name, data=filt)
            ds.attrs['description'] = args.description
            ds.attrs['nicknames'] = args.nickname
            



