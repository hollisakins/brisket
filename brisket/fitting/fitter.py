from __future__ import print_function, division, absolute_import
from astropy.io import fits
from contextlib import redirect_stdout
import numpy as np
import os
import time
import warnings
import sys


from copy import deepcopy

try:
    with open(os.devnull, "w") as f, redirect_stdout(f):
        import pymultinest as pmn
    multinest_available = True
except (ImportError, RuntimeError, SystemExit):
    print('BRISKET: PyMultiNest import failed, fitting will use the Ultranest sampler instead.')
    multinest_available = False

try:
    from nautilus import Sampler
    nautilus_available = True
except (ImportError, RuntimeError, SystemExit):
    print('BRISKET: Nautilus import failed, fitting will use the Ultranest sampler instead.')
    nautilus_available = False

try:
    from ultranest import ReactiveNestedSampler
    from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
    ultranest_available = True
except (ImportError, RuntimeError, SystemExit):
    print("BRISKET: Ultranest import failed, fitting will use the Nautilus sampler instead.")
    ultranest_available = False


# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0

from brisket import utils
# from .. import plotting

from brisket.fitting.fitted_model import FittedModel
from brisket.fitting.posterior import Posterior
from brisket.parameters import Params


class Fitter(object):
    """ Top-level class for fitting models to observational data.
    Interfaces with MultiNest to sample from the posterior distribution
    of a fitted_model object. Performs loading and saving of results.

    Parameters
    ----------

    galaxy : bagpipes.Galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    parameters : brisket.parameters.Params
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int - optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    """

    def __init__(self, galaxy, parameters, run=".", time_calls=False,
                 n_posterior=500, logger=utils.NullLogger):

        self.run = run
        self.galaxy = galaxy
        self.n_posterior = n_posterior
        self.logger = logger

        # Handle the input parameters, whether provided in dictionary form or in Params object.
        if isinstance(parameters, Params):
            self.logger.debug(f'Parameters loaded')            
            self.parameters = parameters
        elif isinstance(parameters, dict):
            self.logger.info(f'Loading parameter dictionary')           
            self.parameters = Params(parameters)
        else:
            msg = "Input `parameters` must be either a brisket.parameters.Params object or python dictionary"
            self.logger.error(msg)
            raise TypeError(msg)

        self.parameters = deepcopy(self.parameters)

        # Set up the directory structure for saving outputs.
        if rank == 0:
            utils.make_dirs(run=run)

        # The base name for output files.
        self.fname = f'brisket/posterior/{run}/{self.galaxy.ID}_'

        # A dictionary containing properties of the model to be saved.
        self.results = {}

        # If a posterior file already exists load it.
        if os.path.exists(f'{self.fname}brisket_results.fits'):

            file = fits.open(f'{self.fname}brisket_results.fits')['RESULTS']
            self.parameters.data = utils.str_to_dict(file.header['PARAMS'])
            self.results['samples2d'] = file.data['samples2d']
            self.results['lnlike'] = file.data['lnlike']
            self.results['lnz'] = file.header['LNZ']
            self.results['lnz_err'] = file.header['LNZ_ERR']
            self.results["median"] = np.median(self.results['samples2d'], axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                     (16, 84), axis=0)
            
            self.posterior = Posterior(self.galaxy, run=run,
                                       n_samples=n_posterior)
            
            if rank == 0:
                self.logger.info(f'Loaded results from {self.fname}brisket_results.fits')

        # Set up the model which is to be fitted to the data.
        self.fitted_model = FittedModel(galaxy, self.parameters,
                                       time_calls=time_calls)

    def fit(self, verbose=False, n_live=400, use_MPI=True, 
            sampler="multinest", n_eff=0, discard_exploration=False,
            n_networks=4, pool=1, nsteps=4, overwrite=False):
        """ Fit the specified model to the input galaxy data.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.

        sampler : string - optional
            The sampler to use. Available options are "ultranest", 
            "multinest", and "nautilus".

        n_eff : float - optional
            Target minimum effective sample size. Only used by nautilus.

        discard_exploration : bool - optional
            Whether to discard the exploration phase to get more accurate
            results. Only used by nautilus.

        n_networks : int - optional
            Number of neural networks. Only used by nautilus.

        pool : int - optional
            Pool size used for parallelization. Only used by nautilus.
            MultiNest is parallelized with MPI.

        """
        if "lnz" in list(self.results):
            if rank == 0:
                self.logger.info(f'Fitting not performed as results have already been loaded from {self.fname[:-1]}.h5. To start over delete this file or change run.')
            self._print_results()
            return

        # Figure out which sampling algorithm to use
        sampler = sampler.lower()

        if (sampler == "multinest" and not multinest_available and
                nautilus_available):
            sampler = "nautilus"
            print("MultiNest not available. Switching to nautilus.")

        elif (sampler == "nautilus" and not nautilus_available and
                multinest_available):
            sampler = "multinest"
            print("Nautilus not available. Switching to MultiNest.")

        elif sampler not in ["multinest", "nautilus", "ultranest"]:
            raise ValueError("Sampler {} not supported.".format(sampler))

        elif not (multinest_available or nautilus_available or ultranest_available):
            raise RuntimeError("No sampling algorithm could be loaded.")

        if rank == 0 or not use_MPI:
            self.logger.info(f'Fitting object {self.galaxy.ID}')

            start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            os.environ['PYTHONWARNINGS'] = 'ignore'

            if sampler == 'multinest':
                pmn.run(self.fitted_model.lnlike,
                        self.fitted_model.prior.transform,
                        self.fitted_model.ndim, n_live_points=n_live,
                        importance_nested_sampling=False, verbose=verbose,
                        sampling_efficiency='model',
                        outputfiles_basename=self.fname, use_MPI=use_MPI)

            elif sampler == 'nautilus':
                n_sampler = Sampler(self.fitted_model.prior.transform,
                                    self.fitted_model.lnlike, n_live=n_live,
                                    n_networks=n_networks, pool=pool,
                                    n_dim=self.fitted_model.ndim,
                                    filepath=self.fname + '.h5')

                n_sampler.run(verbose=verbose, n_eff=n_eff,
                              discard_exploration=discard_exploration)

            elif sampler == 'ultranest':
                # os.environ['OMP_NUM_THREADS'] = '1'
                resume = 'resume'
                if overwrite:
                    resume = 'overwrite'

                u_sampler = ReactiveNestedSampler(self.fitted_model.params, 
                                                self.fitted_model.lnlike, 
                                                transform=self.fitted_model.prior.transform, 
                                                log_dir='/'.join(self.fname.split('/')[:-1]), 
                                                resume=resume, 
                                                run_num=None)
                u_sampler.stepsampler = SliceSampler(nsteps=nsteps,#len(self.fitted_model.params)*4,
                                                     generate_direction=generate_mixture_random_direction)
                u_sampler.run(
                    min_num_live_points=n_live,
                    dlogz=0.5, # desired accuracy on logz -- could allow to specify
                    min_ess=self.n_posterior, # number of effective samples
                    # update_interval_volume_fraction=0.4, # how often to update region
                    # max_num_improvement_loops=3, # how many times to go back and improve
                )

            os.environ["PYTHONWARNINGS"] = ""

            # if self.logger != utils.NullLogger:
            # print(self.logger.handlers)
            # print(self.logger.handlers[0].baseFilename)
            # with open(self.logger.handlers[0].baseFilename, "a") as f:
                # with redirect_stdout(f):
            # pmn.run(self.fitted_model.lnlike,
            #         self.fitted_model.prior.transform,
            #         self.fitted_model.ndim, n_live_points=n_live,
            #         importance_nested_sampling=False, verbose=verbose,
            #         sampling_efficiency="model",
            #         outputfiles_basename=self.fname, use_MPI=use_MPI)

        if rank == 0 or not use_MPI:
            runtime = time.time() - start_time
            if runtime > 60:
                runtime = f'{int(np.floor(runtime/60))}m{runtime-np.floor(runtime/60)*60:.1f}s'
            else: 
                runtime = f'{runtime:.1f} seconds'

            self.logger.info(f'Completed in {runtime}.')

            # Load sampler outputs 
            if sampler == "multinest":
                samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
                lnz_line = open(self.fname + "stats.dat").readline().split()
                self.results["samples2d"] = samples2d[:, :-1]
                self.results["lnlike"] = samples2d[:, -1]
                self.results["lnz"] = float(lnz_line[-3])
                self.results["lnz_err"] = float(lnz_line[-1])
                
                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == "nautilus":
                samples2d = np.zeros((0, self.fitted_model.ndim))
                log_l = np.zeros(0)
                while len(samples2d) < self.n_posterior:
                    result = n_sampler.posterior(equal_weight=True)
                    samples2d = np.vstack((samples2d, result[0]))
                    log_l = np.concatenate((log_l, result[2]))
                self.results["samples2d"] = samples2d
                self.results["lnlike"] = log_l
                self.results["lnz"] = n_sampler.log_z
                self.results["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)

                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == 'ultranest':
                self.results['samples2d'] = u_sampler.results['samples']
                self.results['lnlike'] = u_sampler.results['weighted_samples']['logl']
                self.results['lnz'] =  u_sampler.results['logz']
                self.results['lnz_err'] =  u_sampler.results['logzerr']
   
                # clean up output from the sampler
                os.system(f'rm -r ' + '/'.join(self.fname.split('/')[:-1]) + '/*')
            
            columns = []
            columns.append(fits.Column(name='samples2d', array=self.results['samples2d'], format=f'{self.fitted_model.ndim}D'))
            columns.append(fits.Column(name='lnlike', array=self.results['lnlike'], format='D'))
            hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), 
                header=fits.Header({'EXTNAME':'RESULTS',
                                    'PARAMS':utils.dict_to_str(self.parameters.data),
                                    'LNZ':self.results['lnz'],
                                    'LNZ_ERR':self.results['lnz_err']}))
            hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdulist.writeto(f'{self.fname}brisket_results.fits')

            self.results["median"] = np.median(self.results['samples2d'], axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"], (16, 84), axis=0)

            self._print_results()

            # Create a posterior object to hold the results of the fit.
            self.posterior = Posterior(self.galaxy, run=self.run,
                                       n_samples=self.n_posterior, logger=self.logger)

    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        parameter_len_max = np.max([len(p) for p in self.fitted_model.params])
        parameter_len = np.max([parameter_len_max+2, 25])

        self.logger.info('╔' + '═'*parameter_len + '╦' + '═'*12 + '╦' + '═'*12 + '╦' + '═'*12 + '╦' + '═'*54 + '╗')
        self.logger.info('║ ' + 'Parameter' + ' '*(parameter_len-10) + '║    16th    ║    50th    ║    84th    ║' + ' '*21 + 'Distribution' + ' '*21 + '║')
        self.logger.info('╠' + '═'*parameter_len + '╬' + '═'*12 + '╬' + '═'*12 + '╬' + '═'*12 + '╬' + '═'*54 + '╣')
        for i in range(self.fitted_model.ndim):
            s = "║ "
            s += f"{self.fitted_model.params[i]}" + ' '*(parameter_len-len(self.fitted_model.params[i])-2) 
            s += " ║ "
            
            p00 = self.fitted_model.prior.limits[i][0]
            p99 = self.fitted_model.prior.limits[i][0]
            p16 = self.results['conf_int'][0,i]
            p50 = self.results['median'][i]
            p84 = self.results['conf_int'][1,i]
            sig_digit = int(np.floor(np.log10(np.min([p84-p50,p50-p16]))))-1
            if sig_digit >= 0: 
                p00 = int(np.round(p00, -sig_digit))
                p16 = int(np.round(p16, -sig_digit))
                p50 = int(np.round(p50, -sig_digit))
                p84 = int(np.round(p84, -sig_digit))
                p99 = int(np.round(p99, -sig_digit))
                s += f"{p16:<10d} ║ {p50:<10d} ║ {p84:<10d}"
            else:
                p00 = np.round(p00, -sig_digit)
                p16 = np.round(p16, -sig_digit)
                p50 = np.round(p50, -sig_digit)
                p84 = np.round(p84, -sig_digit)
                p99 = np.round(p99, -sig_digit)
                s += f"{p16:<10} ║ {p50:<10} ║ {p84:<10}"

            s += ' ║ '

            bins = np.linspace(self.fitted_model.prior.limits[i][0], self.fitted_model.prior.limits[i][1], 39)
            ys, _ = np.histogram(self.results['samples2d'][i], bins=bins)
            ys = ys/np.max(ys)
            s += f"{p00:<7}"
            for y in ys:
                if y<1/16: s += ' '
                elif y<3/16: s += '▁'
                elif y<5/16: s += '▂'
                elif y<7/16: s += '▃'
                elif y<9/16: s += '▄'
                elif y<11/16: s += '▅'
                elif y<13/16: s += '▆'
                elif y<15/16: s += '▇'
                else: s += '█'
              
                    
            s += '║'
        
            self.logger.info(s)
        self.logger.info('╚' + '═'*parameter_len + '╩' + '═'*12 + '╩' + '═'*12 + '╩' + '═'*12 + '╩' + '═'*54 + '╝')

        # self.logger.info(f"{'Parameter':<25} {'16th':>10} {'50th':>10} {'84th':>10}")
        # self.logger.info("-"*58)
        # for i in range(self.fitted_model.ndim):
        #     s = f"{self.fitted_model.params[i]:<25}"
        #     if self.results['conf_int'][0, i]<0: s += f" {self.results['conf_int'][0, i]:>9.3f}"
        #     else: s += f"  {self.results['conf_int'][0, i]:>10.3f}"
        #     s += f" {self.results['median'][i]:>10.3f}"
        #     s += f" {self.results['conf_int'][1, i]:>10.3f}"
        #     self.logger.info(s)

    # def plot_corner(self, show=False, save=True):
    #     return plotting.plot_corner(self, show=show, save=save)

    # def plot_1d_posterior(self, show=False, save=True):
    #     return plotting.plot_1d_posterior(self, show=show, save=save)

    # def plot_sfh_posterior(self, show=False, save=True, colorscheme="bw"):
    #     return plotting.plot_sfh_posterior(self, show=show, save=save,
    #                                        colorscheme=colorscheme)

    # def plot_spectrum_posterior(self, show=False, save=True):
    #     return plotting.plot_spectrum_posterior(self, show=show, save=save)

    # def plot_calibration(self, show=False, save=True):
    #     return plotting.plot_calibration(self, show=show, save=save)
