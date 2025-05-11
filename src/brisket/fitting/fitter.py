'''
This module defines the Fitter class, which is the primary interface to fitting SEDs.
'''
from __future__ import annotations 

import numpy as np
import os, sys, time, warnings
from copy import deepcopy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..parameters import Params
    from .observation import Observation

from ..utils.console import setup_logger
from ..models.core import Model
from .priors import PriorVolume
# from .results import Results


# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0

class Fitter:
    def __init__(self, 
                 params: Params, 
                 obs: Observation, 
                 run: str, 
                 outdir: str = None,
                 n_posterior: int = 500, 
                 verbose: bool = False):

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        self.verbose = verbose
        self.run = run
        self.obs = obs
        assert self.obs.ID is not None, "Observation must have associated ID to fit model"

        self.params = deepcopy(params)
        self.n_posterior = n_posterior

        # Set up the output directory # TODO add rank==0 logic        
        self.outpath = f'harmonized/{run}/'
        if outdir is not None:
            assert os.path.exists(outdir), "outdir doesn't exist"
            self.outpath = os.path.join(outdir, self.outpath) # append outdir to beginning of path
        
        # A dictionary containing properties of the model to be saved.
        # self.results = {}

        # Set up the model which is to be fitted to the data.
        # self.fitted_model = FittedModel(galaxy, self.parameters,
        #                                time_calls=time_calls)

        # self._set_constants()
        self.ndim = self.params.ndim
        self.param_names = self.params.free_param_names
        # self.mirrors = self.parameters.free_param_mirrors   # Params which mirror a fitted param
        # self.transforms = self.parameters.free_param_transforms   # Params which are a transform of another param

        self.prior = PriorVolume(self.params.free_param_priors)

        # Initialize the Model object with a set of parameters randomly drawn from the prior cube
        self.params.update_from_vector(self.param_names, self.prior.sample())
        self.mod = Model(self.params, self.obs, verbose=verbose)

    def _update_model(self, x):
        '''
        Vectorization note: x can be a 1D array of size ndim, or a 2D array of size (?, ndim).
        '''
        self.params.update_from_vector(self.param_names, x)
        self.mod.update(self.params) 

    def _set_constants(self):
        """ Calculate constant factors used in the lnlike function. """

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

    def lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-likelihood for a given parameter vector. """

        self._update_model(x)

        lnlike = 0.

        for i in range(self.obs.N_spec):
            spec_obs = self.obs.spec_list[i]
            spec_mod = self.mod.obs.spec_list[i]
            # TODO implement
            # lnlike += ...
        
        for i in range(self.obs.N_phot):
            ymod = self.mod.phot['total'].value
            yobs = self.obs.phot['fnu'].value
            yobs_err = self.obs.phot['fnu_err'].value
            K_phot = -0.5*np.sum(np.log(2*np.pi*yobs_err**2))
            lnlike += K_phot - 0.5 * np.sum((yobs - ymod)**2 / (yobs_err**2))

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        return lnlike


    def fit(self, 
            sampler: str = "ultranest", 
            verbose: bool = False, 
            overwrite: bool = False,
            **kwargs):
        """ Fit the specified model to the input galaxy data.
        Args:
            sampler (str, optional)
                Which sampler to use. Options are "ultranest". Default is "ultranest".
            For ultranest, these are:
                nsteps (int, default: 4)

        """
        self.overwrite = overwrite
        self.basename = os.path.join(self.outpath, f'{self.obs.ID}_brisket_results')

        if os.path.exists(self.basename + '.fits') and not overwrite:
            # if rank == 0:
            #     self.logger.info(f'Fitting not performed as results have already been loaded from {self.fname}brisket_results.fits. To start over delete this file or change run.')
            self.results = Results.load(file=filepath)
            print(self.results)
            return self.results

        # Figure out which sampling algorithm to use
        sampler = sampler.lower()

        if sampler not in ["ultranest"]:
            e = ValueError(f"Sampler {sampler} not supported.")
            self.logger.error(e)
            raise e

        if rank == 0 or not use_MPI:
            self.logger.info(f'Fitting object') # TODO implement object ID
            start_time = time.time()

        with warnings.catch_warnings():# and os.environ['PYTHONWARNINGS'] as 'ignore':
            warnings.simplefilter('ignore')

            if sampler == 'multinest':
                self._run_multinest(**kwargs)

            elif sampler == 'nautilus':
                self._run_nautilus(**kwargs)

            elif sampler == 'ultranest':
                u_sampler = self._run_ultranest(**kwargs)

        if rank == 0 or not use_MPI:
            
            runtime = time.time() - start_time
            if runtime > 60:
                runtime = f'{int(np.floor(runtime/60))}m{runtime-np.floor(runtime/60)*60:.1f}s'
            else: 
                runtime = f'{runtime:.1f} seconds'

            self.logger.info(f'Completed in {runtime}.')

            # Load sampler outputs 
            sampler_output = {}
            if sampler == "multinest":
                samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
                lnz_line = open(self.fname + "stats.dat").readline().split()
                sampler_output["samples2d"] = samples2d[:, :-1]
                sampler_output["lnlike"] = samples2d[:, -1]
                sampler_output["lnz"] = float(lnz_line[-3])
                sampler_output["lnz_err"] = float(lnz_line[-1])
                
                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == "nautilus":
                samples2d = np.zeros((0, self.fitted_model.ndim))
                log_l = np.zeros(0)
                while len(samples2d) < self.n_posterior:
                    result = n_sampler.posterior(equal_weight=True)
                    samples2d = np.vstack((samples2d, result[0]))
                    log_l = np.concatenate((log_l, result[2]))
                sampler_output["samples2d"] = samples2d
                sampler_output["lnlike"] = log_l
                sampler_output["lnz"] = n_sampler.log_z
                sampler_output["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)

                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == 'ultranest':
                sampler_output['samples2d'] = u_sampler.results['samples']
                sampler_output['lnlike'] = u_sampler.results['weighted_samples']['logl']
                sampler_output['lnz'] =  u_sampler.results['logz']
                sampler_output['lnz_err'] =  u_sampler.results['logzerr']
   
                # clean up output from the sampler
                # os.system(f'rm -r ' + '/'.join(self.basename.split('/')[:-1]) + '/*')
            
            # columns = []
            # columns.append(fits.Column(name='samples2d', array=self.results['samples2d'], format=f'{self.fitted_model.ndim}D'))
            # columns.append(fits.Column(name='lnlike', array=self.results['lnlike'], format='D'))
            # hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), 
            #     header=fits.Header({'EXTNAME':'RESULTS',
            #                         'PARAMS':utils.dict_to_str(self.parameters.data),
            #                         'LNZ':self.results['lnz'],
            #                         'LNZ_ERR':self.results['lnz_err']}))
            # hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
            # hdulist.writeto(f'{self.fname}brisket_results.fits')

            # self.results["median"] = np.median(self.results['samples2d'], axis=0)
            # self.results["conf_int"] = np.percentile(self.results["samples2d"], (16, 84), axis=0)

            self.results = Results(
                 model=self.mod,
                 param_names=self.param_names,
                 sampler_output=sampler_output,
                 n_posterior=self.n_posterior, 
                 run=self.run,
            )
            print(self.results)

            return self.results

    def _run_ultranest(self, nsteps_per_param=4, n_live=100):
        try:
            from ultranest import ReactiveNestedSampler
            from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
        except (ImportError, RuntimeError, SystemExit):
            self.logger.error('Ultranest import failed.')


         # os.environ['OMP_NUM_THREADS'] = '1'
        resume = 'resume'
        if self.overwrite:
            resume = 'overwrite'

        u_sampler = ReactiveNestedSampler(self.param_names, 
                                        self.lnlike, 
                                        transform=self.prior.transform, 
                                        log_dir='/'.join(self.basename.split('/')[:-1]), 
                                        resume=resume, 
                                        run_num=None)
        u_sampler.stepsampler = SliceSampler(nsteps=nsteps_per_param*self.ndim,
                                             generate_direction=generate_mixture_random_direction)
        u_sampler.run(
            min_num_live_points=n_live,
            dlogz=0.5, # desired accuracy on logz -- could allow to specify
            min_ess=self.n_posterior, # number of effective samples
            # update_interval_volume_fraction=0.4, # how often to update region
            # max_num_improvement_loops=3, # how many times to go back and improve
        )
        return u_sampler

