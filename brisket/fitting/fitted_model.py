from __future__ import print_function, division, absolute_import

import numpy as np
import time

from copy import deepcopy

# from brisket.fitting.prior import Prior, dirichlet
# from brisket.fitting.calibration import calib_model
# from brisket.fitting.noise import noise_model
from brisket.models.model_galaxy import ModelGalaxy


class FittedModel(object):
    """ Contains a model which is to be fitted to observational data.

    Parameters
    ----------

    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    parameters : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.
    """

    def __init__(self, galaxy, parameters, time_calls=False):

        self.galaxy = galaxy
        self.parameters = deepcopy(parameters)
        self.time_calls = time_calls

        self._set_constants()
        self.params = self.parameters.free_param_names    # Names of parameters to be fit
        self.limits = self.parameters.free_param_limits   # Limits for fitted parameter values
        self.pdfs = self.parameters.free_param_pdfs       # Probability densities within lims
        self.hypers = self.parameters.free_param_hypers  # Hyperparameters of prior distributions
        self.mirrors = self.parameters.free_param_mirrors   # Params which mirror a fitted param
        self.transforms = self.parameters.free_param_transforms   # Params which are a transform of another param
        self.ndim = self.parameters.ndim

        # self.prior = Prior(self.limits, self.pdfs, self.hypers)
        self.prior = PriorVolume(self.parameters.free_param_priors)

        # Initialize the ModelGalaxy object with a set of parameters randomly drawn from the prior cube
        self._update_model_parameters(self.prior.sample())
        self.model_galaxy = ModelGalaxy(self.parameters, filt_list=self.galaxy.filt_list, spec_wavs=self.galaxy.spec_wavs, 
                                        wav_units=self.galaxy.wav_units, spec_units=self.galaxy.spec_units, phot_units=self.galaxy.phot_units)

        if self.time_calls:
            self.times = np.zeros(1000)
            self.n_calls = 0

    # def _process_fit_instructions(self): ### DEPRICATED - all functionality moved to parameters module
    #     all_keys = []           # All keys in fit_instructions and subs
    #     all_vals = []           # All vals in fit_instructions and subs

    #     self.params = []        # Parameters to be fitted
    #     self.limits = []        # Limits for fitted parameter values
    #     self.pdfs = []          # Probability densities within lims
    #     self.hyper_params = []  # Hyperparameters of prior distributions
    #     self.mirror_pars = {}   # Params which mirror a fitted param
    #     self.ndim = len(self.params)

    #     # # # Flatten the input fit_instructions dictionary.
    #     # # for key in list(self.fit_instructions):
    #     # #     if not isinstance(self.fit_instructions[key], dict):
    #     # #         all_keys.append(key)
    #     # #         all_vals.append(self.fit_instructions[key])
    #     # #     else:
    #     # #         for sub_key in list(self.fit_instructions[key]):
    #     # #             if not isinstance(self.fit_instructions[key][sub_key], dict):
    #     # #                 all_keys.append(f'{key}:{sub_key}')
    #     # #                 all_vals.append(self.fit_instructions[key][sub_key])
    #     # #             else:
    #     # #                 for sub_sub_key in list(self.fit_instructions[key][sub_key]):
    #     # #                     all_keys.append(f'{key}:{sub_key}:{sub_sub_key}')
    #     # #                     all_vals.append(self.fit_instructions[key][sub_key][sub_sub_key])

    #     # # Sort the resulting lists alphabetically by parameter name.
    #     # indices = np.argsort(all_keys)
    #     # all_vals = [all_vals[i] for i in indices]
    #     # all_keys.sort()
        
    #     # for i in range(len(all_keys)):
    #     #     print(i, all_keys[i], all_vals[i])
    #     # quit()

    #     # # Find parameters to be fitted and extract their priors.
    #     # for i in range(len(all_vals)):
    #     #     if isinstance(all_vals[i], tuple):
    #     #         self.params.append(all_keys[i])
    #     #         self.limits.append(all_vals[i])  # Limits on prior.

    #     #         # Prior probability densities between these limits.
    #     #         prior_key = all_keys[i] + "_prior"
    #     #         if prior_key in list(all_keys):
    #     #             self.pdfs.append(all_vals[all_keys.index(prior_key)])

    #     #         else:
    #     #             self.pdfs.append("uniform")

    #     #         # Any hyper-parameters of these prior distributions.
    #     #         self.hyper_params.append({})
    #     #         for i in range(len(all_keys)):
    #     #             if all_keys[i].startswith(prior_key + "_"):
    #     #                 hyp_key = all_keys[i][len(prior_key)+1:]
    #     #                 self.hyper_params[-1][hyp_key] = all_vals[i]

    #         # # Find any parameters which mirror the value of a fit param.
    #         # if all_vals[i] in all_keys:
    #         #     self.mirror_pars[all_keys[i]] = all_vals[i]

    #         # if all_vals[i] == "dirichlet":
    #         #     n = all_vals[all_keys.index(all_keys[i][:-6])]
    #         #     comp = all_keys[i].split(":")[0]
    #         #     for j in range(1, n):
    #         #         self.params.append(comp + ":dirichletr" + str(j))
    #         #         self.pdfs.append("uniform")
    #         #         self.limits.append((0., 1.))
    #         #         self.hyper_params.append({})

    #     # Find the dimensionality of the fit
    #     # self.ndim = len(self.params)

    def _set_constants(self):
        """ Calculate constant factors used in the lnlike function. """

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

        # if self.galaxy.index_list is not None:
        #     log_error_factors = np.log(2*np.pi*self.galaxy.indices[:, 1]**2)
        #     self.K_ind = -0.5*np.sum(log_error_factors)
        #     self.inv_sigma_sq_ind = 1./self.galaxy.indices[:, 1]**2

    def lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-likelihood for a given parameter vector. """

        self._update_model_galaxy(x)
    
        lnlike = 0.

        if self.galaxy.spectrum_exists:# and self.galaxy.index_list is None:
            lnlike += self._lnlike_spec()

        if self.galaxy.photometry_exists:
            lnlike += self._lnlike_phot()

        # if self.galaxy.index_list is not None:
        #     lnlike += self._lnlike_indices()

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        # # Functionality for timing likelihood calls.
        # if self.time_calls:
        #     self.times[self.n_calls] = time.time() - time0
        #     self.n_calls += 1

        #     if self.n_calls == 1000:
        #         self.n_calls = 0
        #         print("Mean likelihood call time:", np.round(np.mean(self.times), 4))
        #         print("Wall time per lnlike call:", np.round((time.time() - self.wall_time0)/1000., 4))

        return lnlike

    def _lnlike_phot(self):
        """ Calculates the log-likelihood for photometric data. """

        diff = (self.galaxy.photometry[:, 1] - self.model_galaxy.photometry)**2
        self.chisq_phot = np.sum(diff*self.inv_sigma_sq_phot)

        return self.K_phot - 0.5*self.chisq_phot

    def _lnlike_spec(self):
        """ Calculates the log-likelihood for spectroscopic data. This
        includes options for fitting flexible spectral calibration and
        covariant noise models. """

        model = self.model_galaxy.spectrum
        if any(np.isnan(model)):
            print(self.parameters.data)
            quit()

        # Calculate differences between model and observed spectrum
        diff = (self.galaxy.spectrum[:, 1] - model)
        # print(any(np.isnan(diff)))

        # if "noise" in list(self.fit_instructions):
        #     if self.galaxy.spec_cov is not None:
        #         raise ValueError("Noise modelling is not currently supported "
        #                          "with manually specified covariance matrix.")

        #     self.noise = noise_model(self.model_components["noise"],
        #                              self.galaxy, model)
        # else:
        self.noise = noise_model({}, self.galaxy, model)


        if self.noise.corellated:
            lnlike_spec = self.noise.gp.lnlikelihood(self.noise.diff)

            return lnlike_spec

        else:
            # Allow for calculation of chi-squared with direct input
            # covariance matrix - experimental!
            # if self.galaxy.spec_cov is not None:
            #     diff_cov = np.dot(diff.T, self.galaxy.spec_cov_inv)
            #     self.chisq_spec = np.dot(diff_cov, diff)

            #     return -0.5*self.chisq_spec

            self.chisq_spec = np.sum(self.noise.inv_var*diff**2)

            # if "noise" in list(self.fit_instructions):
            #     c_spec = -np.log(self.model_components["noise"]["scaling"])
            #     K_spec = self.galaxy.spectrum.shape[0]*c_spec

            # else:
            K_spec = 0.

            return K_spec - 0.5*self.chisq_spec

    def _lnlike_indices(self):
        """ Calculates the log-likelihood for spectral indices. """

        diff = (self.galaxy.indices[:, 0] - self.model_galaxy.indices)**2
        self.chisq_ind = np.sum(diff*self.inv_sigma_sq_ind)

        return self.K_ind - 0.5*self.chisq_ind

    def _update_model_parameters(self, param):
        """ Generates a model object with the current parameters. """
        # for i in range(len(self.params)):
        #     print(self.params[i], param[i])
        # quit()

        # dirichlet_comps = []

        # Substitute values of fit params from param into model_comp.

        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                self.parameters[self.params[i]] = param[i]
            elif len(split) == 2:
                self.parameters[split[0]][split[1]] = param[i]
            elif len(split) == 3:
                self.parameters[split[0]][split[1]][split[2]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirrors):
            
            if key in list(self.transforms):
                if isinstance(self.transforms[key], (int,float)):
                    transform = lambda x: x*self.transforms[key]
                elif callable(self.transforms[key]): 
                    transform = self.transforms[key]
                else: 
                    raise Exception()
            else:
                transform = lambda x: x
                    

            split_par = key.split(":")
            split_val = self.mirrors[key].split(":")

            if len(split_val)==1:
                fit_val = self.parameters[split_val[0]]
            elif len(split_val)==2:
                fit_val = self.parameters[split_val[0]][split_val[1]]
            elif len(split_val)==3:
                fit_val = self.parameters[split_val[0]][split_val[1]][split_val[2]]

            if len(split_par)==1:
                self.parameters[split_par[0]] = transform(fit_val)
            elif len(split_par)==2:
                self.parameters[split_par[0]][split_par[1]] = transform(fit_val)
            elif len(split_par)==3:
                self.parameters[split_par[0]][split_par[1]][split_par[2]] = transform(fit_val)
        
    def _update_model_galaxy(self, param):
        self._update_model_parameters(param)
        self.model_galaxy.update(self.parameters)
