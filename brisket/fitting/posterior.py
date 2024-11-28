from __future__ import print_function, division, absolute_import

import numpy as np

import os

from copy import deepcopy

# from brisket.fitting.fitted_model import FittedModel
# from brisket.fitting.prior import dirichlet

# from brisket.models.star_formation_history import StarFormationHistoryModel
from brisket.models.model_galaxy import ModelGalaxy
from brisket.parameters import Params

from brisket import utils

from astropy.io import fits
from astropy.table import Table, Column
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

class Posterior(object):
    """ Provides access to the outputs from fitting models to data and
    calculating posterior predictions for derived parameters (e.g. for
    star-formation histories, rest-frane magnitudes etc).

    Parameters
    ----------
    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    n_samples : float - optional
        The number of posterior samples to generate for each quantity.
    """

    def __init__(self, galaxy, run=".", n_samples=500, logger=utils.NullLogger):

        self.galaxy = galaxy
        self.run = run
        self.n_samples = n_samples
        self.logger = logger

        self.fname = f"brisket/posterior/{self.run}/{self.galaxy.ID}_brisket_results.fits"

        # Check to see whether the object has been fitted.
        if not os.path.exists(self.fname):
            raise IOError(f"Fit results not found for {self.galaxy.ID}.")

        with fits.open(self.fname) as f:
            hdu = f['RESULTS']
            # Reconstruct the fitted model.
            self.parameters = Params(utils.str_to_dict(hdu.header['PARAMS']))
            # 2D array of samples for the fitted parameters only.
            self.samples2d = deepcopy(hdu.data['samples2d'])

        self.fitted_model = FittedModel(self.galaxy, self.parameters)

        # If fewer than n_samples exist in posterior, reduce n_samples
        if self.samples2d.shape[0] < self.n_samples:
            self.n_samples = self.samples2d.shape[0]

        # Randomly choose points to generate posterior quantities
        self.indices = np.random.choice(self.samples2d.shape[0],
                                        size=self.n_samples, replace=False)

        self.samples = {}  # Store all posterior samples

        # Add 1D posteriors for fitted params to the samples dictionary
        for i in range(self.fitted_model.ndim):
            param_name = self.fitted_model.params[i]
            self.samples[param_name] = self.samples2d[self.indices, i]

        # self.get_dirichlet_tx(dirichlet_comps)

        self._compute_posterior_quantities()


    def _compute_posterior_quantities(self):

        self.fitted_model._update_model_galaxy(self.samples2d[0, :])
        self.fitted_model.model_galaxy._compute_properties()
        for key in self.fitted_model.model_galaxy.properties:
            try: # for arrays
                l = len(self.fitted_model.model_galaxy.properties[key])
                self.samples[key] = np.zeros((self.n_samples, l))
            except TypeError: # for keys with no len() (i.e., floats)
                self.samples[key] = np.zeros(self.n_samples)

        
        # for q in quantity_names:
        #     size = getattr(self.model_galaxy, q).shape[0]
        #     self.samples[q] = np.zeros((self.n_samples, size))

        # if self.galaxy.photometry_exists:
        #     self.samples["chisq_phot"] = np.zeros(self.n_samples)

        # if "dust_atten" in list(self.fitted_model.model_components):
        #     size = self.model_galaxy.spectrum_full.shape[0]
        #     self.samples["dust_curve"] = np.zeros((self.n_samples, size))

        # if "calib" in list(self.fitted_model.model_components):
        #     size = self.model_galaxy.spectrum.shape[0]
        #     self.samples["calib"] = np.zeros((self.n_samples, size))

        # if "noise" in list(self.fitted_model.model_components):
        #     type = self.fitted_model.model_components["noise"]["type"]
        #     if type.startswith("GP"):
        #         size = self.model_galaxy.spectrum.shape[0]
        #         self.samples["noise"] = np.zeros((self.n_samples, size))
        self.logger.info('Computing derived posterior properties...')
        with logging_redirect_tqdm(loggers=[self.logger]):
            for i in tqdm(range(self.n_samples)):
                param = self.samples2d[self.indices[i], :]
                self.fitted_model._update_model_galaxy(param)
                self.fitted_model.model_galaxy._compute_properties()
                for key in self.fitted_model.model_galaxy.properties:
                    self.samples[key][i] = self.fitted_model.model_galaxy.properties[key]
            
            # if self.galaxy.photometry_exists:
            #     self.samples["chisq_phot"][i] = self.fitted_model.chisq_phot

            # if "dust_atten" in list(self.fitted_model.model_components):
            #     dust_curve = self.fitted_model.model_galaxy.dust_atten.A_cont
            #     self.samples["dust_curve"][i] = dust_curve

            # if "calib" in list(self.fitted_model.model_components):
            #     self.samples["calib"][i] = self.fitted_model.calib.model

            # if "noise" in list(self.fitted_model.model_components):
            #     type = self.fitted_model.model_components["noise"]["type"]
            #     if type.startswith("GP"):
            #         self.samples["noise"][i] = self.fitted_model.noise.mean()

            # for q in quantity_names:
            #     if q == "spectrum":
            #         spectrum = getattr(self.fitted_model.model_galaxy, q)[:, 1]
            #         self.samples[q][i] = spectrum
            #         continue

            #     self.samples[q][i] = getattr(self.fitted_model.model_galaxy, q)

    def _write_posterior_quantities(self):
        with fits.open(self.fname) as hdul:
            extensions = [hdu.name for hdu in hdul]

            if not 'SED_MED' in extensions:
                self.logger.info(f'Writing posterior quantities to {self.fname}')
                # MEDIAN SED | should have wav_rest, f_nu_16, f_nu_50, f_nu_84, f_lam_16, f_lam_50, f_lam_84# + convolved
                header = fits.Header({'EXTNAME':'SED_MED'})
                columns = []
                columns.append(fits.Column(name='wav_rest', array=self.fitted_model.model_galaxy.wav_rest, format='D'))
                columns.append(fits.Column(name='f_lam_16', array=np.percentile(self.samples['SED'], 16, axis=0), format='D'))
                columns.append(fits.Column(name='f_lam_50', array=np.percentile(self.samples['SED'], 50, axis=0), format='D'))
                columns.append(fits.Column(name='f_lam_84', array=np.percentile(self.samples['SED'], 84, axis=0), format='D'))
                hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), header=header)
                hdul.append(hdu) 


                # header = fits.Header({'EXTNAME':'SED_MAP'}) # should have wav_rest, wav_obs, f_nu, f_lam, and parameters in header #### seds for each subset? 
                if self.fitted_model.model_galaxy.phot_output:
                    header = fits.Header({'EXTNAME':'PHOT'}) # should have filter, wav_obs, f_nu_obs, f_nu_obs_err, f_nu_mod (array?) 
                    columns = []
                    columns.append(fits.Column(name='filter', array=self.galaxy.filt_list, format=f'{np.max([len(f) for f in self.galaxy.filt_list])}A'))
                    columns.append(fits.Column(name='wav', array=self.galaxy.photometry[:,0], format='D'))
                    columns.append(fits.Column(name='flux', array=self.galaxy.photometry[:,1], format='D'))
                    columns.append(fits.Column(name='flux_err', array=self.galaxy.photometry[:,2], format='D'))
                    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), header=header)
                    hdul.append(hdu) 
                
                if self.fitted_model.model_galaxy.spec_output:
                    header = fits.Header({'EXTNAME':'SPEC'}) # should have wav_rest, wav_obs, f_nu_obs, f_nu_obs_err, f_nu_mod (array?)
                    columns = []
                    columns.append(fits.Column(name='filter', array=self.galaxy.filt_list, format=f'{np.max([len(f) for f in self.galaxy.filt_list])}A'))
                    columns.append(fits.Column(name='wav', array=self.galaxy.photometry[:,0], format='D'))
                    columns.append(fits.Column(name='flux', array=self.galaxy.photometry[:,1], format='D'))
                    columns.append(fits.Column(name='flux_err', array=self.galaxy.photometry[:,2], format='D'))
                    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), header=header)
                    hdul.append(hdu) 
                
                # properties
                header = fits.Header({'EXTNAME':'PROPS'}) # should have wav_rest, wav_obs, f_nu_obs, f_nu_obs_err, f_nu_mod (array?)
                columns = []
                columns.append(fits.Column(name='beta_UV', array=self.samples['beta_UV'], format='D'))
                columns.append(fits.Column(name='M_UV', array=self.samples['M_UV'], format='D'))
                columns.append(fits.Column(name='m_UV', array=self.samples['m_UV'], format='D'))


            
            

            hdul.writeto(self.fname, overwrite=True)

        
        # t1 = fits.BinTableHDU(data=fits.ColDefs(columns), 
        #                       header=fits.Header({'EXTNAME':'SED_MAP'})) # should have wav_rest, wav_obs, f_nu, f_lam, and parameters in header #### seds for each subset? 


        # t2 = fits.BinTableHDU(data=fits.ColDefs(columns),
        #                       header=fits.Header({'EXTNAME':'PROPS'})) # samples from the posterior distribution for free parameters, and derived physical properties at each iteration
        # t3 = fits.BinTableHDU(data=fits.ColDefs(columns),
        #                       header=fits.Header({'EXTNAME':'ATTEN'})) # the posterior dust attenuation curve
        # t4 = fits.BinTableHDU(data=fits.ColDefs(columns),
        #                       header=fits.Header({'EXTNAME':'SFH'})) # the posterior SFH 


    # def get_basic_quantities(self):
    #     """Calculates basic posterior quantities, these are fast as they
    #     are derived only from the SFH model, not the spectral model. """

    #     if "stellar_mass" in list(self.samples):
    #         return

    #     self.fitted_model._update_model_components(self.samples2d[0, :])
    #     self.sfh = star_formation_history(self.fitted_model.model_components)

    #     quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
    #                       "mass_weighted_age", "tform", "tquench"]

    #     for q in quantity_names:
    #         self.samples[q] = np.zeros(self.n_samples)

    #     self.samples["sfh"] = np.zeros((self.n_samples,
    #                                     self.sfh.ages.shape[0]))

    #     quantity_names += ["sfh"]

    #     for i in range(self.n_samples):
    #         param = self.samples2d[self.indices[i], :]
    #         self.fitted_model._update_model_components(param)
    #         self.sfh.update(self.fitted_model.model_components)

    #         for q in quantity_names:
    #             self.samples[q][i] = getattr(self.sfh, q)

    # def get_advanced_quantities(self):
    #     """Calculates advanced derived posterior quantities, these are
    #     slower because they require the full model spectra. """

    #     if "spectrum_full" in list(self.samples):
    #         return

    #     self.fitted_model._update_model_components(self.samples2d[0, :])
    #     self.model_galaxy = model_galaxy(self.fitted_model.model_components,
    #                                      filt_list=self.galaxy.filt_list,
    #                                      spec_wavs=self.galaxy.spec_wavs,
    #                                      index_list=self.galaxy.index_list)

    #     all_names = ["photometry", "spectrum", "spectrum_full", "uvj",
    #                  "indices"]

    #     all_model_keys = dir(self.model_galaxy)
    #     quantity_names = [q for q in all_names if q in all_model_keys]

    #     for q in quantity_names:
    #         size = getattr(self.model_galaxy, q).shape[0]
    #         self.samples[q] = np.zeros((self.n_samples, size))

    #     if self.galaxy.photometry_exists:
    #         self.samples["chisq_phot"] = np.zeros(self.n_samples)

    #     if "dust_atten" in list(self.fitted_model.model_components):
    #         size = self.model_galaxy.spectrum_full.shape[0]
    #         self.samples["dust_curve"] = np.zeros((self.n_samples, size))

    #     if "calib" in list(self.fitted_model.model_components):
    #         size = self.model_galaxy.spectrum.shape[0]
    #         self.samples["calib"] = np.zeros((self.n_samples, size))

    #     if "noise" in list(self.fitted_model.model_components):
    #         type = self.fitted_model.model_components["noise"]["type"]
    #         if type.startswith("GP"):
    #             size = self.model_galaxy.spectrum.shape[0]
    #             self.samples["noise"] = np.zeros((self.n_samples, size))

    #     for i in range(self.n_samples):
    #         param = self.samples2d[self.indices[i], :]
    #         self.fitted_model._update_model_components(param)
    #         self.fitted_model.lnlike(param)

    #         if self.galaxy.photometry_exists:
    #             self.samples["chisq_phot"][i] = self.fitted_model.chisq_phot

    #         if "dust_atten" in list(self.fitted_model.model_components):
    #             dust_curve = self.fitted_model.model_galaxy.dust_atten.A_cont
    #             self.samples["dust_curve"][i] = dust_curve

    #         if "calib" in list(self.fitted_model.model_components):
    #             self.samples["calib"][i] = self.fitted_model.calib.model

    #         if "noise" in list(self.fitted_model.model_components):
    #             type = self.fitted_model.model_components["noise"]["type"]
    #             if type.startswith("GP"):
    #                 self.samples["noise"][i] = self.fitted_model.noise.mean()

    #         for q in quantity_names:
    #             if q == "spectrum":
    #                 spectrum = getattr(self.fitted_model.model_galaxy, q)[:, 1]
    #                 self.samples[q][i] = spectrum
    #                 continue

    #             self.samples[q][i] = getattr(self.fitted_model.model_galaxy, q)

    # def predict(self, filt_list=None, spec_wavs=None, spec_units="ergscma",
    #             phot_units="ergscma", index_list=None):
    #     """Obtain posterior predictions for new observables not included
    #     in the data. """

    #     self.prediction = {}

    #     self.fitted_model._update_model_components(self.samples2d[0, :])
    #     model = model_galaxy(self.fitted_model.model_components,
    #                          filt_list=filt_list, phot_units=phot_units,
    #                          spec_wavs=spec_wavs, index_list=index_list)

    #     all_names = ["photometry", "spectrum", "indices"]

    #     all_model_keys = dir(model)
    #     quantity_names = [q for q in all_names if q in all_model_keys]

    #     for q in quantity_names:
    #         size = getattr(model, q).shape[0]
    #         self.prediction[q] = np.zeros((self.n_samples, size))

    #     for i in range(self.n_samples):
    #         param = self.samples2d[self.indices[i], :]
    #         self.fitted_model._update_model_components(param)
    #         model.update(self.fitted_model.model_components)

    #         for q in quantity_names:
    #             if q == "spectrum":
    #                 spectrum = getattr(model, q)[:, 1]
    #                 self.prediction[q][i] = spectrum
    #                 continue

    #             self.prediction[q][i] = getattr(model, q)

    # def predict_basic_quantities_at_redshift(self, redshift,
    #                                          sfh_type="dblplaw"):
    #     """ Predicts basic (SFH-based) quantities at a specified higher
    #     redshift. This is a bit experimental, there's probably a better
    #     way. Only works for models with a single SFH component. """

    #     self.prediction_at_z = {}

    #     #if "stellar_mass" in list(self.prediction_at_z):
    #     #    return

    #     self.fitted_model._update_model_components(self.samples2d[0, :])
    #     self.sfh = star_formation_history(self.fitted_model.model_components)

    #     quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
    #                       "mass_weighted_age", "tform", "tquench"]

    #     for q in quantity_names:
    #         self.prediction_at_z[q] = np.zeros(self.n_samples)

    #     self.prediction_at_z["sfh"] = np.zeros((self.n_samples,
    #                                             self.sfh.ages.shape[0]))

    #     quantity_names += ["sfh"]

    #     for i in range(self.n_samples):
    #         param = self.samples2d[self.indices[i], :]
    #         self.fitted_model._update_model_components(param)
    #         self.sfh.update(self.fitted_model.model_components)

    #         formed_mass_at_z = self.sfh.massformed_at_redshift(redshift)

    #         model_comp = deepcopy(self.fitted_model.model_components)

    #         model_comp["redshift"] = redshift
    #         model_comp[sfh_type]["massformed"] = formed_mass_at_z

    #         self.sfh.update(model_comp)

    #         for q in quantity_names:
    #             self.prediction_at_z[q][i] = getattr(self.sfh, q)
