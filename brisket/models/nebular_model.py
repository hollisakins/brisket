from __future__ import print_function, division, absolute_import

import numpy as np

from brisket import config
from brisket import utils

from brisket.models.star_formation_history import StarFormationHistoryModel


class NebularModel(object):
    """ Allows access to and maniuplation of nebular emission models.
    These must be pre-computed using Cloudy and the relevant set of
    stellar emission models. This has already been done for the default
    stellar models.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the stellar models.
    """

    def __init__(self, wavelengths, params, logger=utils.NullLogger):
        self.wavelengths = wavelengths
        self.logger = logger
        
        self.flag = True
        if not 'nebular' in list(params):
            self.logger.info("Skipping nebular emission module")
            self.flag = False; return
        
        self.model = params['stellar_model']
        self.logger.info(f"Initializing nebular emission module".ljust(50) + f'(model: {self.model})'.rjust(20))

        if 'metallicity' in list(params['nebular']):
            self.logger.info("Computing independent chemical enrichment history for nebular models")
            self.neb_sfh = StarFormationHistoryModel(params)

        self.metallicities = config.stellar_models[self.model]['metallicities']
        self.logU = config.nebular_models[self.model]['logU']
        self.neb_ages = config.nebular_models[self.model]['neb_ages']
        self.neb_wavs = config.nebular_models[self.model]['neb_wavs']
        self.cont_grid = config.nebular_models[self.model]['cont_grid']
        self.line_grid = config.nebular_models[self.model]['line_grid']

        self.combined_grid, self.line_grid = self._setup_grids()

    def _gauss(self, x, L, mu, fwhm, fwhm_unit='A'):
        if fwhm_unit=='A':
            sigma = fwhm/2.355
            y = np.exp(-(x-mu)**2/(2*sigma**2))
            y *= L/np.sum(y)
        if fwhm_unit=='kms':
            sigma = mu * fwhm/2.998e5 / 2.355
            y = np.exp(-(x-mu)**2/(2*sigma**2))
            y *= L/np.trapz(y,x=x)
        return y


    def _setup_grids(self):
        """ Loads Cloudy nebular continuum grid and resamples to the
        input wavelengths. Loads nebular line grids and adds line fluxes
        to the correct pixels in order to create a combined grid. """
        
        self.logger.debug("Setting up nebular emission grid")

        fwhm = config.fwhm

        comb_grid = np.zeros((self.wavelengths.shape[0],
                              self.metallicities.shape[0],
                              self.logU.shape[0],
                              self.neb_ages.shape[0]))

        line_grid = np.zeros((config.line_wavs.shape[0],
                              self.metallicities.shape[0],
                              self.logU.shape[0],
                              self.neb_ages.shape[0]))

        for i in range(self.metallicities.shape[0]):
            for j in range(self.logU.shape[0]):

                hdu_index = self.metallicities.shape[0]*j + i
                
                raw_cont_grid = self.cont_grid[hdu_index]
                raw_line_grid = self.line_grid[hdu_index]

                line_grid[:, i, j, :] = raw_line_grid[1:, 1:].T

                for k in range(self.neb_ages.shape[0]):
                    comb_grid[:, i, j, k] = np.interp(self.wavelengths,
                                                      self.neb_wavs,
                                                      raw_cont_grid[k+1, 1:],
                                                      left=0, right=0)

        # Add the nebular lines to the resampled nebular continuum grid.
        # for i in range(config.line_wavs.shape[0]):
        #     ind = np.abs(self.wavelengths - config.line_wavs[i]).argmin()
        #     if ind != 0 and ind != self.wavelengths.shape[0]-1:
        #         width = (self.wavelengths[ind+1] - self.wavelengths[ind-1])/2
        #         comb_grid[ind, :, :, :] += line_grid[i, :, :, :]/width
        
        for i in range(config.line_wavs.shape[0]):
            ind = np.abs(self.wavelengths - config.line_wavs[i]).argmin()
            if ind != 0 and ind != self.wavelengths.shape[0]-1:
                for j in range(self.metallicities.shape[0]):
                    for k in range(self.logU.shape[0]):
                        for l in range(self.neb_ages.shape[0]):
                            comb_grid[:, j, k, l] += self._gauss(self.wavelengths, line_grid[i, j, k, l], config.line_wavs[i], fwhm, fwhm_unit='kms')

        return comb_grid, line_grid

    def spectrum(self, sfh_ceh, t_bc, logU):
        """ Obtain a 1D spectrum for a given star-formation and
        chemical enrichment history, ionization parameter and t_bc.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        logU : float
            Log10 of the ionization parameter.

        t_bc : float
            The maximum age at which to include nebular emission.
        """
        
        self.logger.debug("Interpolating nebular continuum grid")
        return self._interpolate_grid(self.combined_grid, sfh_ceh, t_bc, logU)

    def line_fluxes(self, sfh_ceh, t_bc, logU):
        """ Obtain line fluxes for a given star-formation and
        chemical enrichment history, ionization parameter and t_bc.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        logU : float
            Log10 of the ionization parameter.

        t_bc : float
            The maximum age at which to include nebular emission.
        """
        self.logger.debug("Interpolating nebular emission line grid")
        return self._interpolate_grid(self.line_grid, sfh_ceh, t_bc, logU)

    def _interpolate_grid(self, grid, sfh_ceh, t_bc, logU):
        """ Interpolates a chosen grid in logU and collapses over star-
        formation and chemical enrichment history to get 1D models. """
        

        t_bc *= 10**9

        if logU == self.logU[0]:
            logU += 10**-10

        spectrum_low_logU = np.zeros_like(grid[:, 0, 0, 0])
        spectrum_high_logU = np.zeros_like(grid[:, 0, 0, 0])

        logU_ind = self.logU[self.logU < logU].shape[0]
        logU_weight = ((self.logU[logU_ind] - logU)
                       / (self.logU[logU_ind] - self.logU[logU_ind-1]))

        index = config.age_bins[config.age_bins < t_bc].shape[0]
        weight = 1 - (config.age_bins[index] - t_bc)/config.age_widths[index-1]

        for i in range(self.metallicities.shape[0]):
            if sfh_ceh[i, :index].sum() > 0.:
                sfh_ceh[:, index-1] *= weight

                spectrum_low_logU += np.sum(grid[:, i, logU_ind-1, :index]
                                            * sfh_ceh[i, :index], axis=1)

                spectrum_high_logU += np.sum(grid[:, i, logU_ind, :index]
                                             * sfh_ceh[i, :index], axis=1)

                sfh_ceh[:, index-1] /= weight

        spectrum = (spectrum_high_logU*(1 - logU_weight)
                    + spectrum_low_logU*logU_weight)

        return spectrum

    def __bool__(self):
        return self.flag