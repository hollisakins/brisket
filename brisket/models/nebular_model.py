from __future__ import print_function, division, absolute_import

import astropy.units as u
import numpy as np
from copy import copy
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
        if not 'nebular' in params:
            self.logger.info("Skipping nebular emission module")
            self.flag = False; return
        
        self.type = params['nebular']['type']
        if self.type == 'flex':
            self.logger.info(f"Initializing nebular emission module".ljust(50) + f'({self.type})'.rjust(20))

        elif self.type == 'cloudy':
            self.model = params['stellar_model']
            self.logger.info(f"Initializing nebular emission module".ljust(50) + f'({self.type}, {self.model})'.rjust(20))

        #### Prepare CLOUDY modeling 
        if self.type == 'cloudy':
            if 'metallicity' in list(params['nebular']):
                self.logger.info("Computing independent chemical enrichment history for nebular models")
                self.neb_sfh = StarFormationHistoryModel(params)
            self.metallicities = config.stellar_models[self.model]['metallicities']
            self.logU = config.nebular_models[self.model]['logU']
            self.neb_ages = config.nebular_models[self.model]['neb_ages']
            self.neb_wavs = config.nebular_models[self.model]['neb_wavs']
            self.cont_grid = config.nebular_models[self.model]['cont_grid']
            self.line_grid = config.nebular_models[self.model]['line_grid']

            self.continuum_grid, self.line_grid, self.combined_grid = self._setup_cloudy_grids()

        #### Prepare CLOUDY modeling 
        if self.type == 'flex':
            from brisket.models.grids.linelist import linelist
            self.linelist = linelist


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


    def _setup_cloudy_grids(self):
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


        cont_grid = copy(comb_grid)
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

        return cont_grid, line_grid, comb_grid

    def spectrum(self, params, sfh_ceh=None):
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
        if self.type == 'cloudy':
            if 'grid' in params['nebular']:
                if params['nebular']['grid'] == 'continuum':
                    self.logger.debug("Interpolating nebular continuum grid")
                    return self._interpolate_cloudy_grid(self.continuum_grid, sfh_ceh, params['t_bc'], params['nebular']['logU'])
                if params['nebular']['grid'] == 'line':
                    self.logger.debug("Interpolating nebular line grid")
                    return self._interpolate_cloudy_grid(self.line_grid, sfh_ceh, params['t_bc'], params['nebular']['logU'])
            else:
                self.logger.debug("Interpolating combined nebular grid")
                return self._interpolate_cloudy_grid(self.combined_grid, sfh_ceh, params['t_bc'], params['nebular']['logU'])
        elif self.type == 'flex':
            self.logger.debug("Compute flexible nebular spectrum")
            return self._flex_spectrum(params) 

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

        if self.type == 'cloudy':
            self.logger.debug("Interpolating nebular emission line grid")
            return self._interpolate_cloudy_grid(self.line_grid, sfh_ceh, t_bc, logU)
        #


    def _interpolate_cloudy_grid(self, grid, sfh_ceh, t_bc, logU):
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

    def _flex_spectrum(self, params):
        redshift = params['redshift']
        lum_flux = 1
        if redshift > 0.:
            dL = config.cosmo.luminosity_distance(redshift).to(u.cm).value
            lum_flux = 4*np.pi*dL**2

        flex_spectrum = np.zeros_like(self.wavelengths)

        # Handle different choices for simple continuum models
        if 'cont_type' in params['nebular']: 
            if params['nebular']['cont_type'] == 'flat':
                flex_spectrum += 1
            elif params['nebular']['cont_type'] == 'plaw':
                if 'cont_alpha' in params['nebular']:
                    flex_spectrum += np.power(self.wavelengths, params['nebular']['cont_alpha']-2)
                elif 'cont_beta' in params['nebular']:
                    flex_spectrum += np.power(self.wavelengths, params['nebular']['cont_beta'])
            elif params['nebular']['cont_type'] == 'dblplaw':
                assert 'cont_break' in params['nebular']
                wavbrk = params['nebular']['cont_break']
                if 'cont_alpha1' in params['nebular']:
                    sl1, sl2 = params['nebular']['cont_alpha1']-2, params['nebular']['cont_alpha2']-2
                elif 'cont_beta1' in params['nebular']:
                    sl1, sl2 = params['nebular']['cont_beta1'], params['nebular']['cont_beta2']
                norm = (wavbrk**sl1)/(wavbrk**sl2)
                flex_spectrum += np.where(self.wavelengths < wavbrk,
                                          self.wavelengths**sl1,
                                          norm*self.wavelengths**sl2)
            elif params['nebular']['cont_type'] == 'multiplaw':
                assert 'cont_breaks' in params['nebular']
                mask0 = np.array(np.zeros(len(flex_spectrum)),dtype=bool)
                breaks = np.append(np.min(self.wavelengths),params['nebular']['cont_breaks'])
                breaks = np.append(breaks, np.max(self.wavelengths))
                if 'cont_alpha1' in params['nebular']:
                    slopes = [params['nebular'][f'cont_alpha{i}']-2 for i in range(1,len(params['nebular']['cont_breaks'])+2)]
                elif 'cont_beta1' in params['nebular']:
                    slopes = [params['nebular'][f'cont_beta{i}'] for i in range(1,len(params['nebular']['cont_breaks'])+2)]
                for i in range(len(slopes)-1):
                    mask1 = (self.wavelengths > breaks[i])&(self.wavelengths < breaks[i+1])
                    mask2 = (self.wavelengths > breaks[i+1])&(self.wavelengths < breaks[i+2])
                    if i==0:
                        flex_spectrum[mask1] = self.wavelengths[mask1]**slopes[i]
                    flex_spectrum[mask2] = self.wavelengths[mask2]**slopes[i+1]
                    flex_spectrum[mask1|mask0] /= flex_spectrum[mask1][-1]
                    flex_spectrum[mask2] /= flex_spectrum[mask2][0]
                    mask0 = mask0|mask1

            if 'f5100' in params['nebular']:
                flex_spectrum /= flex_spectrum[np.argmin(np.abs(self.wavelengths - 5100.))]
                flex_spectrum *= params['nebular']['f5100'] * lum_flux * (1+redshift) / 3.826e33 # convert to Lsun/angstrom
            else:
                raise Exception('Need normalization')

        self.continuum = copy(flex_spectrum)

        line_names, line_wavs, line_fwhms, line_fluxes = [], [], [], []
        for key in params['nebular']:
            if not key.startswith('f_'): continue;
            key_split = key.split('_')
            if len(key_split)==2: # no suffix to the line
                name = key_split[1]
                if name in self.linelist['name']:
                    if 'fwhm' in params['nebular']:
                        fwhm = params['nebular']['fwhm']
                    elif f'fwhm_{name}' in params['nebular']:
                        fwhm = params['nebular'][f'fwhm_{name}']
                    else:
                        fwhm = fwhm_default
                    if f'dv_{name}' in params['nebular']:
                        dv = params['nebular'][f'dv_{name}']
                    else:
                        dv = 0
                else:
                    print(f'Skipping key {key}, {name} not in line list')

            elif len(key_split)==3:
                name = key_split[1]
                suffix = key_split[2]
                if name in self.linelist['name']:
                    if f'fwhm_{suffix}' in params['nebular']:
                        fwhm = params['nebular'][f'fwhm_{suffix}']
                    elif f'fwhm_{name}_{suffix}' in params['nebular']:
                        fwhm = params['nebular'][f'fwhm_{name}_{suffix}']
                    else:
                        fwhm = fwhm_default
                    if f'dv_{suffix}' in params['nebular']:
                        dv = params['nebular'][f'dv_{suffix}']
                    elif f'dv_{name}' in params['nebular']:
                        dv = params['nebular'][f'dv_{name}']
                    elif f'dv_{name}_{suffix}' in params['nebular']:
                        dv = params['nebular'][f'dv_{name}_{suffix}']
                    else:
                        dv = 0
                else:
                    print(f'Skipping key {key}, {name} not in line list')
            else:
                raise Exception
            
            wav = self.linelist['wav'][self.linelist['name']==name][0]
            wav *= 1+dv/2.998e5

            flux = params['nebular'][key] # in erg/s/cm2

            lum = flux * lum_flux * (1+redshift) / 3.826e33
            g = self._gauss(self.wavelengths, lum, wav, fwhm, fwhm_unit='kms')
            flex_spectrum += g

        return flex_spectrum


    def __bool__(self):
        return self.flag