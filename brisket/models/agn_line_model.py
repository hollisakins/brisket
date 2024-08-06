
import numpy as np

from .. import config
from .. import utils


class agn_lines(object):
    """ Allows access to and maniuplation of AGN emission line models.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values desired for the stellar models.
    """

    def __init__(self, wavelengths, model_comp, logger=utils.NullLogger):
        self.wavelengths = wavelengths
        self.type = model_comp['nebular']['type']

        if self.type == 'cloudy':
            self.combined_grid, self.line_grid = self._setup_grids(model_comp['fwhm'])
        elif self.type == 'sdss':
            self.eline_wav = config.sdss_file[:,0]
            self.eline_ratios = config.sdss_file[:,1]
            self.eline_fwhm = config.sdss_file[:,2]
        elif self.type == 'qsogen':
            if not 'scale_nlr' in model_comp.keys():
                model_comp['nebular']['scale_nlr'] = 0
            if not 'scale_halpha' in model_comp.keys():
                model_comp['nebular']['scale_halpha'] = 0
            if not 'scale_lya' in model_comp.keys():
                model_comp['nebular']['scale_lya'] = 0
        else:
            print(self.type)
            raise Exception('nebular emission "type" must be one of "cloudy" or "sdss"')


    def _gauss(self, x, L, mu, fwhm, fwhm_unit='A'):
        if fwhm_unit=='A':
            sigma = fwhm/2.355
            y = np.exp(-(x-mu)**2/(2*sigma**2))
            y *= L/np.sum(y)
        if fwhm_unit=='kms':
            sigma = mu * fwhm/2.998e5 / 2.355
            y = np.exp(-(x-mu)**2/(2*sigma**2))
            y *= L/np.sum(y)
        return y

    @property
    def line_names(self):
        if self.type == 'cloudy':
            return config.line_names
        elif self.type == 'sdss':
            return config.sdss_eline_names
    @property
    def line_wavs(self):
        if self.type == 'cloudy':
            return config.line_wavs
        elif self.type == 'sdss':
            return config.sdss_file[:,0]

    def _setup_grids(self, fwhm):
        """ Loads Cloudy nebular continuum grid and resamples to the
        input wavelengths. Loads nebular line grids and adds line fluxes
        to the correct pixels in order to create a combined grid. """

        comb_grid = np.zeros((self.wavelengths.shape[0],
                              config.logU.shape[0],
                              config.lognH.shape[0]))

        line_grid = np.zeros((config.line_wavs.shape[0],
                              config.logU.shape[0],
                              config.lognH.shape[0]))

        for i in range(config.logU.shape[0]):

            hdu_index = i+1

            raw_cont_grid = config.cont_grid[hdu_index].data
            raw_line_grid = config.line_grid[hdu_index].data

            line_grid[:, i, :] = raw_line_grid[1:, 1:].T

            for j in range(config.lognH.shape[0]):
                comb_grid[:, i, j] = np.interp(self.wavelengths,
                                                config.neb_wavs,
                                                raw_cont_grid[j+1, 1:],
                                                left=0, right=0)

        # Add the nebular lines to the resampled nebular continuum grid.
        for i,mu in enumerate(config.line_wavs):
                for j in range(config.logU.shape[0]):
                    for k in range(config.lognH.shape[0]):
                        comb_grid[:,j,k] += self._gauss(self.wavelengths, line_grid[i,j,k], mu, fwhm, fwhm_unit='kms')

        return comb_grid, line_grid

    def spectrum(self, model_comp):
        if self.type=='cloudy' or self.type=='sdss':
            LHb = model_comp['LHb']
            LHb = np.power(10., LHb)
            LHb /= 3.828e+33*4861 # convet from erg/s to Lsun/angstrom
        if self.type=='cloudy':
            logU, lognH = model_comp['logU'], model_comp['lognH']
            LHb0 = self._interpolate_grid(self.line_grid, logU, lognH)[95]
            return LHb/LHb0*self._interpolate_grid(self.combined_grid, logU, lognH)
        elif self.type=='sdss':
            spectrum = np.zeros_like(self.wavelengths)
            for A, mu, fwhm in zip(self.eline_ratios, self.eline_wav,self.eline_fwhm):
                spectrum += self._gauss(self.wavelengths, A, mu, fwhm, fwhm_unit='A')
            spectrum *= LHb
            return spectrum
        elif self.type=='qsogen':
            varlin = model_comp['nebular']['eline_type']
            scalin = np.power(10., model_comp['nebular']['scale_eline'])
            scahal = np.power(10., model_comp['nebular']['scale_halpha'])
            scalya = np.power(10., model_comp['nebular']['scale_lya'])
            scanlr = np.power(10., model_comp['nebular']['scale_nlr'])
            scoiii = np.power(10., model_comp['nebular']['scale_oiii'])
            wnrm = config.qsogen_wnrm

            nlr = (config.qsogen_nlr + (scoiii-1) * config.qsogen_nlr_oiii) * scalin * scanlr
            blr = (config.qsogen_blr + (scalya-1) * config.qsogen_blr_lya) * scalin
            linval = blr + nlr

            conval = np.interp(self.wavelengths, config.qsogen_wavelengths, config.qsogen_conval)
            spectrum = np.interp(self.wavelengths, config.qsogen_wavelengths, linval)
            nlr = np.interp(self.wavelengths, config.qsogen_wavelengths, nlr)
            blr = np.interp(self.wavelengths, config.qsogen_wavelengths, blr)

            inrm = np.argmin(np.abs(self.wavelengths - wnrm))
            spectrum /= conval[inrm]
            nlr /= conval[inrm]
            blr /= conval[inrm]
            self.nlr = nlr
            self.blr = blr
            
            return spectrum




    def line_fluxes(self, model_comp):
        LHb = model_comp['LHb']
        LHb = np.power(10., LHb)
        LHb /= 3.828e+33*4861 # convet from erg/s to Lsun/angstrom
        if self.type=='cloudy':
            logU, lognH = model_comp['logU'], model_comp['lognH']
            LHb0 = self._interpolate_grid(self.line_grid, logU, lognH)[95]
            return LHb/LHb0*self._interpolate_grid(self.line_grid, logU, lognH)
        elif self.type=='sdss':
            return self.eline_ratios * LHb


    def _interpolate_grid(self, grid, logU, lognH):
        """ Interpolates a chosen grid in logU and collapses over star-
        formation and chemical enrichment history to get 1D models. """

        if logU == config.logU[0]:
            logU += 10**-10

        logU_ind = config.logU[config.logU < logU].shape[0]
        logU_weight = ((config.logU[logU_ind] - logU)
                    / (config.logU[logU_ind] - config.logU[logU_ind-1]))

        lognH_ind = config.lognH[config.lognH < lognH].shape[0]
        lognH_weight = ((config.lognH[lognH_ind] - lognH)
                    / (config.lognH[lognH_ind] - config.logU[lognH_ind-1]))

        spectrum_low = grid[:, logU_ind-1, lognH_ind-1]
        spectrum_high = grid[:, logU_ind, lognH_ind]

        low_weight = (lognH_weight+logU_weight)/2
        high_weight = 1-low_weight

        spectrum = spectrum_high * high_weight + spectrum_low * low_weight

        return spectrum
