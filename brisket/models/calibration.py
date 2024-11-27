import numpy as np

import spectres
from brisket import utils
from brisket import config
from brisket.models.base_models import *
from numpy.polynomial.chebyshev import chebval, chebfit

# should add new parameters called spectrum_corr and sed_corr? or something? 

class SpectralCalibrationModel(BaseModel):
    """
    A class for modelling spectrophotometric calibration.
    Applies correction factors to forward-model the internal 
    model SED to produce a mock spectrum, accounting for 
    offsets/resolution effects. Note that this reversed from
    the implementation in BAGPIPES, which prefers to correct 
    the input spectrum. 
    """
    type = 'calibration'
    order = 1000

    def __init__(self, params):
        self._build_defaults(params)

        self.R_curve = None
        if self.apply_R_curve:
            if isinstance(params['R_curve'].value, str):
                self.R_curve = config.R_curves[str(params['R_curve'])]
            else:
                self.R_curve = np.array(params['R_curve']) # 2D array, col 1 is wavelength in angstroms, col 2 is resolution
            
            self.oversample = int(params['oversample'])

    def _resample(self, wavelengths):
        pass
    
    def _validate_components(self, params):
        pass



    def _build_defaults(self, params):
        self.apply_R_curve = False
        self.apply_poly = False

        if 'R_curve' in params:
            self.apply_R_curve = True
            if 'f_LSF' not in params: 
                params['f_LSF'] = 1
            if 'oversample' not in params:
                params['oversample'] = 4
        
        # if 'poly' in params:
            # self.apply_poly = True


    def apply(self, params, spec_wavs, sed):

        if self.apply_R_curve:
            f_LSF = float(params['f_LSF'])
            z = float(params['redshift'])

            spec_wavs_R = [0.95*spec_wavs[0]]
            while spec_wavs_R[-1] < 1.05*spec_wavs[-1]:
                R_val = np.interp(spec_wavs_R[-1], self.R_curve[:, 0], self.R_curve[:, 1])
                dwav = spec_wavs_R[-1]/R_val/self.oversample
                spec_wavs_R.append(spec_wavs_R[-1] + dwav)

            self.spec_wavs_R = np.array(spec_wavs_R)
            sed.resample(self.spec_wavs_R/(1+z))

            sigma_pix = self.oversample/2.35/f_LSF  # sigma width of kernel in pixels
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)
            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(sed, kernel, mode="valid")
            spectrum.wav_rest = spectrum.wav_rest[k_size:-k_size]
            # spectrum.resample(spec_wavs)

            return spectrum

        if self.apply_poly:
            pass

    def polynomial_bayesian(self):
        """ Bayesian fitting of Chebyshev calibration polynomial. """

        coefs = []
        while str(len(coefs)) in list(self.param):
            coefs.append(self.param[str(len(coefs))])

        self.poly_coefs = np.array(coefs)
        self.model = chebval(self.x, coefs)

    def double_polynomial_bayesian(self):
        """ Bayesian fitting of Chebyshev calibration polynomial. """

        x_blue = self.wavs[self.wavs < self.param["wav_cut"]]
        x_red = self.wavs[self.wavs > self.param["wav_cut"]]

        self.x_blue = 2.*(x_blue - (x_blue[0] + (x_blue[-1] - x_blue[0])/2.))
        self.x_blue /= (x_blue[-1] - x_blue[0])

        self.x_red = 2.*(x_red - (x_red[0] + (x_red[-1] - x_red[0])/2.))
        self.x_red /= (x_red[-1] - x_red[0])

        blue_coefs = []
        red_coefs = []

        while "blue" + str(len(blue_coefs)) in list(self.param):
            blue_coefs.append(self.param["blue" + str(len(blue_coefs))])

        while "red" + str(len(red_coefs)) in list(self.param):
            red_coefs.append(self.param["red" + str(len(red_coefs))])

        self.blue_poly_coefs = np.array(blue_coefs)
        self.red_poly_coefs = np.array(red_coefs)

        model = np.zeros_like(self.x)
        model[self.wavs < self.param["wav_cut"]] = chebval(self.x_blue,
                                                           blue_coefs)

        model[self.wavs > self.param["wav_cut"]] = chebval(self.x_red,
                                                           red_coefs)

        self.model = model

    def polynomial_max_like(self):
        order = int(self.param["order"])

        mask = (self.y == 0.)

        ratio = self.y_model/self.y
        errs = np.abs(self.y_err*self.y_model/self.y**2)

        ratio[mask] = 0.
        errs[mask] = 9.9*10**99

        coefs = chebfit(self.x, ratio, order, w=1./errs)

        self.poly_coefs = np.array(coefs)
        self.model = chebval(self.x, coefs)
