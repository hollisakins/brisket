import numpy as np

import spectres
from brisket import utils
from brisket import config
from numpy.polynomial.chebyshev import chebval, chebfit

# should add new parameters called spectrum_corr and sed_corr? or something? 

class SpectralCalibrationModel(object):
    """ A class for modelling spectrophotometric calibration.
    Applied correction factors to forward-model the internal 
    model SED to match the observed spectrum. Note that this 
    is the reverse form the implementation in BAGPIPES, which
    prefers to adjust the input spectrum. 

    Parameters
    ----------

    params : dictionary
        Model parameters dictionary. Calibration model will be 
        initialized if 'calib' keyword present in 'params'
    """

    def __init__(self, spec_wavs, params, logger=utils.NullLogger):
        self.spec_wavs = spec_wavs
        # self.param = calib_dict
        # self.y = spectrum[:, 1]
        # self.y_err = spectrum[:, 2]
        # self.y_model = spectral_model[:, 1]
        # self.wavs = spectrum[:, 0]

        # # Transform the spectral wavelengths to the interval (-1, 1).
        # x = spectrum[:, 0]
        # self.x = 2.*(x - (x[0] + (x[-1] - x[0])/2.))/(x[-1] - x[0])

        # # Call the appropriate method to calculate the calibration.
        # getattr(self, self.param["type"])()

        self.flag = True
        if not 'calib' in params:
            self.logger.info("Skipping spectral calibration module")
            self.flag = False; return

        self.R_curve = None
        if 'R_curve' in params['calib']:
            if isinstance(params['calib']['R_curve'], str):
                self.R_curve = config.R_curves[params['calib']['R_curve']]
            else:
                self.R_curve = params['calib']['R_curve'] # 2D array, col 1 is wavelength in angstroms, col 2 is resolution
            
            self.oversample = params['calib']['oversample']

            spec_wavs_R = [0.95*self.spec_wavs[0]]
            while spec_wavs_R[-1] < 1.05*self.spec_wavs[-1]:
                R_val = np.interp(spec_wavs_R[-1], self.R_curve[:, 0], self.R_curve[:, 1])
                dwav = spec_wavs_R[-1]/R_val/self.oversample
                spec_wavs_R.append(spec_wavs_R[-1] + dwav)

            self.spec_wavs_R = np.array(spec_wavs_R)
    
    def convolve_R_curve(self, wav_obs, spectrum, f_LSF):
        spectrum = spectres.spectres(self.spec_wavs_R, wav_obs, spectrum, fill=0, verbose=False)
        sigma_pix = self.oversample/2.35/f_LSF  # sigma width of kernel in pixels
        k_size = 4*int(sigma_pix+1)
        x_kernel_pix = np.arange(-k_size, k_size+1)
        kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
        kernel /= np.trapz(kernel)  # Explicitly normalise kernel
        spectrum = np.convolve(spectrum, kernel, mode="valid")
        wav_obs = self.spec_wavs_R[k_size:-k_size]
        return wav_obs, spectrum


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

    def __bool__(self):
        return self.flag