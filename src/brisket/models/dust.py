


import numpy as np
# from scipy.interpolate import CubicSpline

from ..parameters import Params
from ..utils.console import setup_logger
from .base import *

class BaseDustAttenuationModel(AbsorberModel):

    name = "dust_atten"

    def __init__(self, verbose=False, **kwargs):

        self.verbose = verbose
        self.logger = setup_logger(__name__, self.verbose)
        self.params = self.validate(kwargs)
    
    def validate(self, kwargs):
        """
        Validate/parse parameters for the stellar model.
        """
        params = super().validate(kwargs)

        if 'Av' not in kwargs:
            alt_keys = ['AV', 'A_V', 'A_v']
            if any(key in kwargs for key in alt_mass_keys):
                k = next(key for key in alt_mass_keys if key in kwargs)
                raise exceptions.MisspelledParameter(f"Parameter '{k}' not understood. Did you mean 'Av'?")
            else:
                raise exceptions.MissingParameter("Parameter 'Av' not specified, cannot create dust attenuation model.")
        params['Av'] = kwargs['Av']
        del kwargs['Av']

        return params

    def _assign_wavelengths(self, wavelengths):
        self.wavelengths = wavelengths
    
    def prepare(self, wavelengths):
        """
        Prototype for child defined "prepare" methods.
        In general, should call _assign_wavelengths. 
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent. "
            "How did you get here!?"
        )

    def get_transmission(self, params):
        """
        Prototype for child defined get_transmission methods.
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent. "
            "How did you get here!?"
        )

    
    # def transmission(self):
    #     Av = self.param['dust_atten']['Av']
    #     logfscat = self.param['dust_atten']['logfscat']
    #     fscat = np.power(10., logfscat)

    #     trans_cont = 10**(-Av*self.A_cont/2.5) + fscat
        
    #     bc_Av = Av * self.param['dust_atten']['eta']
    #     trans_bc = 10**(-bc_Av*self.A_cont/2.5) + fscat

    #     trans_line = 10**(-bc_Av*self.A_line/2.5) + fscat
    #     return trans_cont, trans_line, trans_bc


class CalzettiDustAttenuationModel(BaseDustAttenuationModel):
    def __init__(self, wavelengths, params):
        super().__init__(wavelengths, params)
        self.A_cont = self._calzetti(wavelengths)
        self.A_line = self._calzetti(config.line_wavs)


    def _calzetti(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the Calzetti et al.
        (2000) attenuation curve. """

        A_lambda = np.zeros_like(wavs)

        wavs_mic = wavs*10**-4

        mask1 = (wavs < 1200.)
        mask2 = (wavs < 6300.) & (wavs >= 1200.)
        mask3 = (wavs < 31000.) & (wavs >= 6300.)

        A_lambda[mask1] = ((wavs_mic[mask1]/0.12)**-0.77
                           * (4.05 + 2.695*(- 2.156 + 1.509/0.12
                                            - 0.198/0.12**2 + 0.011/0.12**3)))

        A_lambda[mask2] = (4.05 + 2.695*(- 2.156
                                         + 1.509/wavs_mic[mask2]
                                         - 0.198/wavs_mic[mask2]**2
                                         + 0.011/wavs_mic[mask2]**3))

        A_lambda[mask3] = 2.659*(-1.857 + 1.040/wavs_mic[mask3]) + 4.05

        A_lambda /= 4.05

        return A_lambda

    def _Alam(self, params):
        return self.A_cont, self.A_line

class CardelliDustAttenuationModel(BaseDustAttenuationModel):
    def __init__(self, wavelengths, params):
        super().__init__(wavelengths, params)
        self.A_cont = self._cardelli(wavelengths)
        self.A_line = self._cardelli(config.line_wavs)

    def _cardelli(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the Cardelli et al.
        (1989) extinction curve. """

        A_lambda = np.zeros_like(wavs)

        inv_mic = 1./(wavs*10.**-4.)

        mask1 = (inv_mic < 1.1)
        mask2 = (inv_mic >= 1.1) & (inv_mic < 3.3)
        mask3 = (inv_mic >= 3.3) & (inv_mic < 5.9)
        mask4 = (inv_mic >= 5.9) & (inv_mic < 8.)
        mask5 = (inv_mic >= 8.)

        A_lambda[mask1] = (0.574*inv_mic[mask1]**1.61
                           + (-0.527*inv_mic[mask1]**1.61)/3.1)

        y = inv_mic[mask2] - 1.82

        A_lambda[mask2] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
                           + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6
                           + 0.32999*y**7 + (1.41388*y + 2.28305*y**2
                                             + 1.07233*y**3 - 5.38434*y**4
                                             - 0.62251*y**5 + 5.30260*y**6
                                             - 2.09002*y**7)/3.1)

        A_lambda[mask3] = ((1.752 - 0.316*inv_mic[mask3]
                            - (0.104)/((inv_mic[mask3]-4.67)**2 + 0.341))
                           + (-3.09 + 1.825*inv_mic[mask3]
                              + 1.206/((inv_mic[mask3]-4.62)**2 + 0.263))/3.1)

        A_lambda[mask4] = ((1.752 - 0.316*inv_mic[mask4]
                            - (0.104)/((inv_mic[mask4]-4.67)**2 + 0.341)
                            - 0.04473*(inv_mic[mask4] - 5.9)**2
                            - 0.009779*(inv_mic[mask4]-5.9)**3)
                           + (-3.09 + 1.825*inv_mic[mask4]
                              + 1.206/((inv_mic[mask4]-4.62)**2 + 0.263)
                              + 0.2130*(inv_mic[mask4]-5.9)**2
                              + 0.1207*(inv_mic[mask4]-5.9)**3)/3.1)

        A_lambda[mask5] = ((-1.073
                            - 0.628*(inv_mic[mask5] - 8.)
                            + 0.137*(inv_mic[mask5] - 8.)**2
                            - 0.070*(inv_mic[mask5] - 8.)**3)
                           + (13.670
                              + 4.257*(inv_mic[mask5] - 8.)
                              - 0.420*(inv_mic[mask5] - 8.)**2
                              + 0.374*(inv_mic[mask5] - 8.)**3)/3.1)

        return A_lambda

    def _Alam(self, params):
        return self.A_cont, self.A_line

class SMCDustAttenuationModel(BaseDustAttenuationModel):
    def __init__(self, wavelengths, params):
        super().__init__(wavelengths, params)
        self.A_cont = self._smc_gordon(wavelengths)
        self.A_line = self._smc_gordon(config.line_wavs)
    

    def _smc_gordon(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the Gordon et al.
        (2003) Small Magellanic Cloud extinction curve. Warning: this
        currently diverges at small wavelengths, probably some sort of
        power law interpolation at the blue end should be added. """

        A_lambda = np.zeros_like(wavs)

        inv_mic = 1./(wavs*10.**-4.)

        c1 = -4.959
        c2 = 2.264
        c3 = 0.389
        c4 = 0.461
        x0 = 4.6
        gamma = 1.0
        Rv = 2.74

        D = inv_mic**2/((inv_mic**2 - x0**2)**2 + inv_mic**2*gamma**2)
        F = 0.5392*(inv_mic - 5.9)**2 + 0.05644*(inv_mic - 5.9)**3
        F[inv_mic < 5.9] = 0.
        # values at 2.198 and 1.25 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104)

        A_lambda = (c1 + c2*inv_mic + c3*D + c4*F)/Rv + 1.

        # Generate region redder than 2760AA by interpolation
        ref_wavs = np.array([0.276, 0.296, 0.37, 0.44, 0.55,
                             0.65, 0.81, 1.25, 1.65, 2.198, 3.1])*10**4

        ref_ext = np.array([2.220, 2.000, 1.672, 1.374, 1.00,
                            0.801, 0.567, 0.25, 0.169, 0.11, 0.])

        if np.max(wavs) > 2760.:
            A_lambda[wavs > 2760.] = np.interp(wavs[wavs > 2760.],
                                               ref_wavs, ref_ext, right=0.)

        """
        import matplotlib.pyplot as plt
        print(np.c_[wavs, inv_mic])
        plt.figure()
        plt.plot(wavs, c3*D/Rv, color="blue")
        plt.plot(wavs, c4*F/Rv, color="red")
        plt.plot(wavs, c1/Rv + np.zeros_like(wavs), color="yellow")
        plt.plot(wavs, c2*inv_mic/Rv, color="green")
        plt.plot(wavs, A_lambda, color="black")
        plt.show()
        """
        return A_lambda

    def _Alam(self, params):
        return self.A_cont, self.A_line


class SalimDustAttenuationModel(BaseDustAttenuationModel):
    def __init__(self, wavelengths, params):
        super().__init__(wavelengths, params)
        calz = CalzettiDustAttenuationModel(wavelengths, params)
        self.A_cont_calz = calz._calzetti(wavelengths)
        self.A_line_calz = calz._calzetti(config.line_wavs)

    def _Alam(self, params):
        delta = param["delta"]
        B = param["B"]
        Rv_m = 4.05/((4.05+1)*(4400./5500.)**delta - 4.05)

        drude = B*self.wavelengths**2*350.**2
        drude /= (self.wavelengths**2 - 2175.**2)**2 + self.wavelengths**2*375.**2
        A_cont = self.A_cont_calz*Rv_m*(self.wavelengths/5500.)**delta + drude
        A_cont /= Rv_m

        drude = B*config.line_wavs**2*350.**2
        drude /= (config.line_wavs**2 - 2175.**2)**2 + config.line_wavs**2*375.**2
        A_line = self.A_line_calz*Rv_m*(config.line_wavs/5500.)**delta + drude
        A_line /= Rv_m

        return A_cont, A_line

class CF00DustAttenuationModel(BaseDustAttenuationModel):
    def __init__(self, wavelengths, params):
        super().__init__(wavelengths, params)

    def _Alam(self, params):
        """ Modified Charlot + Fall (2000) model of Carnall et al.
        (2018) and Carnall et al. (2019b). """
        A_cont = (5500./self.wavelengths)**params["n"]
        A_line = (5500./config.line_wavs)**params["n"]
        return A_cont, A_line
