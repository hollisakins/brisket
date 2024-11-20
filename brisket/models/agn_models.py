import astropy.units as u
from astropy.constants import c
c = c.to(u.angstrom*u.Hz).value
import numpy as np
from brisket import config
from brisket import utils
from brisket.utils.sed import SED
from brisket.models.base_models import *
from brisket.console import log

class PowerlawAccrectionDiskModel(BaseFunctionalModel, BaseSourceModel):
    type = 'source'
    order = 1
    
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        if not 'beta' in params:
            params['beta'] = -2.0
        if not 'Muv' in params:
            raise Exception("Key Muv must be specified in parameters")

    def emit(self, params):
        beta, Muv, redshift = float(params['beta']), float(params['Muv']), float(params['redshift']) # absolute magnitude
        sed = SED(wav_rest=self.wavelengths, flam=np.power(self.wavelengths, beta), redshift=redshift, verbose=False)
        sed *= np.power(10., -0.4*(Muv-sed.Muv))

        # fnu_uv = np.power(10., -(Muv+48.60)/2.5) * u.erg/u.s/u.Hz/u.cm**2
        # Snu_uv = (fnu_uv * 4*np.pi*(10*u.pc)**2).to(u.erg/u.s/u.Hz)
        # Slam_uv = (Snu_uv * c/(1500*u.angstrom)**2).to(u.Lsun/u.angstrom).value # in Lsun/angstrom
        # tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
        # y0 = np.trapz(y*tophat, x=self.wavelengths)
        # y *= Slam_uv/y0
        return sed


# class DoublePowerLawAccretionDiskModel(BaseAGNModel):
#     def __init__(self, wavelengths, model_comp):
#         self.wavelengths = wavelengths
#         self.logger = logger
#     def spectrum(self, model_comp):
#         """Set multi-powerlaw continuum in flux density per unit frequency."""
#         # Flip signs of powerlaw slopes to enable calculation to be performed
#         # as a function of wavelength rather than frequency
#         sl1 = model_comp['beta1']
#         sl2 = model_comp['beta2']
#         wavbrk1 = model_comp['wav1']
#         flxnrm=1.0
#         wavnrm=5500
#         # Define normalisation constant to ensure continuity at wavbrk
#         const2 = flxnrm/(wavnrm**sl2)
#         const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

#         # Define basic continuum using the specified normalisation fnorm at
#         # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)
#         fluxtemp = np.where(self.wavelengths < wavbrk1,
#                             const1*self.wavelengths**sl1,
#                             const2*self.wavelengths**sl2)

#         # Also add steeper power-law component for sub-Lyman-alpha wavelengths
#         sl3 = sl1 + 1.0
#         wavbrk3 = 1200
#         # Define normalisation constant to ensure continuity
#         const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)

#         y = np.where(self.wavelengths < wavbrk3,
#             const3*self.wavelengths**sl3,
#             fluxtemp)


#         Muv = model_comp['Muv'] # absolute magnitude
#         fnu_uv = np.power(10., -(Muv+48.60)/2.5)
#         Snu_uv = (fnu_uv * 4*np.pi*(3.086e19)**2) # in erg/s/Hz
#         Slam_uv = (Snu_uv * c/(1500)**2)/3.846e33 # in Lsun/A
#         tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
#         y0 = np.trapz(y*tophat, x=self.wavelengths)
#         y *= Slam_uv/y0

#         y[self.wavelengths < 912] = 0

#         return y

# class SDSSModel(BaseAGNModel):
#     def __init__(self, wavelengths, model_comp):
#         self.wavelengths = wavelengths
#         self.logger = logger
#     def spectrum(self, model_comp):
#         sdss_wavs = config.sdss_composite['WAVELENGTH'] # A
#         sdss_fnus = config.sdss_composite['FLUX'] # erg/s/cm^2/A.

#         y = np.interp(self.wavelengths, sdss_wavs, sdss_fnus)
#         tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
#         y0 = np.trapz(y*tophat, x=self.wavelengths)

#         Muv = model_comp['Muv'] # absolute magnitude
#         fnu_uv = np.power(10., -(Muv+48.60)/2.5) * u.erg/u.s/u.Hz/u.cm**2
#         Snu_uv = (fnu_uv * 4*np.pi*(10*u.pc)**2).to(u.erg/u.s/u.Hz)
#         Slam_uv = (Snu_uv * c/(1500*u.angstrom)**2).to(u.Lsun/u.angstrom).value # in Lsun/angstrom
#         y *= Slam_uv/y0
#         return y


# class AGNSlimAccretionDiskModel(BaseGriddedModel, BaseSourceModel):
#     def _resample(self, wavelengths):
#         pass
#     def emit(self, model_comp):
#         # call some AGNslim grids? 
#         return y