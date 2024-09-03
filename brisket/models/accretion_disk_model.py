import astropy.units as u
from astropy.constants import c
c = c.to(u.angstrom*u.Hz).value
import numpy as np
from brisket import config
from brisket import utils

class AccretionDiskModel(object):
    """ Allows access to and maniuplation of AGN accretion disk models.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the stellar models.
    
    model_comp : dict
        Subset of the model_components dictionary with AGN accretion disk parameters
    """

    def __init__(self, wavelengths, model_comp, logger=utils.NullLogger):
        self.wavelengths = wavelengths
        self.type = model_comp['type']
        # could be optimized more

    def spectrum(self, model_comp):
        if self.type=='plaw':
            beta = model_comp['beta']
            Muv = model_comp['Muv'] # absolute magnitude
            fnu_uv = np.power(10., -(Muv+48.60)/2.5) * u.erg/u.s/u.Hz/u.cm**2
            Snu_uv = (fnu_uv * 4*np.pi*(10*u.pc)**2).to(u.erg/u.s/u.Hz)
            Slam_uv = (Snu_uv * c/(1500*u.angstrom)**2).to(u.Lsun/u.angstrom).value # in Lsun/angstrom
            y = np.power(self.wavelengths, beta)
            tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
            y0 = np.trapz(y*tophat, x=self.wavelengths)
            y *= Slam_uv/y0
            # y[self.wavelengths < 1216] *= np.exp(0.05*(self.wavelengths[self.wavelengths < 1216]-1216))
            # y[self.wavelengths < 100] = 0
            return y

        if self.type=='dblplw':
            """Set multi-powerlaw continuum in flux density per unit frequency."""
            # Flip signs of powerlaw slopes to enable calculation to be performed
            # as a function of wavelength rather than frequency
            sl1 = model_comp['beta1']
            sl2 = model_comp['beta2']
            wavbrk1 = model_comp['wav1']
            flxnrm=1.0
            wavnrm=5500
            # Define normalisation constant to ensure continuity at wavbrk
            const2 = flxnrm/(wavnrm**sl2)
            const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

            # Define basic continuum using the specified normalisation fnorm at
            # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)
            fluxtemp = np.where(self.wavelengths < wavbrk1,
                                const1*self.wavelengths**sl1,
                                const2*self.wavelengths**sl2)

            # Also add steeper power-law component for sub-Lyman-alpha wavelengths
            sl3 = sl1 + 1.0
            wavbrk3 = 1200
            # Define normalisation constant to ensure continuity
            const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)

            y = np.where(self.wavelengths < wavbrk3,
                const3*self.wavelengths**sl3,
                fluxtemp)


            Muv = model_comp['Muv'] # absolute magnitude
            fnu_uv = np.power(10., -(Muv+48.60)/2.5)
            Snu_uv = (fnu_uv * 4*np.pi*(3.086e19)**2) # in erg/s/Hz
            Slam_uv = (Snu_uv * c/(1500)**2)/3.846e33 # in Lsun/A
            tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
            y0 = np.trapz(y*tophat, x=self.wavelengths)
            y *= Slam_uv/y0

            y[self.wavelengths < 912] = 0

            return y

        elif self.type=='sdss':
            sdss_wavs = config.sdss_composite['WAVELENGTH'] # A
            sdss_fnus = config.sdss_composite['FLUX'] # erg/s/cm^2/A.

            y = np.interp(self.wavelengths, sdss_wavs, sdss_fnus)
            tophat = np.where((self.wavelengths > 1450)&(self.wavelengths < 1550), 1, 0)
            y0 = np.trapz(y*tophat, x=self.wavelengths)

            Muv = model_comp['Muv'] # absolute magnitude
            fnu_uv = np.power(10., -(Muv+48.60)/2.5) * u.erg/u.s/u.Hz/u.cm**2
            Snu_uv = (fnu_uv * 4*np.pi*(10*u.pc)**2).to(u.erg/u.s/u.Hz)
            Slam_uv = (Snu_uv * c/(1500*u.angstrom)**2).to(u.Lsun/u.angstrom).value # in Lsun/angstrom
            y *= Slam_uv/y0
            return y
