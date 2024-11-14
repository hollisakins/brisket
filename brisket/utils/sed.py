import sys
import spectres
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant

# temporary
from dotmap import DotMap
import astropy.units as u
config = DotMap(default_wavelength_unit=u.angstrom, default_frequency_unit=u.GHz, default_energy_unit=u.keV)

class SED(object):
    '''
    Generalized class for manipulating galaxy SEDs.
    '''
    
    def __init__(self, wav_rest, 
                Lnu=None, Llam=None, Fnu=None, Flam=None, 
                nuLnu=None, lamLlam=None, nuFnu=None, lamFlam=None, redshift=0):
        
        if sum(x is not None for x in (Lnu,Llam,Fnu,Flam,nuLnu,lamLlam,nuFnu,lamFlam)) != 1:
            print("Must supply exactly one specification of the SED fluxes")
            # self.logger.warning('No flux information provided, populating flam=fnu=0. If this is intended, you can ignore this message.')
            self.fnu = np.zeros(len(self.wav))

        # if not hasattr(wav_rest, "unit"):
        #     raise Exception
        # if not hasattr(self.fnu, "unit"):
        #     raise Exception
        # if isinstance(self.sed_units, str): self.sed_units = utils.unit_parser(self.sed_units)
        # if isinstance(self.wav_units, str): self.wav_units = utils.unit_parser(self.wav_units)

        self.wav_rest = wav_rest.to(config.default_wavelength_unit)
        self.fnu = Fnu

    #################################################################################
    def resample(self, new_wavs):
        self.total = spectres.spectres(new_wavs, self.wav, self.total)
        self.wav = new_wavs
        return self.total

    def __repr__(self):
        wunit, funit = 'AA', 'ergscma'
        w = self.wav_rest.value
        f = self.fnu.value
        if len(self.wav_rest) > 4:
            wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
            fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnu.unit}'

        return f'''BRISKET-SED: wav: {wstr}\n             fnu: {fstr}'''

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        newobj = SED(wav)
        newobj.flux = self.flux + other.flux
        return newobj


    def to(self, unit, inplace=False):
        # if unit is wavelength or frquency, adjust x-units
        # if unit is flux or flux density, adjust y-units
        # if unit is tuple of (wavelength OR frequency, flux OR flux density), adjust both x and y-units
        pass

        
        # if 'spectral flux density' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Converting SED flux units to f_nu ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2 * (1 * self.wav_units)**2 / speed_of_light).to(self.sed_units).value
        # elif 'spectral flux density wav' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Keeping SED flux units in f_lam ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2).to(self.sed_units).value

    def measure_monochromatic_luminosity(self):
        pass
    def measure_slope(self):
        pass



    ########################################################################################################################
    @property
    def Lbol(self):
        return None

    @property
    def beta(self):
        '''UV spectral slope measured using the Calzetti et al. (1994) spectral windows'''
        w = self.wav_rest.to(u.angstrom).value
        windows = ((w>=1268)&(w<=1284))|((w>=1309)&(w<=1316))|((w>=1342)&(w<=1371))|((w>=1407)&(w<=1515))|((w>=1562)&(w<=1583))|((w>=1677)&(w<=1740))|((w>=1760)&(w<=1833))|((w>=1866)&(w<=1890))|((w>=1930)&(w<=1950))|((w>=2400)&(w<=2580))
        p = np.polyfit(np.log10(w[windows]), np.log10(self.flam[windows]), deg=1)
        return p[0]

    @property
    def properties(self):
        return dict(beta=self.beta, Lbol=self.Lbol)
    @property 
    def wav_obs(self):
        return self.wav_rest * (1+self.redshift)        
    @property 
    def freq_rest(self):
        return (speed_of_light/self.wav_rest).to(config.default_frequency_unit)
    @property 
    def freq_obs(self):
        return self.freq_rest / (1+self.redshift)
    @property 
    def energy_rest(self):
        return (plancks_constant * self.freq_rest).to(config.default_energy_unit)
    @property 
    def energy_obs(self):
        return self.energy_rest / (1+self.redshift)

    # things to compute automatically:
    # wavelength, frequency, energy
    # Lnu, Llam, Fnu, Flam
    # bolometric luminosity
    # sed.measure_window_luminosity((1400.0 * Angstrom, 1600.0 * Angstrom))
    # sed.measure_balmer_break()
    # sed.measure_d4000()
    # sed.measure_beta(windows='Calzetti94')









import numpy as np
import astropy.units as u
wav = np.linspace(1, 1000, 100) * u.angstrom
flux = np.zeros(len(wav)) * u.uJy
s = SED(wav_rest=wav, Fnu=flux)
print(s)

