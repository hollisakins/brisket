import sys
import spectres
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant

# temporary
from dotmap import DotMap
import astropy.units as u
config = DotMap(default_wavelength_unit=u.angstrom, default_frequency_unit=u.GHz, default_energy_unit=u.keV, default_fnu_unit=u.uJy, default_flam_unit=u.erg/u.s/u.cm**2/u.angstrom)

import matplotlib.pyplot as plt

class SED(object):
    '''
    Generalized class for manipulating galaxy SEDs.
    '''
    
    def __init__(self, wav_rest, 
                Lnu=None, Llam=None, fnu=None, flam=None, 
                nuLnu=None, lamLlam=None, nufnu=None, lamflam=None, redshift=0, verbose=True):
        
        self.redshift = redshift
        if sum(x is not None for x in (Lnu,Llam,fnu,flam,nuLnu,lamLlam,nufnu,lamflam)) == 0:
            if verbose: print('No flux/luminosity information provided, populating with zeros. If this is intended, you can ignore this message.')
            fnu = np.zeros(len(wav_rest)) * config.default_fnu_unit
        
        if sum(x is not None for x in (Lnu,Llam,fnu,flam,nuLnu,lamLlam,nufnu,lamflam)) != 1:
            # self.fnu = np.zeros(len(self.wav))
            raise Exception("Must supply exactly one specification of the SED fluxes")

        if not hasattr(wav_rest, "unit"):
            print(f"No wavelength units specified, adopting default ({config.default_wavelength_unit})")
            wav_rest = wav_rest * config.default_wavelength_unit            
        if fnu is not None:
            if not hasattr(fnu, "unit"):
                print(f"No units specified for fnu, adopting default ({config.default_fnu_unit})")
                fnu = fnu * config.default_fnu_unit
        if flam is not None:
            if not hasattr(flam, "unit"):
                print(f"No units specified for flam, adopting default ({config.default_flam_unit})")
                flam = flam * config.default_flam_unit
        if Lnu is not None:
            if not hasattr(Lnu, "unit"):
                print(f"No units specified for Lnu, adopting default ({config.default_Lnu_unit})")
                Lnu = Lnu * config.default_fnu_unit
        if Llam is not None:
            if not hasattr(Llam, "unit"):
                print(f"No units specified for Llam, adopting default ({config.default_Llam_unit})")
                Llam = Llam * config.default_Llam_unit
        if nuLnu is not None:
            if not hasattr(nuLnu, "unit"):
                print(f"No units specified for nuLnu, adopting default ({config.default_lum_unit})")
                nuLnu = nuLnu * config.default_lum_unit
        if lamLlam is not None:
            if not hasattr(lamLlam, "unit"):
                print(f"No units specified for lamLlam, adopting default ({config.default_lum_unit})")
                lamLlam = lamLlam * config.default_lum_unit
        if nufnu is not None:
            if not hasattr(nufnu, "unit"):
                print(f"No units specified for nufnu, adopting default ({config.default_flux_unit})")
                nufnu = nufnu * config.default_flux_unit
        if lamflam is not None:
            if not hasattr(lamflam, "unit"):
                print(f"No units specified for lamflam, adopting default ({config.default_flux_unit})")
                lamflam = lamflam * config.default_flux_unit


        #     raise Exception
        # if not hasattr(self.fnu, "unit"):
        #     raise Exception
        # if isinstance(self.sed_units, str): self.sed_units = utils.unit_parser(self.sed_units)
        # if isinstance(self.wav_units, str): self.wav_units = utils.unit_parser(self.wav_units)

        self.wav_rest = wav_rest.to(config.default_wavelength_unit)

        self._Lnu = Lnu
        self._Llam = Llam
        self._fnu = fnu
        self._flam = flam
        if lamLlam is not None: self._L = lamLlam
        elif nuLnu is not None: self._L = nuLnu
        else: self._L = None
        if lamflam is not None: self._f = lamflam
        elif nufnu is not None: self._f = nufnu
        else: self._f = None

    @property
    def _which(self):
        '''Used internally, string specification of the flux defined at construction'''
        if self._fnu is not None: return 'fnu'
        if self._flam is not None: return 'flam'
        if self._Lnu is not None: return 'Lnu'
        if self._Llam is not None: return 'Llam'
        if self._L is not None: return 'L'
        if self._f is not None: return 'f'

    @property
    def _y(self):
        '''Used internally, alias for the flux specification defined at construction'''
        if self._fnu is not None: return self._fnu
        if self._flam is not None: return self._flam
        if self._Lnu is not None: return self._Lnu
        if self._Llam is not None: return self._Llam
        if self._L is not None: return self._L
        if self._f is not None: return self._f
    @_y.setter
    def _y(self, value):
        if self._fnu is not None: self._fnu = value
        if self._flam is not None: self._flam = value
        if self._Lnu is not None: self._Lnu = value
        if self._Llam is not None: self._Llam = value
        if self._L is not None: self._L = value
        if self._f is not None: self._f = value

    @property
    def fnu(self):
        if self._fnu is not None:
            return (self._fnu).to(config.default_fnu_unit)
        elif self._flam is not None:
            return (self._flam * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._Lnu is not None:
            return (self._Lnu / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._Llam is not None:
            return (self._Llam / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._L is not None:
            return (self._L / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._f is not None:
            return (self._f / self.nu_obs).to(config.default_fnu_unit)
        else:
            raise Exception
    
    @property
    def flam(self):
        if self._fnu is not None:
            return (self._fnu / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._flam is not None:
            return (self._flam).to(config.default_flam_unit)
        elif self._Lnu is not None:
            return (self._Lnu / (4*np.pi*self.luminosity_distance**2) / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._Llam is not None:
            return (self._Llam / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._L is not None:
            return (self._L / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._f is not None:
            return (self._f / self.lam_obs).to(config.default_flam_unit)
        else:
            raise Exception

    #TODO define flam, Lnu, etc




    #################################################################################
    def resample(self, new_wavs, fill=0):
        xnew = new_wavs.to(self.wav_rest.unit).value
        x = self.wav_rest.value
        y = self._y.value
        self._y = spectres.spectres(xnew, x, y, fill=fill, verbose=False) * self._y.unit
        self.wav_rest = new_wavs
        return self._y

    def __repr__(self):
        w = self.wav_rest.value
        f = self.fnu.value
        wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
        return f'''BRISKET-SED: wav: {wstr}, flux {np.shape(f)}'''
        # if np.ndim(f) == 1:
            # if len(self.wav_rest) > 4:
            #     wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
            #     fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnu.unit}'
            # l = max((len(wstr),len(fstr)))
            # wstr = wstr.ljust(l+3)
            # fstr = fstr.ljust(l+3)
            # wstr += str(np.shape(w))
            # fstr += str(np.shape(f))

            # return f'''BRISKET-SED: wav: {wstr}, flux {np.shape(f)}'''
        # elif np.ndim(f)==2:
        #     if len(self.wav_rest) > 4:
        #         wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
        #         fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnu.unit}'
        #     l = max((len(wstr),len(fstr)))
        #     wstr = wstr.ljust(l+3)
        #     fstr = fstr.ljust(l+3)
        #     wstr += str(np.shape(w))
        #     fstr += str(np.shape(f))

        #     return f'''BRISKET-SED: wav: {wstr}\n             fnu: {fstr}'''

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if not np.all(other.wav_rest==self.wav_rest):
            other.resample(self.wav_rest)
        newobj = SED(self.wav_rest, redshift=self.redshift, verbose=False)
        setattr(newobj, '_'+self._which, self._y + getattr(other, self._which))
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
        p = np.polyfit(np.log10(w[windows]), np.log10(self._flam[windows].value), deg=1)
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


    def plot(self, ax=plt.gca(), x='wav_rest', y='fnu', show=True, save=False):
        x_plot = getattr(self, x)
        y_plot = getattr(self, y)
        ax.plot(x_plot, y_plot)
        
        if show:
            plt.show()



# class PosteriorSED():
#     pass






import numpy as np
import astropy.units as u
wav = np.linspace(1, 3000, 1000) * u.angstrom
flux = np.ones((2,len(wav))) * 1e-18 * u.erg/u.s/u.cm**2/u.angstrom
s1 = SED(wav_rest=wav, flam=flux, redshift=1)
print(s1.fnu)

flux = np.zeros((2,len(wav))) * u.uJy
s2 = SED(wav_rest=wav, fnu=flux, redshift=1)

s = s1 + s2
print(s.fnu)
# print(s)

# s.plot()
# new_wav = np.linspace(1, 3000, 500) * u.angstrom
# s.resample(new_wav)
# # print(y)

# s.plot()

# print(s.beta)


# s.compute_photometry(['f070w'])
