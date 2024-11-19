import sys
import spectres
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant

from brisket import config
import matplotlib.pyplot as plt
import matplotlib as mpl

border_chars = config.border_chars

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

        if self._fnu is not None: self._which='fnu'
        if self._flam is not None: self._which='flam'
        if self._Lnu is not None: self._which='Lnu'
        if self._Llam is not None: self._which='Llam'
        if self._L is not None: self._which='L'
        if self._f is not None: self._which='f'

    @property
    def _y(self):
        '''Used internally, alias for the flux specification defined at construction'''
        if self._which=='fnu': return self._fnu
        if self._which=='flam': return self._flam
        if self._which=='Lnu': return self._Lnu
        if self._which=='Llam': return self._Llam
        if self._which=='L': return self._L
        if self._which=='f': return self._f

    @_y.setter
    def _y(self, value):
        if self._which=='fnu': self._fnu = value
        if self._which=='flam': self._flam = value
        if self._which=='Lnu': self._Lnu = value
        if self._which=='Llam': self._Llam = value
        if self._which=='L': self._L = value
        if self._which=='f': self._f = value

    @property
    def fnu(self):
        if self._which=='fnu':
            return (self._fnu).to(config.default_fnu_unit)
        elif self._which=='flam':
            return (self._flam * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._which=='Lnu':
            return (self._Lnu / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._which=='Llam':
            return (self._Llam / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._which=='L':
            return (self._L / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._which=='f':
            return (self._f / self.nu_obs).to(config.default_fnu_unit)
        else:
            raise Exception
    
    @property
    def flam(self):
        if self._which=='fnu':
            return (self._fnu / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._which=='flam':
            return (self._flam).to(config.default_flam_unit)
        elif self._which=='Lnu':
            return (self._Lnu / (4*np.pi*self.luminosity_distance**2) / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._which=='Llam':
            return (self._Llam / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._which=='L':
            return (self._L / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._which=='f':
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
        all_flux_defs = ['fnu','flam','Lnu','Llam','nufnu','lamflam','nuLnu','lamLlam']
        w = self.wav_rest.value
        f = self._y.value
        wstr = f'wav_rest: [{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {self.wav_rest.unit} {np.shape(w)}'
        fstr1 = f'{self._which} (base): [{f[0]:.2f}, {f[1]:.2f}, ..., {f[-2]:.2f}, {f[-1]:.2f}] {self._y.unit} {np.shape(w)}'
        fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which])) 
        betastr = f'beta: {self.beta:.2f}, Muv: ?, Lbol: ?'
        width = config.cols-2
        # width = np.max([width, len(wstr)+4])
        # border_chars = '═║╔╦╗╠╬╣╚╩╝'
        outstr = border_chars[2] + border_chars[0]*width + border_chars[4]
        outstr += '\n' + border_chars[1] + 'BRISKET-SED'.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + wstr.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + fstr1.center(width) + border_chars[1]
        outstr += '\n' + border_chars[1] + fstr2.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + betastr.center(width) + border_chars[1]
        outstr += '\n' + border_chars[8] + border_chars[0]*width + border_chars[10]
        return outstr
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
        # print(newobj._which)
        setattr(newobj, '_which', self._which)
        setattr(newobj, '_y', self._y + getattr(other, self._which))
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
        p = np.polyfit(np.log10(w[windows]), np.log10(self.flam[windows].value), deg=1)
        return p[0]

    @property
    def Muv(self):
        return -22

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


    def plot(self, ax=None, x='wav_rest', y='fnu', 
             xscale='linear', yscale='linear',
             verbose_labels=False,
             show=False, save=False, eng=False):

        x_plot = getattr(self, x)
        y_plot = getattr(self, y)        
        if eng: 
            x_plot = x_plot.to(u.m)

        if y == 'fnu':
            ylabel = r'$f_{\nu}$'
        elif y == 'flam':
            ylabel = r'$f_{\lambda}$'

        if x == 'wav_rest':
            if verbose_labels: xlabel = 'Rest Wavelength'
            else: xlabel = r'$\lambda_{\rm rest}$'
        elif x == 'wav_obs':
            if verbose_labels: xlabel = 'Observed Wavelength'
            else: xlabel = r'$\lambda_{\rm obs}$'
        elif x == 'freq_rest':
            if verbose_labels: xlabel = 'Rest Frequency'
            else: xlabel = r'$\nu_{\rm rest}$'
        elif x == 'freq_obs':
            if verbose_labels: xlabel = 'Observed Frequency'
            else: xlabel = r'$\nu_{\rm obs}$'
        
        yunitstr = y_plot.unit.to_string(format="latex_inline")
        if r'erg\,\mathring{A}^{-1}' in yunitstr:
            yunitstr = yunitstr.replace(r'\,\mathring{A}^{-1}', r'\,')
            yunitstr = yunitstr.replace(r'\,cm^{-2}', r'\,cm^{-2}\,\mathring{A}^{-1}')
        ylabel +=  fr' [{yunitstr}]'
        xunitstr = x_plot.unit.to_string(format="latex_inline")
        xlabel +=  fr' [{xunitstr}]'

    
        with plt.style.context('brisket.brisket'):
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,2.5))
            else:
                fig = plt.gcf()
            ax.plot(x_plot, y_plot)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if eng:
                ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m', places=1))

            if show:
                plt.show()

        return fig, ax



# class PosteriorSED():
#     pass






import numpy as np
import astropy.units as u
wav = np.linspace(1, 3000, 1000) * u.angstrom
flux = np.ones(len(wav)) * 1e-18 * u.erg/u.s/u.cm**2/u.angstrom
s1 = SED(wav_rest=wav, flam=flux, redshift=1)

flux = np.ones(len(wav)) * u.uJy
s2 = SED(wav_rest=wav, fnu=flux, redshift=1)
print(s2)

# fig, ax = s1.plot(y='flam')
# s2.plot(y='flam', xscale='log', yscale='log', eng=True)
# s.plot(ax=ax, y='flam')
# plt.show()



# s.compute_photometry(['f070w'])
