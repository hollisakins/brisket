from __future__ import annotations
from collections.abc import Iterable

from copy import deepcopy
import sys
import astropy.units as u
import numpy as np  
import spectres
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant

from brisket import config
from brisket.console import setup_logger
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl


border_chars = config.border_chars

np_handled_array_functions = {}


class SED:
    '''
    Generalized class for manipulating galaxy SEDs.
    '''
    
    def __init__(self, 
                 wav_rest: Iterable[float], 
                 redshift: float = 0, 
                 verbose: bool = True, 
                 units: bool = True, 
                 **kwargs):
        '''
        wav_rest : array-like
            Rest-frame wavelengths of the SED.
        redshift : float, optional 
            Redshift of the source (defaults to 0).
        verbose : bool, optional
            Verbosity flag (defaults to True).
        units : bool, optional
            Whether to use units (defaults to True). If True, the class 
            expects wavelengths/fluxes to be provided with units via astropy.units, 
            and will assign the default units (specified in brisket.config) if none 
            are provided.
        kwargs : 
            Flux specification: 
            no more than one of 'Llam', 'Lnu', 'flam', 'fnu',
                        'nuLnu', 'lamLlam', 'nufnu', 'lamflam' 
            If none are provided, the SED will be populated with zeros (in L_lam).
        '''

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        self.units = units
        self.redshift = redshift
        self.wav_rest = wav_rest
        self.luminosity_distance = config.cosmo.luminosity_distance(self.redshift).to(u.pc)
        if self.redshift == 0:
            self.luminosity_distance = 10 * u.pc
        if not units:
            self.luminosity_distance = self.luminosity_distance.value

        self.flux_keys = ['Llam', 'Lnu', 'flam', 'fnu', 'nuLnu', 'lamLlam', 'nufnu', 'lamflam']
        self.flux_specs = [kwargs.get(k, None) for k in self.flux_keys]
        if sum(x is not None for x in self.flux_specs) == 0:
            self.logger.info('No flux/luminosity information provided, populating with zeros. If this is intended, you can ignore this message.')
            self.flux_specs[0] = np.zeros(len(wav_rest))
            self._which = 0
        
        elif sum(x is not None for x in self.flux_specs) != 1:
            self.logger.error("Must supply at most one specification of the SED fluxes"); sys.exit()
        
        else:
            self._which = [x is not None for x in self.flux_specs].index(True)
        self._which_str = self.flux_keys[self._which]

        if units:
            flux_default_units = [config.default_Llam_unit, config.default_Lnu_unit, config.default_flam_unit, config.default_fnu_unit, config.default_lum_unit, config.default_lum_unit, config.default_flux_unit, config.default_flux_unit]
            if not hasattr(self.wav_rest, "unit"):
                self.logger.info(f"No wavelength units specified, adopting default ({config.default_wavelength_unit})")
                self.wav_rest = self.wav_rest * config.default_wavelength_unit            
            for i in range(len(self.flux_keys)):
                if self.flux_specs[i] is not None:
                    if not hasattr(self.flux_specs[i], "unit"):
                        self.logger.info(f"No units specified for {self.flux_keys[i]}, adopting default ({flux_default_units[i]})")
                        self.flux_specs[i] *= flux_default_units[i]

        #     raise Exception
        # if not hasattr(self.fnu, "unit"):
        #     raise Exception
        # if isinstance(self.sed_units, str): self.sed_units = utils.unit_parser(self.sed_units)
        # if isinstance(self.wav_units, str): self.wav_units = utils.unit_parser(self.wav_units)

        if units:
            self.wav_rest = self.wav_rest.to(config.default_wavelength_unit)

        # self._Lnu = Lnu
        # self._Llam = Llam
        # self._fnu = fnu
        # self._flam = flam
        # if lamLlam is not None: self._L = lamLlam
        # elif nuLnu is not None: self._L = nuLnu
        # else: self._L = None
        # if lamflam is not None: self._f = lamflam
        # elif nufnu is not None: self._f = nufnu
        # else: self._f = None

        # which = 
        # if self._fnu is not None: self._which='fnu'
        # if self._flam is not None: self._which='flam'
        # if self._Lnu is not None: self._which='Lnu'
        # if self._Llam is not None: self._which='Llam'
        # if self._L is not None: self._which='L'
        # if self._f is not None: self._which='f'

    @property
    def _y(self):
        '''Used internally, alias for the flux specification defined at construction'''
        return self.flux_specs[self._which]

    @_y.setter
    def _y(self, value):
        self.flux_specs[self._which] = value

    def __getitem__(self, indices):
        newobj = deepcopy(self)
        newobj._y = newobj._y[indices]
        return newobj
    
    def __setitem__(self, indices, values):
        self._y[indices] = values

    @property
    def fnu(self):
        if self._which_str=='fnu':
            return (self._y).to(config.default_fnu_unit)
        elif self._which_str=='flam':
            return (self._y * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._which_str=='Lnu':
            return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._which_str=='Llam':
            return (self._y / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._which_str=='L':
            return (self._y / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._which_str=='f':
            return (self._y / self.nu_obs).to(config.default_fnu_unit)
        else:
            raise Exception
    
    @property
    def flam(self):
        if self._which_str=='fnu':
            return (self._y / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._which_str=='flam':
            return (self._y).to(config.default_flam_unit)
        elif self._which_str=='Lnu':
            return (self._y / (4*np.pi*self.luminosity_distance**2) / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._which_str=='Llam':
            return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._which_str=='L':
            return (self._y / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._which_str=='f':
            return (self._y / self.lam_obs).to(config.default_flam_unit)
        else:
            raise Exception

    #TODO define flam, Lnu, etc

    #################################################################################
    def resample(self, new_wavs, fill=0):
        if self.units:
            self._y = spectres.spectres(new_wavs.to(self.wav_rest.unit).value, self.wav_rest.value, self._y.value, fill=fill, verbose=False) * self._y.unit
        else:
            self._y = spectres.spectres(new_wavs, self.wav_rest, self._y, fill=fill, verbose=False)
        self.wav_rest = new_wavs
        return self._y

    def __repr__(self):
        if self.units:
            all_flux_defs = ['fnu','flam','Lnu','Llam','nufnu','lamflam','nuLnu','lamLlam']
            w = self.wav_rest.value
            f = self._y.value
            wstr = f'wav_rest: [{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {self.wav_rest.unit} {np.shape(w)}'
            if np.ndim(f) > 1:
                fstr1 = f'{self._which_str} (base): [...] {self._y.unit} {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: ?, Muv: ?, Lbol: ?'
            else:
                fstr1 = f'{self._which_str} (base): [{f[0]:.2f}, {f[1]:.2f}, ..., {f[-2]:.2f}, {f[-1]:.2f}] {self._y.unit} {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: {self.beta:.2f}, Muv: {self.Muv:.1f}, Lbol: ?'
            width = config.cols-2
        else:
            all_flux_defs = ['fnu','flam','Lnu','Llam','nufnu','lamflam','nuLnu','lamLlam']
            w = self.wav_rest
            f = self._y
            wstr = f'wav_rest: [{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {np.shape(w)}'
            if np.ndim(f) > 1:
                fstr1 = f'{self._which_str} (base): [...] {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: ?, Muv: ?, Lbol: ?'
            else:
                fstr1 = f'{self._which_str} (base): [{f[0]:.2f}, {f[1]:.2f}, ..., {f[-2]:.2f}, {f[-1]:.2f}] {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: {self.beta:.2f}, Muv: {self.Muv:.1f}, Lbol: ?'
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

    def __add__(self, other: SED) -> SED:
        if not np.all(other.wav_rest==self.wav_rest):
            other.resample(self.wav_rest)
        
        if self.units:
            newobj = SED(self.wav_rest, redshift=self.redshift, verbose=False)
            setattr(newobj, '_which_str', self._which_str)
            setattr(newobj, '_y', self._y + getattr(other, self._which_str))
        else:
            newobj = SED(self.wav_rest, redshift=self.redshift, verbose=False, units=False)
            setattr(newobj, '_which', self._which)
            setattr(newobj, '_which_str', self._which_str)
            setattr(newobj, '_y', self._y + other._y)

        return newobj
    
    def __mul__(self, other: int | float | np.ndarray) -> SED: 
        newobj = deepcopy(self)
        newobj._y = self._y * other
        return newobj

    def __array__(self, dtype=None, copy=None):
        return self._y
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in np_handled_array_functions:
            return NotImplemented
        # Allow subclasses (that don't override __array_function__) to handle SED objects
        # if not all(issubclass(t, self.__class__) for t in types):
        #     print('call __array_function__')
        #     return NotImplemented
        return np_handled_array_functions[func](*args, **kwargs)

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     if method == '__call__':
    #         print('?')
    #         # newobj = SED(self.wav_rest, redshift=self.redshift, verbose=False)
    #         # args = [np.array(x._y) if isinstance(x, SED) else x for x in inputs]
    #         # setattr(newobj, '_y', ufunc(*args, **kwargs))
    #     else:
    #         return NotImplemented


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

    def measure_window_luminosity(self, window):
        pass

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
        if self.units: w = self.wav_rest.to(u.angstrom).value
        else: w = self.wav_rest
        windows = ((w>=1268)&(w<=1284))|((w>=1309)&(w<=1316))|((w>=1342)&(w<=1371))|((w>=1407)&(w<=1515))|((w>=1562)&(w<=1583))|((w>=1677)&(w<=1740))|((w>=1760)&(w<=1833))|((w>=1866)&(w<=1890))|((w>=1930)&(w<=1950))|((w>=2400)&(w<=2580))
        p = np.polyfit(np.log10(w[windows]), np.log10(self.flam[windows].value), deg=1)
        return p[0]

    @property
    def Muv(self):
        w = self.wav_rest.to(u.angstrom).value
        tophat = (w > 1450)&(w < 1550)
        mUV = (np.mean(self.fnu[tophat])/(1+self.redshift)).to(u.ABmag).value
        return mUV - 5*(np.log10(self.luminosity_distance.to(u.pc).value)-1)

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


    def plot(self, ax: mpl.axes.Axes = None, 
             x: str = 'wav_rest', 
             y: str = 'fnu', 
             xscale: str = 'linear',
             yscale: str = 'linear',
             xunit: str | u.Unit = None,
             yunit: str | u.Unit = None,
             verbose_labels: bool = False,
             show: bool = False, 
             save: bool = False, 
             eng: bool = False, 
             xlim: tuple[float,float] = None, 
             ylim: tuple[float,float] = None, 
             step: bool = False, 
             **kwargs):

        x_plot = getattr(self, x)
        y_plot = getattr(self, y)     
        if xlim is None:
            xmin, xmax = np.min(x_plot), np.max(x_plot)
        else:
            xmin, xmax = xlim   
        
        if ylim is None:
            ymin, ymax = np.min(y_plot), np.max(y_plot)
        else:
            ymin, ymax = ylim



        if eng: 
            x_plot = x_plot.to(u.m)
        else: 
            if xunit is None:
                if xmax.to(u.micron).value < 1:
                    xunit = u.angstrom
                else:#if xmax.to(u.micron).value < 100:
                    xunit = u.micron
                # else:
                #     xunit = u.mm
            elif isintance(xunit, str):
                xunit = utils.unit_parser(xunit)
            elif isintance(yunit, str):
                yunit = utils.unit_parser(yunit)

        x_plot = x_plot.to(xunit)
        # y_plot = y_plot.to(yunit)

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
        if r'\mathrm{\mu' in yunitstr:
            yunitstr = yunitstr.replace(r'\mathrm{\mu', r'\mu\mathrm{')
        ylabel +=  fr' [{yunitstr}]'
        
        xunitstr = x_plot.unit.to_string(format="latex_inline")
        if r'\mathrm{\mu' in xunitstr:
            xunitstr = xunitstr.replace(r'\mathrm{\mu', r'\mu\mathrm{')
        xlabel +=  fr' [{xunitstr}]'

    
        with plt.style.context('brisket.brisket'):
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,2.5))
            else:
                fig = plt.gcf()
                fig.canvas.draw()

            if step:
                ax.step(x_plot, y_plot, where='mid', **kwargs)
            else:
                ax.plot(x_plot, y_plot, **kwargs)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if eng:
                ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m', places=1))
            
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

            if show:
                plt.show()

        return fig, ax


    def implements(np_function):
        "Register an __array_function__ implementation for SED objects."
        def decorator(func):
            np_handled_array_functions[np_function] = func
            return func
        return decorator

    @implements(np.sum)
    def sum(sed: SED, axis: int | None = None) -> SED:
        "Implementation of np.sum for SED objects"
        newobj = deepcopy(sed)
        newobj._y = np.sum(sed._y, axis=axis)
        return newobj

    @implements(np.convolve)
    def convolve(sed: SED, kernel: np.ndarray, mode: str = 'full') -> SED:
        newobj = deepcopy(sed)
        newobj._y = np.convolve(sed._y, kernel, mode=mode)
        return newobj

    @implements(np.shape)
    def shape(sed: SED) -> tuple:
        "Implementation of np.shape for SED objects"
        return np.shape(sed._y)

    @property 
    def T(self):
        newobj = deepcopy(self)
        newobj._y = np.transpose(self._y)
        return newobj






# class PosteriorSED():
#     pass


# flux = np.ones(len(wav)) * u.uJy
if __name__ == '__main__':
    wav = np.linspace(100, 8000, 5000)
    Llam = np.zeros(len(wav))
    sed = SED(wav_rest=wav, Llam = Llam, redshift=1, units=False)
    
    # print(np.convolve(sed, np.ones(10), mode='full'))

    # newsed = sed.T * Llam.T
    # import numpy as np
    # print(np.sum(sed, axis=0))



# print(s2)

# fig, ax = s1.plot(y='flam')
# s2.plot(y='flam', xscale='log', yscale='log', eng=True)
# s.plot(ax=ax, y='flam')
# plt.show()



# s.compute_photometry(['f070w'])
