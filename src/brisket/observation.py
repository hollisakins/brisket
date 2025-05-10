"""
Module for handling observational data in harmonizer. 

The ``Observation`` class serves as the interface between idealized models generated with the 
``Model`` class and actual, observed data. It can be used to provide ``Model`` with the 
necessary info to generate proper synthetic observations (photometry or spectra) and to provide 
the ``Fitter`` class with real data to fit. 

Harmonizer provides two subclasses of ``Observation``, ``Photometry`` and ``Spectrum`` for handling
photometry and spectroscopic data, respectively. 

.. highlight:: python

:: 

    obs = Observation()
    obs.add_phot(filters, fluxes, errors)
    obs.add_spec(wavs, fluxes, errors)

Generating a simple model SED does not require specifying any information about the observation. 
However, it will not yield any observables (photometry/spectra), only the internal model SED, and
will be generated at the configured default wavelength resolution (``config.R_default``), which 
may not be sufficient for predicting observables.  

::

    mod = ModelGalaxy(params) 

You can add an observation after the fact, which will determine the optimal wavelength sampling and 
re-sample the model 

::

    mod.add_obs(obs) # resamples to optimized wavelength resolution

However, it is often better to just provide the observation info from the initial ModelGalaxy construction: 

::

    mod = ModelGalaxy(params, obs=obs)

When fitting data, the Observation class becomes the mechanism for providing the data to fit. 

::

    obs = Photometry(filters, fluxes, errors) + Spectrum(wavs, fluxes, errors, calib)
    result = Fitter(params, obs).fit()
"""
from __future__ import annotations


import numpy as np
import sys, os
import astropy.units as u

from .parameters import Params
from . import config
from .utils.filters import Filters
from .utils import exceptions
from .utils.console import setup_logger



# # or a more complicated observation
# import harmonizer
# phot_optical = harmonizer.Photometry(filters=filters, fluxes=fluxes, errors=errors)
# phot_fir = harmonizer.Photometry(filters=[harmonizer.filters.TopHat(850*u.micron, 1*u.micron)], fluxes=[...], errors=[...])
# spec_G235M = harmonizer.Spectrum(wavs, fluxes, errors, calibration='calib_g235m')
# spec_G395M = harmonizer.Spectrum(wavs, fluxes, errors, calibration='calib_g395m')
# observation = phot_optical + phot_fir + spec_G235M + spec_G395M



class Observation:
    """
    A container for observational data.

    """
    def __init__(self, 
                 verbose: bool = True, 
                 **kwargs
        ):

        self.logger = setup_logger(__name__, verbose)

        # Initialize the various models and resample to the internal, optimized wavelength grid
        self.logger.info('Preparing observation models')

        self.params = Params(verbose=verbose, model=self)
        for name, model_component in kwargs.items():
            self.params.add_child(name, params=model_component.params)
            self.params[name].model = model_component
        
        if verbose:
            self.params.print_tree()
    
    # @property
    # def phot_list(self):
    #     if isinstance(self, Spectrum):
    #         return None
    #     return self._phot
    
    # @property
    # def phot(self):
    #     if isinstance(self, Spectrum):
    #         return None
    #     if len(self._phot) == 1:
    #         return self._phot[0]
    #     else:
    #         return self._phot
    
    # @property 
    # def N_phot(self):
    #     return len(self._phot)
    
    # def add_phot(self, *args, **kwargs):
    #     phot = Photometry(*args, **kwargs)
    #     self._phot.append(phot)
    
    # @property
    # def spec_list(self):
    #     if isinstance(self, Photometry):
    #         return None
    #     return self._spec

    # @property
    # def spec(self):
    #     if isinstance(self, Photometry):
    #         return None
    #     if len(self._spec) == 1:
    #         return self._spec[0]
    #     else:
    #         return self._spec

    # @property 
    # def N_spec(self):
    #     return len(self._spec)

    # def add_spec(self, *args, **kwargs):
    #     spec = Spectrum(*args, **kwargs)
    #     self._spec.append(spec)


class Photometry:

    def __init__(self, 
                 filters: list | np.ndarray | brisket.filters.Filters, 
                 fnu: list | np.ndarray | u.Quantity = None, 
                 fnu_err: list | np.ndarray | u.Quantity = None, 
                 flam: list | np.ndarray | u.Quantity = None, 
                 flam_err: list | np.ndarray | u.Quantity = None, 
                 fnu_units: str | u.Unit = 'uJy',
                 flam_units: str | u.Unit = 'ergscm2a', 
                 verbose: bool = False, 
                 **kwargs):
        
        self.params = Params()
        self.verbose = verbose
        self.filters = filters
        # self.fluxes = fluxes
        # self.errors = errors

        if not isinstance(self.filters, Filters):
            self.filters = Filters(self.filters)

        self.wav = self.filters.wav
        self.wav_min = self.filters.wav_min
        self.wav_max = self.filters.wav_max

    @property
    def R(self):
        # compute pseudo-resolution based on the input filters 
        dwav = self.filters.wav_max - self.filters.wav_min
        return np.max(self.filters.wav/dwav)

    @property
    def wav_range(self):
        return np.min(self.filters.wav_min), np.max(self.filters.wav_max)
    
    def __len__(self):
        return len(self.filters)

    def predict(self, sed):
        self.logger.info('Providing photometry prediction in fnu units')
        sed.convert_units(yunit=u.uJy)
        f = self.filters.get_photometry(sed['total'], redshift=sed.redshift)
        args = {'redshift':sed.redshift, 'total':f}
        for component in sed.components:
            f = self.filters.get_photometry(sed[component], redshift=sed.redshift)
            args.update({component:f})
        self.update(**args)
        return self

    # def __repr__(self):
    #     out = ''
    #     return out



class Spectrum:

    def __init__(self, 
                 wavs: list | np.ndarray | u.Quantity['length'], 
                 flam: list | np.ndarray | u.Quantity[''] = None,
                 flam_err: list | np.ndarray | u.Quantity[''] = None,
                 fnu: list | np.ndarray | u.Quantity[''] = None,
                 fnu_err: list | np.ndarray | u.Quantity[''] = None,
                 mask: list | np.ndarray = None,
                 verbose: bool = True,
                 **kwargs):
        

        self.verbose = verbose

        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        self.params = self.validate(kwargs)
        
        if isinstance(wavs, u.Quantity):
            self.wavs = wavs
            self.wavs_unit = self.wavs.unit
            self.wavs = wavs.to(u.angstrom).value
        else:
            self.logger.warning(f'Spectrum: no units provided for wavs, assuming {config.default_wavelength_unit}')
            self.wavs = np.array(wavs)
            self.wavs_unit = u.Unit(config.default_wavelength_unit)
    
        self._R = R
        # self.errors # apply mask

        if fnu is None and flam is None:
            # No data 
            self.flux_type = None
            self.flux = None
            self.err = None

        elif fnu is not None:
            # fnu provided
            self.flux_type = 'fnu'
            self.flux = fnu
            self.err = fnu_err

        elif flam is not None: 
            # flam provided
            self.flux_type = 'flam'
            self.flux = flam
            self.err = flam_err
        
        else: 
            # can't provide both!
            raise ValueError()

        if self.flux is not None:
            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.flux[startn] == 0.:
                startn += 1
            endn = 0
            while self.flux[-endn-1] == 0.:
                endn += 1
            if endn == 0:
                self.flux = self.flux[startn:]
                self.wavs = self.wavs[startn:]
            else:
                self.flux = self.flux[startn:-endn]
                self.wavs = self.wavs[startn:-endn]
        if self.err is not None:
            if endn == 0:
                self.err = self.err[startn:]
            else:
                self.err = self.err[startn:-endn]

    def validate(self, kwargs):
        """
        Validate/parse parameters for the spectral model.
        """
        params = Params()

        if 'calibration' in kwargs:
            raise exceptions.NotImplementedError('Calibration not implemented yet.')
        
        for kwarg in kwargs:
            self.logger.warning(f"Ignoring unexpected parameter '{kwarg}'.")

        return params

    @property
    def R(self):
        if self._R is not None:
            return self._R
        # compute pseudo-resolution based on the input spec_wavs
        return np.max((self.wavs.value[1:]+self.wavs.value[:-1])/2/np.diff(self.wavs.value))

    def _mask(self, spec):
        """ Set the error spectrum to infinity in masked regions. """
        pass

    def __add__(self, other):
        result = Observation()
        # result.photometry += self

    @property
    def wav_range(self):
        return np.min(self.wavs.to(u.angstrom).value), np.max(self.wavs.to(u.angstrom).value)


    def predict(self, wavelengths, sed, redshift, params):
        """
        Predict the spectrum based on the input SED. 
        Wavelengths are assumed to be in Angstroms. 
        """
        self.logger.info('Predicting observed spectrum')

        # Resample the spectrum to the observation wavelengths
        fluxes = spectres.spectres(self.wavs, redshifted_wavs, spectrum, fill=0)
        return fluxes


class SpectralResolutionModel:
    """
    A class for handling spectral resolution models. 
    """
    model_type = 'calibration'

    def __init__(self, 
                 verbose: bool = False,
                 **kwargs
        ):

        self.name = None
        self.params = self.validate(kwargs)
        
    def validate(self, kwargs):
        params = Params()
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']

        if 'veldisp' in kwargs:
            params['veldisp'] = kwargs['veldisp']
            del kwargs['veldisp']

        if 'R_curve' in kwargs:
            params['R_curve'] = kwargs['R_curve']
            del kwargs['R_curve']

            if 'f_LSF' in kwargs:
                params['f_LSF'] = kwargs['f_LSF']
                del kwargs['f_LSF']

        else:
            if 'R' in kwargs:
                params['R'] = kwargs['R']
                del kwargs['R']
            
            else:
                self.logger.warning('No spectral resolution information provided, model will be generated at the default resolution.')
                params['R'] = config.R_default

        return params


    def convolve(self, wav_obs, sed, params):

        if 'veldisp' in params:
            vres = 2.998e5 / self.R /2. # TODO what is self.R
            sigma_pix = params["veldisp"] / vres
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(sed, kernel, mode="valid")
            spec_wavs = wav_obs[k_size:-k_size]
        else:
            spectrum = sed
            spec_wavs = wav_obs


        if 'R_curve' in kwargs:
            params['R_curve'] = kwargs['R_curve']
            del kwargs['R_curve']

            if 'f_LSF' in kwargs:
                params['f_LSF'] = kwargs['f_LSF']
                del kwargs['f_LSF']

        else:
            if 'R' in kwargs:
                params['R'] = kwargs['R']
                del kwargs['R']
            


        if "R_curve" in list(model_comp):
            oversample = 4  # Number of samples per FWHM at resolution R
            new_wavs = self._get_R_curve_wav_sampling(oversample=oversample)

            # spectrum = np.interp(new_wavs, redshifted_wavs, spectrum)
            spectrum = spectres.spectres(new_wavs, redshifted_wavs,
                                         spectrum, fill=0)
            redshifted_wavs = new_wavs

            sigma_pix = oversample/2.35  # sigma width of kernel in pixels
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            # Disperse non-uniformly sampled spectrum
            spectrum = np.convolve(spectrum, kernel, mode="valid")
            redshifted_wavs = redshifted_wavs[k_size:-k_size]
