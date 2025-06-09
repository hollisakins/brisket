from __future__ import annotations

from functools import cached_property
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import sys, os
import astropy.units as u
from abc import ABC, abstractmethod
from typing import List
import numpyro.distributions as dist

from .parameters import Parameter
from .utils.filters import Filters
from . import config

# obs = brisket.Photometry(filters=filters, fluxes=fluxes, errors=errors)

# obs._n_phot # should return 1
# obs._phot_list # should return a list with one Photometry object
# obs._phot_list[0] # should return the Photometry object itself

# # This combined observation also needs to work with the same syntax
# photometry = brisket.Photometry(filters=filters, fluxes=fluxes, errors=errors)
# spectrum1 = brisket.Spectrum(wavs1, fluxes1, errors1, calibration=calib1)
# spectrum2 = brisket.Spectrum(wavs2, fluxes2, errors2, calibration=calib2)
# obs = photometry + spectrum1 + spectrum2

# obs._n_phot # should return 1
# obs._phot_list # should return a list with one Photometry object
# obs._n_spec # should return 2
# obs._spec_list # should return a list with the two Spectrum objects


class Observation(ABC):
    """
    Abstract base class for astronomical observations.
    
    Provides a unified interface for handling photometry and spectral data,
    supporting both single datasets and combinations of multiple datasets.
    """
    
    def __init__(self, **kwargs):
        self._phot_list: List['Photometry'] = []
        self._spec_list: List['Spectrum'] = []

        self.parameters: Dict[str, Parameter] = {}
        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self.parameters[key] = value
    
    @property
    def _n_phot(self) -> int:
        """Number of photometric datasets."""
        return len(self._phot_list)
    
    @property
    def _n_spec(self) -> int:
        """Number of spectroscopic datasets."""
        return len(self._spec_list)
    
    def __add__(self, other: 'Observation') -> 'CombinedObservation':
        """
        Combine observations using the + operator.
        
        Parameters
        ----------
        other : Observation
            Another observation to combine with this one.
            
        Returns
        -------
        CombinedObservation
            A new combined observation containing all datasets.
        """
        if not isinstance(other, Observation):
            raise TypeError("Can only add Observation objects together")
        
        combined = CombinedObservation()
        combined._phot_list = self._phot_list.copy() + other._phot_list.copy()
        combined._spec_list = self._spec_list.copy() + other._spec_list.copy()
        
        return combined
    
    @abstractmethod
    def _validate(self):
        """Validate the observation data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict(self, model: 'Model') -> jnp.ndarray:
        """Predict the observation given a model. Must be implemented by subclasses."""
        pass

    @property
    def has_data(self) -> bool:
        """Check if the observation contains any data."""
        for phot in self._phot_list:
            if phot.flux is not None and len(phot.flux) > 0:
                return True
        for spec in self._spec_list:
            if spec.flux is not None and len(spec.flux) > 0:
                return True

    @cached_property
    def model_wavelengths(self):
        """
        Optimized wavelength grid for model evaluation.
        Uses resolution information from any photometric or spectroscopic components to 
        determine the spectral resolution to use for the underlying model SED. 
        """
        
        oversample = config.oversample_wavelengths
        smoothing_kernel_size = config.resolution_smoothing_kernel_size
        max_wav = config.max_wavelength
        R_default = config.default_resolution
        
        R_wav = np.logspace(0, np.log10(max_wav), 1000)
        R = np.full_like(R_wav, R_default)

        for phot in self._phot_list:
            in_phot_range = np.logical_and(
                R_wav > phot.wav_range[0]/(1+config.max_redshift), 
                R_wav < phot.wav_range[1]
            )
            new_R = phot.R * oversample
            R[in_phot_range & (R <= new_R)] = new_R

        for spec in self._spec_list:
            in_spec_range = np.logical_and(
                R_wav > spec.wav_range[0]/(1+config.max_redshift),
                R_wav < spec.wav_range[1]
            )
            new_R = spec.R * oversample
            R[in_spec_range & (R <= new_R)] = new_R

        # Smooth the resolution curve to avoid weird edge effects
        R = np.convolve(R, np.ones(smoothing_kernel_size)/smoothing_kernel_size, mode='same')
        
        # Use mean resolution to determine number of wavelength points
        mean_R = np.mean(R)
        nwav = int(np.log10(max_wav) * mean_R * 4.5)
        wavelengths = np.logspace(0, np.log10(max_wav), nwav)
        self.R_array = np.interp(wavelengths, R_wav, R)

        return wavelengths

    def predict(self, model: 'Model') -> 'Observation':
        """Predict the photometry given a model."""

        # Update the model's observation attribute and wavelengths
        # This also recomputes any preprocessing steps in the model
        if model.obs is None or model.obs is not self:
            model.obs = self
            model._update_wavelengths(self.model_wavelengths)

        if not model.evaluated:
            model.evaluate()

        # Resample the filter curves to the model wavelengths
        for phot in self._phot_list:
            if phot.filters.wavelengths is None or (phot.filters.wavelengths != self.model_wavelengths).any():
                phot.filters.resample_filter_curves(self.model_wavelengths)

        self.mod = self._predict(model)
        return self

class Photometry(Observation):
    """
    Container for photometric observations.
    
    Parameters
    ----------
    filters : array-like or brisket.filters.Filters
        Filters
    """
    
    # @u.quantity_input
    def __init__(self, 
        name: str,
        filters: list | npt.ArrayLike | Filters, 
        fnu: list | npt.ArrayLike | u.Quantity['spectral flux density'] = None,
        fnu_err: list | npt.ArrayLike | u.Quantity['spectral flux density'] = None,
        flam: list | npt.ArrayLike | u.Quantity['spectral flux density'] = None,
        flam_err: list | npt.ArrayLike | u.Quantity['spectral flux density'] = None,
        fnu_units: str | u.Unit = 'uJy',
        flam_units: str | u.Unit = 'ergscm2a', 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.filters = filters
        self.fnu = fnu
        self.fnu_err = fnu_err
        self.flam = flam
        self.flam_err = flam_err
        self.fnu_units = fnu_units
        self.flam_units = flam_units
        
        self._validate()
        
        # Add this photometry object to its own phot_list
        self._phot_list = [self]


    
    def _validate(self):
        """Validate photometric data."""
        if len(self.filters) == 0:
            raise ValueError("Must provide at least one photometric measurement")
    
        if not isinstance(self.filters, Filters):
            self.filters = Filters(self.filters)

        self.wav_obs = self.filters.wav
        self.min_wav_obs = self.filters.wav_min
        self.max_wav_obs = self.filters.wav_max

        if self.fnu is None and self.flam is None:
            # No data 
            self.flux_type = None
            self.flux = None
            self.err = None
        
        else:
            if self.fnu is not None and self.flam is None:
                # fnu provided
                self.flux_type = 'fnu'
                self.flux = self.fnu
                self.err = self.fnu_err

            if self.fnu is None and self.flam is not None: 
                # flam provided
                self.flux_type = 'flam'
                self.flux = self.flam
                self.err = self.flam_err
        
            if len(self.filters) != len(self.flux) or len(self.filters) != len(self.err):
                raise ValueError("filters, fluxes, and errors must have the same length")

            self.flux = jnp.asarray(self.flux)
            self.err = jnp.asarray(self.err)
        
        if self.fnu is not None and self.flam is not None: 
            # can't provide both!
            raise ValueError("Cannot provide both fnu and flam fluxes. Please provide only one type of flux measurement.")


        
    def __repr__(self):
        return f"Photometry(n_bands={len(self.filters)})"

    @property
    def R(self):
        # compute pseudo-resolution based on the input filters 
        dwav = self.filters.wav_max - self.filters.wav_min
        return np.max(self.filters.wav/dwav)

    @property
    def wav_range(self):
        return np.min(self.filters.wav_min), np.max(self.filters.wav_max)

    @property
    def likelihood_distribution(self):
        """Return the numpyro likelihood distribution for the photometry."""
        if not self.has_data:
            raise ValueError("Photometry has no data to compute likelihood")

        return dist.Normal(self.mod, self.err)

    def __len__(self):
        return len(self.filters)

    def _predict(self, model: 'Model') -> jnp.ndarray:
        mod = self.filters.get_photometry(model.sed, model.redshift)
        return mod

class Spectrum(Observation):
    """
    Container for spectroscopic observations.
    
    Parameters
    ----------
    wavelengths : array-like
        Wavelength array
    fluxes : array-like
        Flux measurements
    errors : array-like
        Flux uncertainties
    calibration : optional
        Calibration information (implementation-specific)
    """
    
    @u.quantity_input
    def __init__(self, 
        name: str,
        wavelengths: list | npt.ArrayLike | u.Quantity['length'], 
        fnu: list | npt.ArrayLike | u.Quantity[''] = None,
        fnu_err: list | npt.ArrayLike | u.Quantity[''] = None,
        flam: list | npt.ArrayLike | u.Quantity[''] = None,
        flam_err: list | npt.ArrayLike | u.Quantity[''] = None,
        mask: list | npt.ArrayLike = None,
        calibration: Model = None,
        noise: Model = None,
        **kwargs,
    ):


        super().__init__()
        self.wavelengths = wavelengths
        self.fnu = fnu
        self.fnu_err = fnu_err
        self.flam = flam
        self.flam_err = flam_err
        self.mask = mask
        self.calibration = calibration
        self.noise = noise
        
        self._validate()
        
        # Add this spectrum object to its own spec_list
        self._spec_list = [self]
    
    def _validate(self):
        """Validate spectroscopic data."""

        if len(self.wavelengths) == 0:
            raise ValueError("Must provide at least one spectral point")
        
        if isinstance(self.wavelengths, u.Quantity):
            self.wavlength_unit = self.wavelengths.unit
            self._wavelengths = self.wavelengths.to(u.angstrom).value
            self.wavelengths = self.wavelengths.value
        else:
            print(f'Spectrum: no units provided for wavs, assuming {config.default_wavelength_unit}')
            self.wavlength_unit = u.Unit(config.default_wavelength_unit)
            self._wavelengths = (self.wavelengths*self.wavlength_unit).to(u.angstrom).value

        # Check wavelengths are monotonically increasing
        if not np.all(np.diff(self._wavelengths) > 0):
            raise ValueError("Wavelengths must be monotonically increasing")

        if self.fnu is None and self.flam is None:
            # No data 
            self.flux_type = None
            self.flux = None
            self.err = None
        
        else:
            if self.fnu is not None and self.flam is None:
                # fnu provided
                self.flux_type = 'fnu'
                self.flux = self.fnu
                self.err = self.fnu_err

            if self.fnu is None and self.flam is not None: 
                # flam provided
                self.flux_type = 'flam'
                self.flux = self.flam
                self.err = self.flam_err
        
            if len(self.filters) != len(self.flux) or len(self.filters) != len(self.err):
                raise ValueError("filters, fluxes, and errors must have the same length")
        
        if self.fnu is not None and self.flam is not None: 
            # can't provide both!
            raise ValueError("Cannot provide both fnu and flam fluxes. Please provide only one type of flux measurement.")

        if len(self.wavelengths) != len(self.flux) or len(self.wavelengths) != len(self.err):
            raise ValueError("wavelengths, fluxes, and errors must have the same length")


        # Remove points at the edges of the spectrum with zero flux.
        if self.flux is not None:
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
        
    @property
    def R(self):
        # compute pseudo-resolution based on the input spec_wavs
        return np.max((self._wavelengths[1:]+self._wavelengths[:-1])/2/np.diff(self._wavelengths))

    @property
    def wav_range(self):
        return np.min(self._wavelengths), np.max(self._wavelengths)

    def __repr__(self):
        return f"Spectrum(n_points={len(self.wavelengths)}, λ_range=[{self.wavelengths[0]:.1f}, {self.wavelengths[-1]:.1f}])"

    def predict(self, model: 'Model') -> 'Observation':
        """Predict the spectrum given a model."""

        sed = model.sed
        raise NotImplementedError("Spectrum prediction not implemented yet")
        return self

class CombinedObservation(Observation):
    """
    Container for combined observations (photometry + spectra).
    
    This class is created automatically when observations are combined
    using the + operator.
    """
    
    def __init__(self):
        super().__init__()
    
    def _validate(self):
        """Validate combined observation data."""
        if self._n_phot == 0 and self._n_spec == 0:
            raise ValueError("Combined observation must contain at least one dataset")
    
    def __repr__(self):
        return f"CombinedObservation(n_phot={self._n_phot}, n_spec={self._n_spec})"
