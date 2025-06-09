
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import astropy.units as u
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Callable
from collections import defaultdict

from ..parameters import Parameter, ParameterRegistry, ParameterManager
from ..utils.units import (
    sanitize_wavelength_array, sanitize_velocity, sanitize_mass, 
    sanitize_age, SEDUnits, INTERNAL_SED_UNIT
)
from .. import config


def create_default_wavelength_grid() -> jnp.ndarray:
    """
    Create the default wavelength grid for model evaluation.
    
    Returns
    -------
    jnp.ndarray
        Default wavelength grid in Angstroms
    """
    max_wav = config.max_wavelength
    R = config.default_resolution
    nwav = int(jnp.log10(max_wav) * R * 4.5)
    return jnp.logspace(0, jnp.log10(max_wav), nwav)

class Model(ABC):
    """
    Base class for all model components in brisket.
    
    Each model component:
    1. Does expensive preprocessing during __init__
    2. Registers its parameters with the global registry
    3. Implements fast evaluation given parameter values
    4. Supports functional composition via __call__
    5. Generates SEDs in internal units (L_sun/Å)
    
    Unit Conventions:
    - Internal wavelengths: Angstroms
    - Internal SEDs: L_sun/Å (luminosity per unit wavelength)
    - All parameters converted to internal units during init
    """

    deterministic_properties = []  # List of properties to save as deterministic in NumPyro
    sed_unit = INTERNAL_SED_UNIT  # All models generate SEDs in this unit
    
    def __init__(self, **kwargs):
        self.preprocessed_data = {}
        self.obs = None  # Will be set by the observation object
        self.sed = None  # Will hold the evaluated spectral energy distribution
        
        # Use ParameterManager for all models
        self.parameter_manager = ParameterManager()
        
        # Store parameters in manager (single source of truth)
        for key, value in kwargs.items():
            self.parameter_manager.add_parameter(self, key, value)
            
            if key == 'redshift':
                self.redshift = value  # Special case for redshift
        
        # Only require redshift for models that need it (can be overridden in subclasses)
        # Redshift can be inherited from composite models, so don't require it at init
        if not hasattr(self, 'redshift') and getattr(self, '_requires_redshift', True):
            # Set a placeholder - will be inherited from composite model if needed
            self.redshift = None

        # Validate parameters 
        self._validate()
        
        # Calculate the wavelength sampling for the model
        self.wavelengths = create_default_wavelength_grid()

        # Do expensive preprocessing
        self._preprocess(self.wavelengths)

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get all parameters for this model (both free and fixed).
        Returns Parameter objects for free parameters and values for fixed parameters.
        """
        return self.parameter_manager.get_all_model_parameters(self)

    @abstractmethod
    def _validate(self):
        """
        Validate input parameters, complain if the user has specified anything incorrectly. 
        """
        pass
            
        
    @abstractmethod
    def _preprocess(self, wavelengths: jnp.ndarray):
        """
        Perform any preprocessing steps that can be done before the likelihood call. 
        Store results in self.preprocessed_data
        """
        pass
    
    @abstractmethod
    def _evaluate(self, input_spectrum: jnp.ndarray, **param_values) -> jnp.ndarray:
        """
        Evaluate the model SED given parameter values.
        
        Args:
            **param_values: Dictionary of parameter names to values
            
        Returns:
            Spectral energy distribution
        """
        pass

    def _update_wavelengths(self, wavelengths: Union[jnp.ndarray, u.Quantity]):
        """
        Update the wavelengths used by the model.
        Re-runs any preprocessing steps.
        
        Args:
            wavelengths: New wavelength array to use for evaluation (with or without units)
        """
        # Convert wavelengths to internal units (Angstroms)
        self.wavelengths = sanitize_wavelength_array(wavelengths)
        self._preprocess(self.wavelengths)
    
    def evaluate(self, input_spectrum: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Evaluate the model using parameters from the parameter manager.
        """
        if input_spectrum is None:
            input_spectrum = jnp.zeros_like(self.wavelengths)  # Default to zero spectrum if not provided

        # Get parameter values from manager
        param_values = self.parameter_manager.get_model_parameters(self)
        
        result = self._evaluate(input_spectrum, param_values)
        self.sed = result
        return result
    
    @property
    def evaluated(self) -> bool:
        """
        Check if the model has been evaluated.
        
        Returns:
            True if the model has a valid SED, False otherwise
        """
        return self.sed is not None and len(self.sed) > 0

    @property 
    def wav_obs(self) -> jnp.ndarray:
        """
        Get the observed wavelength array.
        
        Returns:
            Wavelengths in the observed frame
        """
        return self.wavelengths * (1 + self.redshift)
    
    @property
    def wav_rest(self) -> jnp.ndarray:
        """
        Get the rest-frame wavelength array.
        
        Returns:
            Wavelengths in the rest frame (Angstroms)
        """
        return self.wavelengths
    
    def get_sed_with_units(self, flux_type: str = 'fnu', distance: u.Quantity = None) -> u.Quantity:
        """
        Get the model SED converted to observational units.
        
        Parameters
        ----------
        flux_type : str
            Type of flux to return ('fnu' or 'flam')
        distance : astropy.units.Quantity, optional
            Luminosity distance. If None, uses redshift and cosmology.
            
        Returns
        -------
        astropy.units.Quantity
            SED in requested flux units
        """
        if self.sed is None:
            raise ValueError("Model must be evaluated before getting SED with units")
        
        if not hasattr(self, 'redshift') or self.redshift is None:
            raise ValueError("Model redshift must be set to convert to observed flux")
        
        # Convert internal SED (L_sun/Å) to observed flux
        flux_array = SEDUnits.to_observed_flux(
            self.sed, self.wavelengths, self.redshift, distance, flux_type
        )
        
        if flux_type == 'fnu':
            return flux_array * u.uJy
        elif flux_type == 'flam':
            return flux_array * (u.erg / u.s / u.cm**2 / u.angstrom)
        else:
            raise ValueError(f"Unknown flux_type: {flux_type}")
    
    def get_wavelengths_with_units(self, frame: str = 'observed') -> u.Quantity:
        """
        Get wavelength array with units.
        
        Parameters
        ----------
        frame : str
            Reference frame ('observed' or 'rest')
            
        Returns
        -------
        astropy.units.Quantity
            Wavelength array with units
        """
        if frame == 'rest':
            return self.wavelengths * u.angstrom
        elif frame == 'observed':
            if not hasattr(self, 'redshift') or self.redshift is None:
                raise ValueError("Model redshift must be set to get observed-frame wavelengths")
            return self.wav_obs * u.angstrom
        else:
            raise ValueError(f"Unknown frame: {frame}. Must be 'observed' or 'rest'")
    
    def get_numpyro_model(self, obs) -> Callable:
        """
        Create a NumPyro model function for MCMC sampling.
        
        Args:
            obs: Observation object with .flux, .wavelength, .uncertainty
            
        Returns:
            NumPyro model function
        """
        parameter_manager = self.parameter_manager

        # Register observation parameters if any
        for param_name, param in obs.parameters.items():
            if isinstance(param, Parameter):
                parameter_manager.add_parameter(obs, param_name, param)
        
        def numpyro_model(model, obs=None):
            # Sample all parameters
            parameter_manager.sample_all_parameters()

            # Evaluate the model with the sampled parameters
            model.evaluate() 

            # Save deterministic properties of the model, to be sampled with the posterior
            for prop in model.deterministic_properties:
                numpyro.deterministic(prop, getattr(model, prop))

            # Compute the predicted observations from the model
            predicted_obs = obs.predict(model)

            for phot in predicted_obs._phot_list:
                numpyro.sample(phot.name, phot.likelihood_distribution, obs=phot.flux)
            
            for spec in predicted_obs._spec_list:
                numpyro.sample(spec.name, spec.likelihood_distribution, obs=spec.flux)
                    
        return numpyro_model
    
    def __add__(self, other: Union['Model', 'CompositeModel']) -> 'CompositeModel':
        """
        Enable addition for parallel composition: model1 + model2
        Both models are applied to the same input spectrum and results are summed.
        """
        if isinstance(other, (Model, CompositeModel)):
            return CompositeModel([self, other], composition_type='parallel')
        else:
            raise TypeError("Can only add Model or CompositeModel instances")

    def __call__(self, input_model: Union['Model', 'CompositeModel']) -> 'CompositeModel':
        """
        Enable sequential composition: model2(model1)
        model1 is applied first, then model2 processes the result.
        """
        if isinstance(input_model, (Model, CompositeModel)):
            return CompositeModel([input_model, self], composition_type='sequential')
        else:
            raise TypeError("Can only call with Model or CompositeModel instances")
        

class CompositeModel(Model):
    """
    A model composed of multiple sub-models.
    
    Supports two types of composition:
    1. Sequential: model2(model1) - model1 output feeds into model2 input
    2. Parallel: model1 + model2 - both models process same input, outputs are summed
    
    Parameters
    ----------
    models : list of Model
        List of models to compose
    composition_type : str
        Either 'sequential' or 'parallel'
    """
    
    def __init__(self, models: list, composition_type: str = 'sequential'):
        if composition_type not in ['sequential', 'parallel']:
            raise ValueError("composition_type must be 'sequential' or 'parallel'")
            
        self.models = models
        self.composition_type = composition_type
        
        # Create unified parameter manager
        self.parameter_manager = ParameterManager()
        
        # Initialize base class properties
        self.obs = None
        self.sed = None
        self.preprocessed_data = {}
        redshift_found = False
        
        # Register all sub-models and collect parameters
        for model in self.models:
            namespace = self.parameter_manager.register_model(model)
            
            # Replace each model's parameter manager with our unified one
            model.parameter_manager = self.parameter_manager

            if 'redshift' in model.parameters:
                redshift_found = True
        
        # CompositeModel doesn't register itself - it's just a container for sub-models
        
        # Set redshift requirement based on sub-models
        self._requires_redshift = redshift_found
        
        # Inherit redshift from the first model that has one
        inherited_redshift = None
        for model in self.models:
            if hasattr(model, 'redshift') and model.redshift is not None:
                inherited_redshift = model.redshift
                break
        
        # Apply inherited redshift to all models that don't have one
        if inherited_redshift is not None:
            for model in self.models:
                if not hasattr(model, 'redshift') or model.redshift is None:
                    model.redshift = inherited_redshift
                    # Add redshift to parameter manager if it's a Parameter
                    if isinstance(inherited_redshift, Parameter):
                        # Remove any existing fixed redshift value first
                        model_id = str(id(model))
                        if (model_id in self.parameter_manager._model_fixed_values and 
                            'redshift' in self.parameter_manager._model_fixed_values[model_id]):
                            del self.parameter_manager._model_fixed_values[model_id]['redshift']
                        
                        self.parameter_manager.add_parameter(model, 'redshift', inherited_redshift)
        
        # Set the inherited redshift on the composite model
        if inherited_redshift is not None:
            self.redshift = inherited_redshift
        
        # Initialize wavelengths from the first model that has them
        self.wavelengths = None
        for model in self.models:
            if hasattr(model, 'wavelengths') and model.wavelengths is not None:
                self.wavelengths = model.wavelengths
                break
        
        # If no model has wavelengths, create default grid
        if self.wavelengths is None:
            self.wavelengths = create_default_wavelength_grid()
        
        # Validate the composite model
        self._validate()
    
    @property
    def registry(self):
        """Backward compatibility - return the parameter manager"""
        return self.parameter_manager
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get all parameters from all sub-models in the composite.
        Uses namespaced names to avoid conflicts between models.
        """
        all_params = {}
        
        # Get parameters using namespaced names from the parameter manager
        for model in self.models:
            model_id = str(id(model))
            if model_id in self.parameter_manager._namespaces:
                namespace = self.parameter_manager._namespaces[model_id]
                
                # Add free parameters with namespaced names
                if model_id in self.parameter_manager._model_parameters:
                    for param_name, global_name in self.parameter_manager._model_parameters[model_id].items():
                        if global_name in self.parameter_manager._free_parameters:
                            all_params[global_name] = self.parameter_manager._free_parameters[global_name]
                
                # Add fixed parameters with namespaced names
                if model_id in self.parameter_manager._model_fixed_values:
                    for param_name, value in self.parameter_manager._model_fixed_values[model_id].items():
                        namespaced_name = f"{namespace}_{param_name}"
                        all_params[namespaced_name] = value
                        
        return all_params
    
    def evaluate(self, input_spectrum: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Override evaluate for CompositeModel - doesn't need to get its own parameters.
        """
        if input_spectrum is None:
            input_spectrum = jnp.zeros_like(self.wavelengths)
        
        # CompositeModel doesn't have its own parameters, so just call _evaluate directly
        result = self._evaluate(input_spectrum, {})
        self.sed = result
        return result
    
    def _validate(self):
        """
        Validate all sub-models and composition logic.
        """
        if len(self.models) == 0:
            raise ValueError("CompositeModel must contain at least one model")
            
        # Validate all sub-models
        for i, model in enumerate(self.models):
            try:
                model._validate()
            except Exception as e:
                raise ValueError(f"Validation failed for model {i}: {e}")
        
        # For sequential composition, ensure models are compatible
        if self.composition_type == 'sequential':
            # Could add checks here for wavelength compatibility, etc.
            pass
            
    def _preprocess(self, wavelengths: jnp.ndarray):
        """
        Preprocess all sub-models with the given wavelength grid.
        """
        self.wavelengths = wavelengths
        
        # Preprocess all sub-models
        for model in self.models:
            model._update_wavelengths(wavelengths)

    def _evaluate(self, input_spectrum: jnp.ndarray, param_values: dict) -> jnp.ndarray:
        """
        Evaluate the composite model based on composition type.
        param_values is ignored since each model gets its params from parameter_manager.
        
        Parameters
        ----------
        input_spectrum : jnp.ndarray
            Input spectrum
        param_values : dict
            Ignored - each model gets parameters from parameter_manager
            
        Returns
        -------
        jnp.ndarray
            Output spectrum
        """
        if self.composition_type == 'sequential':
            # Apply models in sequence: output of model[i] -> input of model[i+1]
            spectrum = input_spectrum
            for model in self.models:
                # Call _evaluate directly on sub-models to avoid the CompositeModel's evaluate() method
                model_params = self.parameter_manager.get_model_parameters(model)
                spectrum = model._evaluate(spectrum, model_params)
            return spectrum
            
        elif self.composition_type == 'parallel':
            # Apply all models to the same input and sum the results
            output_spectrum = jnp.zeros_like(input_spectrum)
            for model in self.models:
                # Call _evaluate directly on sub-models to avoid the CompositeModel's evaluate() method
                model_params = self.parameter_manager.get_model_parameters(model)
                model_output = model._evaluate(input_spectrum, model_params)
                output_spectrum = output_spectrum + model_output
            return output_spectrum
    
    def __add__(self, other: Union['Model', 'CompositeModel']) -> 'CompositeModel':
        """Add another model in parallel composition."""
        if isinstance(other, CompositeModel) and other.composition_type == 'parallel':
            # Flatten parallel compositions
            new_models = self.models + other.models
        else:
            new_models = self.models + [other]
            
        return CompositeModel(new_models, composition_type='parallel')
    
    def __call__(self, input_model: Union['Model', 'CompositeModel']) -> 'CompositeModel':
        """Create sequential composition with input_model first."""
        if isinstance(input_model, CompositeModel) and input_model.composition_type == 'sequential':
            # Flatten sequential compositions
            new_models = input_model.models + self.models
        else:
            new_models = [input_model] + self.models
            
        return CompositeModel(new_models, composition_type='sequential')
    
    def __repr__(self):
        model_names = [model.__class__.__name__ for model in self.models]
        if self.composition_type == 'sequential':
            connector = ' → '
        else:
            connector = ' + '
        return f"CompositeModel({connector.join(model_names)}, {self.composition_type})"