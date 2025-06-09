import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import astropy.units as u
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Callable
from collections import defaultdict

class Parameter:
    """
    Represents a model parameter that will be sampled during MCMC.
    
    This class stores the parameter specification (name and prior distribution)
    and gets resolved to actual values within the NumPyro model context.
    Supports units for better user experience and automatic conversion.
    """
    
    def __init__(self, name: str, prior: dist.Distribution, 
                 transform: Optional[Callable] = None,
                 unit: Optional[u.Unit] = None,
                 description: Optional[str] = None):
        self.name = name
        self.prior = prior
        self.transform = transform  # Optional transformation (e.g., log -> linear)
        self.unit = unit  # Physical unit for this parameter
        self.description = description  # Human-readable description
        self._value = None  # Will be set during NumPyro sampling
        
    def sample(self):
        """Sample this parameter within NumPyro context"""
        sampled_value = numpyro.sample(self.name, self.prior)
        if self.transform is not None:
            sampled_value = self.transform(sampled_value)
        self._value = sampled_value
        return sampled_value
    
    @property
    def value(self):
        """Get the current parameter value (only valid during model evaluation)"""
        if self._value is None:
            raise ValueError(f"Parameter {self.name} has not been sampled yet")
        return self._value
    
    def __repr__(self):
        return f"Parameter(name='{self.name}', prior={self.prior.__class__.__name__})"


# Convenience functions for common parameter types
def Uniform(name: str, low: Union[float, u.Quantity], high: Union[float, u.Quantity], 
           unit: Optional[u.Unit] = None, description: Optional[str] = None) -> Parameter:
    """
    Create a uniform parameter with optional units.
    
    Parameters
    ----------
    name : str
        Parameter name
    low : float or astropy.units.Quantity
        Lower bound
    high : float or astropy.units.Quantity  
        Upper bound
    unit : astropy.units.Unit, optional
        Expected unit for this parameter
    description : str, optional
        Human-readable description
        
    Returns
    -------
    Parameter
        Uniform parameter
    """
    from .utils.units import sanitize_quantity
    
    # Handle units
    if isinstance(low, u.Quantity) or isinstance(high, u.Quantity):
        if unit is None and isinstance(low, u.Quantity):
            unit = low.unit
        elif unit is None and isinstance(high, u.Quantity):
            unit = high.unit
            
        if unit is not None:
            low_val = sanitize_quantity(low, unit, allow_dimensionless=True)
            high_val = sanitize_quantity(high, unit, allow_dimensionless=True)
        else:
            low_val = float(low) if not isinstance(low, u.Quantity) else low.value
            high_val = float(high) if not isinstance(high, u.Quantity) else high.value
    else:
        low_val, high_val = float(low), float(high)
    
    return Parameter(name, dist.Uniform(low_val, high_val), unit=unit, description=description)

def Normal(name: str, loc: Union[float, u.Quantity], scale: Union[float, u.Quantity],
          unit: Optional[u.Unit] = None, description: Optional[str] = None) -> Parameter:
    """
    Create a normal parameter with optional units.
    
    Parameters
    ----------
    name : str
        Parameter name
    loc : float or astropy.units.Quantity
        Mean value
    scale : float or astropy.units.Quantity
        Standard deviation
    unit : astropy.units.Unit, optional
        Expected unit for this parameter
    description : str, optional
        Human-readable description
        
    Returns
    -------
    Parameter
        Normal parameter
    """
    from .utils.units import sanitize_quantity
    
    # Handle units
    if isinstance(loc, u.Quantity) or isinstance(scale, u.Quantity):
        if unit is None and isinstance(loc, u.Quantity):
            unit = loc.unit
        elif unit is None and isinstance(scale, u.Quantity):
            unit = scale.unit
            
        if unit is not None:
            loc_val = sanitize_quantity(loc, unit, allow_dimensionless=True)
            scale_val = sanitize_quantity(scale, unit, allow_dimensionless=True)
        else:
            loc_val = float(loc) if not isinstance(loc, u.Quantity) else loc.value
            scale_val = float(scale) if not isinstance(scale, u.Quantity) else scale.value
    else:
        loc_val, scale_val = float(loc), float(scale)
    
    return Parameter(name, dist.Normal(loc_val, scale_val), unit=unit, description=description)

def LogUniform(name: str, low: Union[float, u.Quantity], high: Union[float, u.Quantity],
              unit: Optional[u.Unit] = None, description: Optional[str] = None) -> Parameter:
    """
    Create a log-uniform parameter with optional units.
    
    Parameters
    ----------
    name : str
        Parameter name
    low : float or astropy.units.Quantity
        Lower bound (will be log-transformed)
    high : float or astropy.units.Quantity
        Upper bound (will be log-transformed)
    unit : astropy.units.Unit, optional
        Expected unit for this parameter
    description : str, optional
        Human-readable description
        
    Returns
    -------
    Parameter
        Log-uniform parameter
    """
    from .utils.units import sanitize_quantity
    
    # Handle units - convert to log space
    if isinstance(low, u.Quantity) or isinstance(high, u.Quantity):
        if unit is None and isinstance(low, u.Quantity):
            unit = low.unit
        elif unit is None and isinstance(high, u.Quantity):
            unit = high.unit
            
        if unit is not None:
            low_val = sanitize_quantity(low, unit, allow_dimensionless=True)
            high_val = sanitize_quantity(high, unit, allow_dimensionless=True)
        else:
            low_val = float(low) if not isinstance(low, u.Quantity) else low.value
            high_val = float(high) if not isinstance(high, u.Quantity) else high.value
    else:
        low_val, high_val = float(low), float(high)
    
    # Transform to log space for the prior
    log_low = jnp.log(low_val)
    log_high = jnp.log(high_val)
    
    return Parameter(name, dist.Uniform(log_low, log_high), transform=jnp.exp, 
                    unit=unit, description=description)

def LogNormal(name: str, loc: Union[float, u.Quantity], scale: Union[float, u.Quantity],
             unit: Optional[u.Unit] = None, description: Optional[str] = None) -> Parameter:
    """
    Create a log-normal parameter with optional units.
    
    Parameters
    ----------
    name : str
        Parameter name
    loc : float or astropy.units.Quantity
        Log-scale location parameter
    scale : float or astropy.units.Quantity
        Log-scale scale parameter
    unit : astropy.units.Unit, optional
        Expected unit for this parameter
    description : str, optional
        Human-readable description
        
    Returns
    -------
    Parameter
        Log-normal parameter
    """
    from .utils.units import sanitize_quantity
    
    # Handle units
    if isinstance(loc, u.Quantity) or isinstance(scale, u.Quantity):
        if unit is None and isinstance(loc, u.Quantity):
            unit = loc.unit
        elif unit is None and isinstance(scale, u.Quantity):
            unit = scale.unit
            
        if unit is not None:
            loc_val = sanitize_quantity(loc, unit, allow_dimensionless=True)
            scale_val = sanitize_quantity(scale, unit, allow_dimensionless=True)
        else:
            loc_val = float(loc) if not isinstance(loc, u.Quantity) else loc.value
            scale_val = float(scale) if not isinstance(scale, u.Quantity) else scale.value
    else:
        loc_val, scale_val = float(loc), float(scale)
    
    return Parameter(name, dist.LogNormal(loc_val, scale_val), unit=unit, description=description)


class ParameterRegistry:
    """
    Global registry to track all parameters in a composite model.
    This ensures each parameter is only sampled once, even if used by multiple components.
    """
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.sampled_values: Dict[str, Any] = {}
    
    def register(self, parameter: Parameter):
        """Register a parameter"""
        if parameter.name in self.parameters:
            # Check if it's the same parameter or a conflict
            existing = self.parameters[parameter.name]
            if existing.prior != parameter.prior:
                raise ValueError(f"Parameter name conflict: {parameter.name}")
        self.parameters[parameter.name] = parameter
    
    def sample_all(self):
        """Sample all registered parameters within NumPyro context"""
        self.sampled_values = {}
        for name, param in self.parameters.items():
            value = param.sample()
            self.sampled_values[name] = value
        return self.sampled_values
    
    def clear(self):
        """Clear the registry (useful for multiple model evaluations)"""
        self.parameters.clear()
        self.sampled_values.clear()
        for param in self.parameters.values():
            param._value = None

    @property
    def sampled(self) -> bool:
        """Check if any parameters have been sampled"""
        return len(self.sampled_values) > 0

    def __len__(self):
        """Number of registered parameters"""
        return len(self.parameters)
    
    def __contains__(self, name: str) -> bool:
        """Check if a parameter is registered"""
        return name in self.parameters

    def __repr__(self):
        """String representation of the registry"""
        return f"ParameterRegistry({self.parameters})"


class ParameterManager:
    """
    Advanced parameter management system for composite models.
    
    Handles parameter registration, namespace management, and sampling
    for complex model hierarchies. Ensures each unique parameter is only
    sampled once while allowing parameter sharing between models.
    """
    
    def __init__(self):
        self._unique_parameters: Dict[str, Parameter] = {}
        self._model_parameters: Dict[str, Dict[str, str]] = {}
        self._namespaces: Dict[str, str] = {}
        self._sampled_values: Dict[str, Any] = {}
        
    def register_model(self, model, namespace: str = None) -> str:
        """
        Register a model and its parameters with optional namespace.
        
        Parameters
        ----------
        model : Model
            Model instance to register
        namespace : str, optional
            Namespace prefix for parameters. If None, uses class name.
            
        Returns
        -------
        str
            The namespace used for this model
        """
        if namespace is None:
            namespace = model.__class__.__name__
            
        # Make namespace unique if needed
        original_namespace = namespace
        counter = 1
        while namespace in self._namespaces.values():
            namespace = f"{original_namespace}_{counter}"
            counter += 1
            
        model_id = str(id(model))
        self._namespaces[model_id] = namespace
        self._model_parameters[model_id] = {}
        
        # Register the model's parameters
        if hasattr(model, 'parameters'):
            for param_name, param in model.parameters.items():
                if isinstance(param, Parameter):
                    self.add_parameter(model, param_name, param)
                    
        return namespace
        
    def add_parameter(self, model, param_name: str, parameter: Parameter):
        """
        Add a parameter for a specific model.
        
        Parameters
        ----------
        model : Model
            Model that owns this parameter
        param_name : str
            Name of the parameter within the model
        parameter : Parameter
            Parameter object
        """
        model_id = str(id(model))
        
        if model_id not in self._model_parameters:
            # Auto-register the model if not already registered
            self.register_model(model)
            
        namespace = self._namespaces[model_id]
        global_name = f"{namespace}_{param_name}"
        
        # Check if this exact parameter already exists globally
        existing_global_param = None
        for existing_name, existing_param in self._unique_parameters.items():
            if (existing_param.name == parameter.name and 
                existing_param.prior == parameter.prior):
                existing_global_param = existing_name
                break
                
        if existing_global_param is not None:
            # Reuse existing parameter
            self._model_parameters[model_id][param_name] = existing_global_param
        else:
            # Create new global parameter
            self._unique_parameters[global_name] = parameter
            self._model_parameters[model_id][param_name] = global_name
            
    def get_model_parameters(self, model) -> Dict[str, Any]:
        """
        Get parameter values for a specific model.
        
        Parameters
        ----------
        model : Model
            Model to get parameters for
            
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to values for this model
        """
        model_id = str(id(model))
        if model_id not in self._model_parameters:
            return {}
            
        model_params = {}
        for param_name, global_name in self._model_parameters[model_id].items():
            if global_name in self._sampled_values:
                model_params[param_name] = self._sampled_values[global_name]
                
        return model_params
        
    def sample_all_parameters(self):
        """Sample all unique parameters within NumPyro context."""
        self._sampled_values = {}
        for global_name, parameter in self._unique_parameters.items():
            # Use the global name for NumPyro sampling
            sampled_value = numpyro.sample(global_name, parameter.prior)
            if parameter.transform is not None:
                sampled_value = parameter.transform(sampled_value)
            self._sampled_values[global_name] = sampled_value
            
    def get_global_parameter_dict(self) -> Dict[str, Parameter]:
        """
        Get a flattened dictionary of all parameters.
        
        Returns
        -------
        Dict[str, Parameter]
            All parameters with their global names
        """
        return dict(self._unique_parameters)
        
    def clear(self):
        """Clear all registered parameters and sampled values."""
        self._unique_parameters.clear()
        self._model_parameters.clear()
        self._namespaces.clear()
        self._sampled_values.clear()
        
    def __len__(self) -> int:
        """Number of unique parameters."""
        return len(self._unique_parameters)
        
    def __repr__(self) -> str:
        """String representation of the parameter manager."""
        n_models = len(self._model_parameters)
        n_params = len(self._unique_parameters)
        return f"ParameterManager(models={n_models}, unique_params={n_params})"