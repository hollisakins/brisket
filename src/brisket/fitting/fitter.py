"""
Model fitting infrastructure for brisket using NumPyro backend.

This module provides a high-level interface for fitting SED models to observational data
while abstracting away NumPyro implementation details.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
from jax import random
from typing import Dict, Any, Optional, Union, Callable
import warnings
from abc import ABC, abstractmethod

from ..models.base import Model, CompositeModel
from .results import FitResults


class BaseSampler(ABC):
    """Abstract base class for samplers"""
    
    @abstractmethod
    def create_kernel(self, model_fn: Callable, **kwargs):
        """Create sampling kernel"""
        pass
    
    @abstractmethod
    def get_default_mcmc_kwargs(self) -> Dict[str, Any]:
        """Get default MCMC configuration"""
        pass


class NUTSSampler(BaseSampler):
    """No-U-Turn Sampler (NUTS) implementation"""
    
    def __init__(self, step_size: float = None, target_accept_prob: float = 0.8,
                 max_tree_depth: int = 10, **kwargs):
        """
        Initialize NUTS sampler.
        
        Parameters
        ----------
        step_size : float, optional
            Step size for the integrator. If None, will be adapted during warmup.
        target_accept_prob : float
            Target acceptance probability for dual averaging
        max_tree_depth : int
            Maximum tree depth for NUTS
        **kwargs
            Additional arguments passed to NUTS
        """
        self.step_size = step_size
        self.target_accept_prob = target_accept_prob
        self.max_tree_depth = max_tree_depth
        self.kwargs = kwargs
    
    def create_kernel(self, model_fn: Callable, **kwargs):
        """Create NUTS kernel"""
        nuts_kwargs = {
            'target_accept_prob': self.target_accept_prob,
            'max_tree_depth': self.max_tree_depth,
            **self.kwargs
        }
        
        # Only add step_size if it's not None
        if self.step_size is not None:
            nuts_kwargs['step_size'] = self.step_size
        
        return NUTS(model_fn, **nuts_kwargs)
    
    def get_default_mcmc_kwargs(self) -> Dict[str, Any]:
        return {
            'num_warmup': 1000,
            'num_samples': 1000,
            'num_chains': 1,
            'progress_bar': True
        }


class HMCSampler(BaseSampler):
    """Hamiltonian Monte Carlo sampler"""
    
    def __init__(self, step_size: float = 0.01, trajectory_length: float = 1.0, **kwargs):
        """
        Initialize HMC sampler.
        
        Parameters
        ----------
        step_size : float
            Step size for the integrator
        trajectory_length : float
            Length of each trajectory
        **kwargs
            Additional arguments passed to HMC
        """
        self.step_size = step_size
        self.trajectory_length = trajectory_length
        self.kwargs = kwargs
    
    def create_kernel(self, model_fn: Callable, **kwargs):
        """Create HMC kernel"""
        return HMC(
            model_fn,
            step_size=self.step_size,
            trajectory_length=self.trajectory_length,
            **self.kwargs
        )
    
    def get_default_mcmc_kwargs(self) -> Dict[str, Any]:
        return {
            'num_warmup': 1000,
            'num_samples': 1000,
            'num_chains': 1,
            'progress_bar': True
        }


class Fitter:
    """
    High-level interface for fitting models to observations using MCMC.
    
    This class abstracts away NumPyro implementation details and provides
    a simple interface for Bayesian SED fitting.
    
    Examples
    --------
    Basic usage:
    >>> fitter = Fitter(model, obs, sampler='nuts')
    >>> results = fitter.run()
    >>> results.summary()
    
    Advanced configuration:
    >>> fitter = Fitter(model, obs, sampler='nuts', 
    ...                 num_warmup=2000, num_samples=2000,
    ...                 sampler_kwargs={'target_accept_prob': 0.9})
    >>> results = fitter.run(rng_key=42)
    """
    
    def __init__(self, model: Union[Model, CompositeModel], obs,
                 sampler: Union[str, BaseSampler] = 'nuts',
                 num_warmup: int = None,
                 num_samples: int = None, 
                 num_chains: int = None,
                 sampler_kwargs: Dict[str, Any] = None,
                 progress_bar: bool = True):
        """
        Initialize the fitter.
        
        Parameters
        ----------
        model : Model or CompositeModel
            The model to fit
        obs : Observation
            The observational data
        sampler : str or BaseSampler
            Sampler to use ('nuts', 'hmc', or custom sampler instance)
        num_warmup : int, optional
            Number of warmup samples (default from sampler)
        num_samples : int, optional  
            Number of posterior samples (default from sampler)
        num_chains : int, optional
            Number of parallel chains (default: 1)
        sampler_kwargs : dict, optional
            Additional keyword arguments for the sampler
        progress_bar : bool
            Whether to show progress bar during sampling
        """
        self.model = model
        self.obs = obs
        self.progress_bar = progress_bar
        
        # Set up the sampler
        if isinstance(sampler, str):
            sampler_kwargs = sampler_kwargs or {}
            if sampler.lower() == 'nuts':
                self.sampler = NUTSSampler(**sampler_kwargs)
            elif sampler.lower() == 'hmc':
                self.sampler = HMCSampler(**sampler_kwargs)
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Use 'nuts', 'hmc', or provide a BaseSampler instance.")
        elif isinstance(sampler, BaseSampler):
            self.sampler = sampler
        else:
            raise TypeError("sampler must be a string or BaseSampler instance")
        
        # Get default MCMC configuration and override with user values
        mcmc_config = self.sampler.get_default_mcmc_kwargs()
        if num_warmup is not None:
            mcmc_config['num_warmup'] = num_warmup
        if num_samples is not None:
            mcmc_config['num_samples'] = num_samples
        if num_chains is not None:
            mcmc_config['num_chains'] = num_chains
        mcmc_config['progress_bar'] = progress_bar
        
        self.mcmc_config = mcmc_config
        
        # Prepare the model
        self._setup_model()
        
        # Create NumPyro model function
        self.numpyro_model = self.model.get_numpyro_model(self.obs)
        
        # Create sampler kernel
        self.kernel = self.sampler.create_kernel(self.numpyro_model)
        
        # Create MCMC object
        self.mcmc = MCMC(self.kernel, **self.mcmc_config)
    
    def _setup_model(self):
        """Prepare the model for fitting"""
        # Update the model's observation attribute and wavelengths
        # This also recomputes any preprocessing steps in the model
        if self.model.obs is None or self.model.obs is not self.obs:
            self.model.obs = self.obs
            self.model._update_wavelengths(self.obs.model_wavelengths)
        
        # Validate that model has free parameters
        if hasattr(self.model, 'parameter_manager'):
            n_free_params = len(self.model.parameter_manager._free_parameters)
        else:
            # Legacy fallback
            n_free_params = len(getattr(self.model, 'registry', {}))
        
        if n_free_params == 0:
            warnings.warn("Model has no free parameters. All parameters appear to be fixed.")
    
    def run(self, rng_key: Union[int, jnp.ndarray] = None, 
            extra_fields: tuple = ()) -> FitResults:
        """
        Run the MCMC sampling.
        
        Parameters
        ----------
        rng_key : int or jax.random.PRNGKey, optional
            Random number generator key. If int, will be used as seed.
            Default: PRNGKey(0)
        extra_fields : tuple, optional
            Extra fields to collect during sampling (e.g., log likelihood)
            
        Returns
        -------
        FitResults
            Results object containing samples and diagnostics
        """
        # Handle random key
        if rng_key is None:
            rng_key = random.PRNGKey(0)
        elif isinstance(rng_key, int):
            rng_key = random.PRNGKey(rng_key)
        
        # Run MCMC
        self.mcmc.run(rng_key, self.model, obs=self.obs, extra_fields=extra_fields)
        
        return FitResults(self.mcmc, self.model, self.obs)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of fitter configuration.
        
        Returns
        -------
        dict
            Configuration summary including model type, sampler, and MCMC settings
        """
        # Get number of free parameters
        if hasattr(self.model, 'parameter_manager'):
            n_free_params = len(self.model.parameter_manager._free_parameters)
        else:
            n_free_params = 'unknown'
        
        return {
            'model_type': type(self.model).__name__,
            'sampler_type': type(self.sampler).__name__,
            'mcmc_config': self.mcmc_config.copy(),
            'n_free_parameters': n_free_params
        }
    
    def __repr__(self) -> str:
        """String representation of the fitter"""
        config = self.get_config_summary()
        return (f"Fitter(model={config['model_type']}, "
                f"sampler={config['sampler_type']}, "
                f"n_params={config['n_free_parameters']})")