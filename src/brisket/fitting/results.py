"""
Results container for brisket MCMC fits.

This module provides the FitResults class which wraps NumPyro MCMC results
and provides convenient access to posterior samples, diagnostics, and summaries.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC
from typing import Dict, Any, Optional, Union
import warnings
from collections import defaultdict

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    warnings.warn("ArviZ not available. Some diagnostic features will be limited.")


class FitResults:
    """
    Container for MCMC fit results with convenient access to samples and diagnostics.
    
    This class wraps NumPyro MCMC results and provides methods for extracting
    posterior samples, computing summaries, and running diagnostics.
    
    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        The MCMC object after running sampling
    model : Model
        The model that was fit
    obs : Observation
        The observational data that was fit
    """
    
    def __init__(self, mcmc: MCMC, model, obs):
        self.mcmc = mcmc
        self.model = model
        self.obs = obs
        self._samples = None
        self._summary_cache = None
        
    @property
    def samples(self) -> Dict[str, jnp.ndarray]:
        """
        Get posterior samples.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to sample arrays
        """
        if self._samples is None:
            self._samples = self.mcmc.get_samples()
        return self._samples
    
    @property 
    def num_samples(self) -> int:
        """Number of posterior samples."""
        if len(self.samples) == 0:
            return 0
        first_param = next(iter(self.samples.values()))
        return first_param.shape[0]
    
    @property
    def num_chains(self) -> int:
        """Number of chains used in sampling."""
        return self.mcmc.num_chains
    
    @property
    def parameter_names(self) -> list:
        """List of parameter names."""
        return list(self.samples.keys())
    
    def get_samples(self, param_name: str) -> jnp.ndarray:
        """
        Get samples for a specific parameter.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
            
        Returns
        -------
        jnp.ndarray
            Sample array for the parameter
        """
        if param_name not in self.samples:
            raise KeyError(f"Parameter '{param_name}' not found in samples")
        return self.samples[param_name]
    
    def summary(self, percentiles: list = [5, 25, 50, 75, 95]) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all parameters.
        
        Parameters
        ----------
        percentiles : list
            Percentiles to compute for each parameter
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with parameter names as keys and
            statistics (mean, std, percentiles) as values
        """
        if self._summary_cache is not None:
            return self._summary_cache
            
        summary = {}
        for param_name, samples in self.samples.items():
            param_summary = {
                'mean': float(jnp.mean(samples)),
                'std': float(jnp.std(samples)),
            }
            
            # Add percentiles
            for p in percentiles:
                param_summary[f'p{p}'] = float(jnp.percentile(samples, p))
                
            summary[param_name] = param_summary
            
        self._summary_cache = summary
        return summary
    
    def get_parameter_summary(self, param_name: str) -> Dict[str, float]:
        """
        Get summary statistics for a single parameter.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
            
        Returns
        -------
        Dict[str, float]
            Summary statistics for the parameter
        """
        full_summary = self.summary()
        if param_name not in full_summary:
            raise KeyError(f"Parameter '{param_name}' not found")
        return full_summary[param_name]
    
    def print_summary(self, parameters: Optional[list] = None):
        """
        Print a formatted summary of results.
        
        Parameters
        ----------
        parameters : list, optional
            List of parameter names to include. If None, includes all parameters.
        """
        summary = self.summary()
        
        if parameters is None:
            parameters = self.parameter_names
            
        print(f"MCMC Results Summary")
        print(f"{'='*50}")
        print(f"Samples: {self.num_samples}, Chains: {self.num_chains}")
        print(f"{'='*50}")
        print(f"{'Parameter':<15} {'Mean':<10} {'Std':<10} {'5%':<8} {'95%':<8}")
        print(f"{'-'*50}")
        
        for param in parameters:
            if param in summary:
                s = summary[param]
                print(f"{param:<15} {s['mean']:<10.3f} {s['std']:<10.3f} "
                      f"{s['p5']:<8.3f} {s['p95']:<8.3f}")
    
    def get_effective_sample_size(self) -> Dict[str, float]:
        """
        Compute effective sample size for all parameters.
        
        Returns
        -------
        Dict[str, float]
            Effective sample sizes
        """
        if not HAS_ARVIZ:
            warnings.warn("ArviZ required for effective sample size calculation")
            return {}
            
        try:
            # Convert to ArviZ InferenceData format
            idata = az.from_numpyro(self.mcmc)
            ess = az.ess(idata)
            
            # Convert back to dictionary
            ess_dict = {}
            for var in ess.data_vars:
                ess_dict[var] = float(ess[var].values)
                
            return ess_dict
        except Exception as e:
            warnings.warn(f"Could not compute effective sample size: {e}")
            return {}
    
    def get_rhat(self) -> Dict[str, float]:
        """
        Compute R-hat convergence diagnostic for all parameters.
        
        Returns
        -------
        Dict[str, float]
            R-hat values (should be close to 1.0 for convergence)
        """
        if not HAS_ARVIZ:
            warnings.warn("ArviZ required for R-hat calculation")
            return {}
            
        if self.num_chains < 2:
            warnings.warn("R-hat requires multiple chains")
            return {}
            
        try:
            # Convert to ArviZ InferenceData format
            idata = az.from_numpyro(self.mcmc)
            rhat = az.rhat(idata)
            
            # Convert back to dictionary
            rhat_dict = {}
            for var in rhat.data_vars:
                rhat_dict[var] = float(rhat[var].values)
                
            return rhat_dict
        except Exception as e:
            warnings.warn(f"Could not compute R-hat: {e}")
            return {}
    
    def check_convergence(self, rhat_threshold: float = 1.1) -> Dict[str, bool]:
        """
        Check convergence for all parameters using R-hat.
        
        Parameters
        ----------
        rhat_threshold : float
            Threshold for R-hat (parameters with R-hat > threshold are not converged)
            
        Returns
        -------
        Dict[str, bool]
            Boolean convergence status for each parameter
        """
        rhat_values = self.get_rhat()
        
        if len(rhat_values) == 0:
            warnings.warn("Could not compute R-hat for convergence check")
            return {}
            
        convergence = {}
        for param, rhat in rhat_values.items():
            convergence[param] = rhat <= rhat_threshold
            
        return convergence
    
    def get_chain_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive chain diagnostics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing various diagnostic statistics
        """
        diagnostics = {
            'num_samples': self.num_samples,
            'num_chains': self.num_chains,
            'effective_sample_size': self.get_effective_sample_size(),
            'rhat': self.get_rhat(),
            'convergence': self.check_convergence()
        }
        
        return diagnostics
    
    def to_arviz(self):
        """
        Convert results to ArviZ InferenceData format.
        
        Returns
        -------
        arviz.InferenceData
            ArviZ InferenceData object for advanced diagnostics and plotting
        """
        if not HAS_ARVIZ:
            raise ImportError("ArviZ is required for this functionality")
            
        return az.from_numpyro(self.mcmc)
    
    def save(self, filename: str):
        """
        Save results to file.
        
        Parameters
        ----------
        filename : str
            Output filename (supports .nc for NetCDF via ArviZ)
        """
        if filename.endswith('.nc'):
            if not HAS_ARVIZ:
                raise ImportError("ArviZ is required to save as NetCDF")
            idata = self.to_arviz()
            idata.to_netcdf(filename)
        else:
            # Save as numpy arrays
            np.savez(filename, **self.samples)
    
    def __repr__(self) -> str:
        """String representation of the results."""
        return (f"FitResults(n_samples={self.num_samples}, "
                f"n_chains={self.num_chains}, "
                f"n_parameters={len(self.parameter_names)})")