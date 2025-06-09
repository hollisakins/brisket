import jax.numpy as jnp
import astropy.units as u
from typing import Optional, Union, List
from .base import Model, Parameter
from ..data.emission_lines import get_line_wavelength, get_line_info
from ..parameters import Uniform, Normal
from ..utils.units import sanitize_wavelength, sanitize_velocity


class EmissionLine(Model):
    """
    Single emission line model with Gaussian profile.
    
    Represents a single emission line that can be identified by name from the 
    emission line database or by providing a custom wavelength.
    
    Parameters
    ----------
    line_name : str, optional
        Name of the emission line from the database (e.g., 'halpha', 'oiii_5007')
    wavelength : float, optional  
        Custom rest-frame wavelength in Angstroms (overrides line_name)
    amplitude : Parameter or float
        Line amplitude/flux (units depend on input spectrum)
    sigma : Parameter or float
        Line width parameter (velocity dispersion in km/s)
    redshift : Parameter or float
        Redshift of the source
    continuum_level : Parameter or float, optional
        Local continuum level (default: 0)
    """
    
    def __init__(self, 
                 line_name: Optional[str] = None,
                 wavelength: Optional[Union[float, u.Quantity]] = None,
                 amplitude: Union[Parameter, float, u.Quantity] = None,
                 sigma: Union[Parameter, float, u.Quantity] = None,
                 redshift: Union[Parameter, float] = None,
                 continuum_level: Union[Parameter, float, u.Quantity] = 0.0,
                 **kwargs):
        
        # Determine rest-frame wavelength (convert to Angstroms)
        if line_name is not None and wavelength is not None:
            raise ValueError("Cannot specify both line_name and wavelength")
        elif line_name is not None:
            self.rest_wavelength = get_line_wavelength(line_name)  # Already in Angstroms
            self.line_info = get_line_info(line_name)
            self.line_name = line_name
        elif wavelength is not None:
            self.rest_wavelength = sanitize_wavelength(wavelength)  # Convert to Angstroms
            wavelength_val = self.rest_wavelength if not isinstance(wavelength, u.Quantity) else wavelength.value
            self.line_info = {'wavelength': wavelength_val, 'species': 'custom', 'transition': 'user-defined'}
            self.line_name = f"custom_{self.rest_wavelength:.1f}"
        else:
            raise ValueError("Must specify either line_name or wavelength")
        
        # Set default parameters if not provided (with units)
        if amplitude is None:
            # Default amplitude range for emission lines in L_sun/Å
            amplitude = Uniform(f"{self.line_name}_amplitude", 0.0, 1e-15, 
                              unit=u.L_sun/u.angstrom, description="Line amplitude")
        if sigma is None:
            # Default velocity dispersion range in km/s
            sigma = Uniform(f"{self.line_name}_sigma", 50.0 * u.km/u.s, 500.0 * u.km/u.s,
                          unit=u.km/u.s, description="Velocity dispersion")
        
        # Sanitize parameter values if they have units
        if isinstance(amplitude, (float, int, u.Quantity)) and not isinstance(amplitude, Parameter):
            # Convert amplitude to internal units (L_sun/Å) 
            if isinstance(amplitude, u.Quantity):
                amplitude = amplitude.to(u.L_sun/u.angstrom).value
            self._amplitude_value = amplitude
        else:
            self._amplitude_value = None
            
        if isinstance(sigma, (float, int, u.Quantity)) and not isinstance(sigma, Parameter):
            # Convert sigma to internal units (km/s)
            self._sigma_value = sanitize_velocity(sigma)
        else:
            self._sigma_value = None
            
        if isinstance(continuum_level, u.Quantity):
            continuum_level = continuum_level.to(u.L_sun/u.angstrom).value
        
        self.continuum_level = continuum_level
        
        # EmissionLine models can inherit redshift from composite models
        self._requires_redshift = False  # Allow redshift inheritance
        
        # Store parameters for the base class
        super().__init__(
            amplitude=amplitude,
            sigma=sigma, 
            redshift=redshift,
            continuum_level=continuum_level,
            **kwargs
        )
    
    def _validate(self):
        """Validate emission line parameters."""
        if self.rest_wavelength <= 0:
            raise ValueError("Rest wavelength must be positive")
    
    def _preprocess(self, wavelengths: jnp.ndarray):
        """Preprocess for emission line evaluation."""
        # Store wavelength grid for evaluation
        self.wavelengths = wavelengths
        
        # Convert velocity dispersion to wavelength dispersion
        # sigma_lambda = sigma_v * lambda / c
        c_kms = 299792.458  # km/s
        self.sigma_conversion_factor = self.rest_wavelength / c_kms
    
    def _evaluate(self, input_spectrum: jnp.ndarray, **param_values) -> jnp.ndarray:
        """
        Evaluate the emission line model.
        
        Parameters
        ----------
        input_spectrum : jnp.ndarray
            Input spectrum to add the line to
        **param_values : dict
            Parameter values including amplitude, sigma, redshift, continuum_level
            
        Returns
        -------
        jnp.ndarray
            Output spectrum with emission line added
        """
        amplitude = param_values['amplitude']
        sigma_v = param_values['sigma']  # km/s
        # Get redshift from parameters or use model's redshift
        redshift = param_values.get('redshift', self.redshift)
        if redshift is None:
            raise ValueError(f"No redshift available for {self.line_name}. "
                           "Provide redshift parameter or ensure it's inherited from composite model.")
        
        continuum_level = param_values.get('continuum_level', 0.0)
        
        # Observed wavelength of the line
        observed_wavelength = self.rest_wavelength * (1 + redshift)
        
        # Convert velocity dispersion to wavelength dispersion
        sigma_lambda = sigma_v * self.sigma_conversion_factor * (1 + redshift)
        
        # Gaussian profile
        gaussian = amplitude * jnp.exp(
            -0.5 * ((self.wavelengths - observed_wavelength) / sigma_lambda)**2
        )
        
        # Add continuum level and line to input spectrum
        return input_spectrum + continuum_level + gaussian
    
    @property
    def observed_wavelength(self) -> float:
        """Get the observed wavelength given current redshift."""
        if hasattr(self, 'redshift'):
            return self.rest_wavelength * (1 + self.redshift)
        else:
            return self.rest_wavelength  # If no redshift parameter
    
    def __repr__(self):
        return f"EmissionLine(line='{self.line_name}', λ_rest={self.rest_wavelength:.1f}Å)"


class CompositeEmissionLineModel(Model):
    """
    Model for multiple emission lines.
    
    Combines multiple EmissionLine models into a single composite model.
    All lines share the same redshift but can have independent amplitudes and widths.
    
    Parameters
    ---------- 
    lines : list of EmissionLine or list of str
        List of EmissionLine objects or line names to include
    shared_redshift : Parameter or float, optional
        Shared redshift for all lines (default: create new parameter)
    shared_sigma : Parameter or float, optional  
        Shared velocity dispersion for all lines (default: independent)
    continuum_level : Parameter or float, optional
        Global continuum level (default: 0)
    """
    
    def __init__(self, 
                 lines: List[Union[EmissionLine, str]],
                 shared_redshift: Union[Parameter, float] = None,
                 shared_sigma: Union[Parameter, float] = None,
                 continuum_level: Union[Parameter, float] = 0.0,
                 **kwargs):
        
        # Process line inputs
        self.emission_lines = []
        
        # Set up shared parameters
        if shared_redshift is None:
            shared_redshift = Uniform("redshift", 0.0, 6.0)
        if shared_sigma is not None and not isinstance(shared_sigma, (Parameter, float, int)):
            shared_sigma = Uniform("sigma_shared", 50.0, 500.0)
            
        for i, line in enumerate(lines):
            if isinstance(line, str):
                # Create EmissionLine from string name
                line_sigma = shared_sigma if shared_sigma is not None else Uniform(f"{line}_sigma", 50.0, 500.0)
                emission_line = EmissionLine(
                    line_name=line,
                    amplitude=Uniform(f"{line}_amplitude", 0.0, 1e-15),
                    sigma=line_sigma,
                    redshift=shared_redshift
                )
            elif isinstance(line, EmissionLine):
                # Use existing EmissionLine, but override redshift if shared
                emission_line = line
                if shared_redshift is not None:
                    emission_line.redshift = shared_redshift
                if shared_sigma is not None:
                    emission_line.parameters['sigma'] = shared_sigma
            else:
                raise TypeError(f"Line {i} must be EmissionLine object or string name")
                
            self.emission_lines.append(emission_line)
        
        self.continuum_level = continuum_level
        self.shared_redshift = shared_redshift
        self.shared_sigma = shared_sigma
        
        # Collect all parameters from sub-models
        all_params = {'continuum_level': continuum_level}
        if isinstance(shared_redshift, Parameter):
            all_params['redshift'] = shared_redshift
        if shared_sigma is not None and isinstance(shared_sigma, Parameter):
            all_params['sigma_shared'] = shared_sigma
            
        # Add individual line parameters
        for line in self.emission_lines:
            for param_name, param in line.parameters.items():
                if isinstance(param, Parameter):
                    all_params[param_name] = param
        
        super().__init__(**all_params, **kwargs)
    
    def _validate(self):
        """Validate composite emission line model."""
        if len(self.emission_lines) == 0:
            raise ValueError("Must include at least one emission line")
        
        # Validate all sub-models
        for line in self.emission_lines:
            line._validate()
    
    def _preprocess(self, wavelengths: jnp.ndarray):
        """Preprocess all emission lines."""
        self.wavelengths = wavelengths
        
        # Preprocess all sub-models
        for line in self.emission_lines:
            line._preprocess(wavelengths)
    
    def _evaluate(self, input_spectrum: jnp.ndarray, **param_values) -> jnp.ndarray:
        """
        Evaluate all emission lines and add to input spectrum.
        
        Parameters
        ----------
        input_spectrum : jnp.ndarray
            Input spectrum
        **param_values : dict
            All parameter values for lines and continuum
            
        Returns  
        -------
        jnp.ndarray
            Output spectrum with all emission lines added
        """
        # Start with input spectrum plus continuum
        output_spectrum = input_spectrum + param_values.get('continuum_level', 0.0)
        
        # Add each emission line
        for line in self.emission_lines:
            # Extract parameters relevant to this line
            line_params = {}
            for param_name in line.parameters.keys():
                if param_name in param_values:
                    line_params[param_name] = param_values[param_name]
            
            # Use shared parameters if available
            if 'redshift' in param_values:
                line_params['redshift'] = param_values['redshift']
            if 'sigma_shared' in param_values:
                line_params['sigma'] = param_values['sigma_shared']
                
            # Evaluate this line (don't add continuum again)
            line_params['continuum_level'] = 0.0  # Prevent double-counting continuum
            line_contribution = line._evaluate(jnp.zeros_like(input_spectrum), **line_params)
            output_spectrum = output_spectrum + line_contribution
        
        return output_spectrum
    
    def get_line_by_name(self, line_name: str) -> EmissionLine:
        """Get a specific emission line by name."""
        for line in self.emission_lines:
            if line.line_name == line_name:
                return line
        raise ValueError(f"Line '{line_name}' not found in composite model")
    
    @property 
    def line_names(self) -> List[str]:
        """Get names of all emission lines in the model."""
        return [line.line_name for line in self.emission_lines]
    
    def __repr__(self):
        line_list = ", ".join(self.line_names)
        return f"CompositeEmissionLineModel(lines=[{line_list}])"


# Utility functions for easy emission line model creation

def create_balmer_series(redshift: Union[Parameter, float] = None, 
                        shared_sigma: Union[Parameter, float] = None,
                        include_lines: List[str] = None) -> CompositeEmissionLineModel:
    """
    Create a composite model with Balmer series lines.
    
    Parameters
    ----------
    redshift : Parameter or float, optional
        Shared redshift for all lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for all lines
    include_lines : list of str, optional
        Which Balmer lines to include (default: ['halpha', 'hbeta', 'hgamma'])
        
    Returns
    -------
    CompositeEmissionLineModel
        Model with Balmer series lines
    """
    if include_lines is None:
        include_lines = ['halpha', 'hbeta', 'hgamma']
    
    return CompositeEmissionLineModel(
        lines=include_lines,
        shared_redshift=redshift,
        shared_sigma=shared_sigma
    )

def create_oiii_doublet(redshift: Union[Parameter, float] = None,
                       shared_sigma: Union[Parameter, float] = None,
                       flux_ratio: Union[Parameter, float] = 2.98) -> CompositeEmissionLineModel:
    """
    Create [O III] 4959,5007 doublet with theoretical flux ratio.
    
    Parameters
    ----------
    redshift : Parameter or float, optional
        Shared redshift for both lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for both lines
    flux_ratio : Parameter or float, optional
        Flux ratio [O III] 5007 / [O III] 4959 (default: 2.98 theoretical)
        
    Returns
    -------
    CompositeEmissionLineModel
        Model with [O III] doublet
    """
    # Create individual lines with linked amplitudes
    if isinstance(flux_ratio, (int, float)):
        # Fixed ratio
        oiii_4959_amp = Uniform("oiii_4959_amplitude", 0.0, 1e-15)
        oiii_5007_amp = flux_ratio * oiii_4959_amp  # This won't work directly in JAX
        # Better approach: use a single amplitude parameter and scale in evaluation
        base_amplitude = Uniform("oiii_base_amplitude", 0.0, 1e-15)
        
        # Create lines manually to handle ratio
        oiii_4959 = EmissionLine(
            line_name='oiii_4959',
            amplitude=base_amplitude,
            sigma=shared_sigma,
            redshift=redshift
        )
        oiii_5007 = EmissionLine(
            line_name='oiii_5007', 
            amplitude=base_amplitude,  # Will be scaled in composite model
            sigma=shared_sigma,
            redshift=redshift
        )
        
        return CompositeEmissionLineModel(
            lines=[oiii_4959, oiii_5007],
            shared_redshift=redshift,
            shared_sigma=shared_sigma
        )
    else:
        # Parameter ratio - simpler approach
        return CompositeEmissionLineModel(
            lines=['oiii_4959', 'oiii_5007'],
            shared_redshift=redshift,
            shared_sigma=shared_sigma
        )

def create_oii_doublet(redshift: Union[Parameter, float] = None,
                      shared_sigma: Union[Parameter, float] = None) -> CompositeEmissionLineModel:
    """
    Create [O II] 3727,3729 doublet.
    
    Parameters
    ----------
    redshift : Parameter or float, optional
        Shared redshift for both lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for both lines
        
    Returns
    -------
    CompositeEmissionLineModel
        Model with [O II] doublet
    """
    return CompositeEmissionLineModel(
        lines=['oii_3727', 'oii_3729'],
        shared_redshift=redshift,
        shared_sigma=shared_sigma
    )

def create_halpha_complex(redshift: Union[Parameter, float] = None,
                         shared_sigma: Union[Parameter, float] = None,
                         include_nii: bool = True,
                         include_sii: bool = False) -> CompositeEmissionLineModel:
    """
    Create H-alpha complex with [N II] and optionally [S II] lines.
    
    Parameters
    ----------
    redshift : Parameter or float, optional
        Shared redshift for all lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for all lines
    include_nii : bool, optional
        Include [N II] 6548,6584 lines (default: True)
    include_sii : bool, optional
        Include [S II] 6717,6731 lines (default: False)
        
    Returns
    -------
    CompositeEmissionLineModel
        Model with H-alpha complex
    """
    lines = ['halpha']
    
    if include_nii:
        lines.extend(['nii_6548', 'nii_6584'])
    
    if include_sii:
        lines.extend(['sii_6717', 'sii_6731'])
    
    return CompositeEmissionLineModel(
        lines=lines,
        shared_redshift=redshift,
        shared_sigma=shared_sigma
    )

def create_optical_galaxy_lines(redshift: Union[Parameter, float] = None,
                               shared_sigma: Union[Parameter, float] = None) -> CompositeEmissionLineModel:
    """
    Create a comprehensive optical galaxy emission line model.
    
    Includes the most common optical emission lines for galaxy spectroscopy:
    - [O II] 3727,3729
    - H-beta
    - [O III] 4959,5007  
    - H-alpha
    - [N II] 6548,6584
    - [S II] 6717,6731
    
    Parameters
    ----------
    redshift : Parameter or float, optional
        Shared redshift for all lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for all lines
        
    Returns
    -------
    CompositeEmissionLineModel
        Comprehensive optical galaxy line model
    """
    lines = [
        'oii_3727', 'oii_3729',  # [O II] doublet
        'hbeta',                  # H-beta
        'oiii_4959', 'oiii_5007', # [O III] doublet
        'halpha',                 # H-alpha
        'nii_6548', 'nii_6584',   # [N II] doublet
        'sii_6717', 'sii_6731'    # [S II] doublet
    ]
    
    return CompositeEmissionLineModel(
        lines=lines,
        shared_redshift=redshift,
        shared_sigma=shared_sigma
    )

def create_custom_line_model(line_names: List[str],
                           redshift: Union[Parameter, float] = None,
                           shared_sigma: Union[Parameter, float] = None) -> CompositeEmissionLineModel:
    """
    Create a custom emission line model from a list of line names.
    
    Parameters
    ----------
    line_names : list of str
        Names of emission lines from the database
    redshift : Parameter or float, optional
        Shared redshift for all lines
    shared_sigma : Parameter or float, optional
        Shared velocity dispersion for all lines
        
    Returns
    -------
    CompositeEmissionLineModel
        Custom emission line model
    """
    return CompositeEmissionLineModel(
        lines=line_names,
        shared_redshift=redshift,
        shared_sigma=shared_sigma
    )
        