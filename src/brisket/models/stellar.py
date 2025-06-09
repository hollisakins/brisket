import jax.numpy as jnp
from typing import Optional, Union
from .base import Model, Parameter


class StellarModel(Model):
    """
    Stellar population synthesis model.
    """
    
    def __init__(self, 
            redshift: float | Parameter = None, 
            log_stellar_mass: float | Parameter = None, 
            age: float | Parameter = None,
            metallicity: float | Parameter = None,
            stellar_library: str = 'bc03',
        ):
        
        self.stellar_library = stellar_library
        
        super().__init__(log_stellar_mass=log_stellar_mass, age=age, metallicity=metallicity, redshift=redshift)
    
    def _validate(self):
        """Validate input parameters"""

        # Check that log_stellar_mass is provided in any form
        has_log_mass = (
            'log_stellar_mass' in self.parameters or 
            'log_stellar_mass' in self.registry.parameters or
            hasattr(self, 'log_stellar_mass')
        )
        if not has_log_mass:
            raise ValueError("log_stellar_mass parameter is required")
        
        if not isinstance(self.stellar_library, str):
            raise ValueError("stellar_library must be a string specifying the library name")


    def _preprocess(self, wavelengths: jnp.ndarray):
        """Load stellar population synthesis templates"""
        print(f"Loading {self.stellar_library} stellar templates...")
        
        # Load age and metallicity grids (expensive!)
        self.age_grid = jnp.logspace(-1, 1.1, 50)  # 0.1 to 13 Gyr
        self.metallicity_grid = jnp.linspace(-2.0, 0.5, 20)  # [Z/H]
        
        # Mock stellar templates: shape (n_age, n_metallicity, n_wavelength)
        n_ages, n_metals, n_wave = len(self.age_grid), len(self.metallicity_grid), len(self.wavelengths)
        
        # Create mock templates with realistic stellar SED shapes
        wave_microns = self.wavelengths / 10000.0
        templates = []
        for i, age in enumerate(self.age_grid):
            age_templates = []
            for j, metal in enumerate(self.metallicity_grid):
                # Mock Planck function with age/metallicity dependence
                T_eff = 5000 * (1 + metal * 0.1) * (age / 5.0)**(-0.1)  # Mock temperature
                template = self._planck_function(wave_microns, T_eff) * (age / 1.0)**(-0.3)
                age_templates.append(template)
            templates.append(jnp.array(age_templates))
        
        self.stellar_templates = jnp.array(templates)
        print("Stellar templates loaded!")
    
    def _planck_function(self, wavelength_microns, temperature):
        """Mock Planck function for stellar SEDs"""
        h = 6.626e-34  # Planck constant
        c = 3e8        # Speed of light
        k = 1.381e-23  # Boltzmann constant
        
        wave_m = wavelength_microns * 1e-6
        exponential = jnp.exp(h * c / (wave_m * k * temperature))
        intensity = (2 * h * c**2 / wave_m**5) / (exponential - 1)
        return intensity * 1e-15  # Normalize
    
    def _evaluate(self, spectrum: jnp.ndarray, param_values: dict) -> jnp.ndarray:
        """
        Generate stellar spectrum. 
        """
        log_stellar_mass = param_values['log_stellar_mass']
        age = param_values['age']
        metallicity = param_values['metallicity']
        
        # Convert to linear mass
        stellar_mass = 10**log_stellar_mass  # Solar masses
        
        # Interpolate stellar template
        stellar_spectrum = self._interpolate_stellar_template(age, metallicity)
        
        # Scale by stellar mass
        return stellar_spectrum * stellar_mass
    
    def _interpolate_stellar_template(self, age, metallicity):
        """Fast interpolation of stellar templates"""
        # Simple bilinear interpolation (in practice, might use more sophisticated methods)
        age_idx = jnp.searchsorted(self.age_grid, age) - 1
        age_idx = jnp.clip(age_idx, 0, len(self.age_grid) - 2)
        
        metal_idx = jnp.searchsorted(self.metallicity_grid, metallicity) - 1
        metal_idx = jnp.clip(metal_idx, 0, len(self.metallicity_grid) - 2)
        
        # Get surrounding templates
        template_00 = self.stellar_templates[age_idx, metal_idx]
        template_01 = self.stellar_templates[age_idx, metal_idx + 1]
        template_10 = self.stellar_templates[age_idx + 1, metal_idx]
        template_11 = self.stellar_templates[age_idx + 1, metal_idx + 1]
        
        # Interpolation weights
        age_weight = (age - self.age_grid[age_idx]) / (self.age_grid[age_idx + 1] - self.age_grid[age_idx])
        metal_weight = (metallicity - self.metallicity_grid[metal_idx]) / (
            self.metallicity_grid[metal_idx + 1] - self.metallicity_grid[metal_idx])
        
        # Bilinear interpolation
        template_0 = template_00 * (1 - metal_weight) + template_01 * metal_weight
        template_1 = template_10 * (1 - metal_weight) + template_11 * metal_weight
        
        return template_0 * (1 - age_weight) + template_1 * age_weight
    
    # def get_initial_spectrum(self, param_values):
    #     """Get the initial stellar spectrum (used by CompositeModel)"""
    #     return self._evaluate(None, **param_values)

