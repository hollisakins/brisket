"""
Unit handling utilities for brisket.

This module provides utilities for converting between different units
and ensuring consistent internal units throughout the package.

Internal Standards:
- Wavelengths: Angstroms
- SEDs: L_sun / Angstrom
- Fluxes: Various (see individual functions)
- Velocity: km/s
- Mass: Solar masses
- Age: Gyr
"""

import numpy as np
import jax.numpy as jnp
import astropy.units as u
from typing import Union, Tuple

# Define internal unit standards
INTERNAL_WAVELENGTH_UNIT = u.angstrom
INTERNAL_SED_UNIT = u.Lsun / u.angstrom  # Luminosity per unit wavelength
INTERNAL_VELOCITY_UNIT = u.km / u.s
INTERNAL_MASS_UNIT = u.Msun
INTERNAL_AGE_UNIT = u.Gyr


def sanitize_quantity(value: Union[float, int, u.Quantity], 
                      target_unit: u.Unit,
                      allow_dimensionless: bool = False) -> float:
    """
    Convert a quantity to the target unit and return the numeric value.
    
    Parameters
    ----------
    value : float, int, or astropy.units.Quantity
        Input value with or without units
    target_unit : astropy.units.Unit
        Target unit for conversion
    allow_dimensionless : bool, optional
        Whether to allow dimensionless quantities (default: False)
        
    Returns
    -------
    float
        Numeric value in target units (JAX-compatible)
        
    Raises
    ------
    ValueError
        If units are incompatible or missing when required
    """
    if isinstance(value, u.Quantity):
        try:
            converted = value.to(target_unit)
            return float(converted.value)
        except u.UnitConversionError:
            raise ValueError(f"Cannot convert {value.unit} to {target_unit}")
    
    elif allow_dimensionless and isinstance(value, (int, float)):
        return float(value)
    
    elif isinstance(value, (int, float)):
        raise ValueError(f"Value {value} has no units. Expected units compatible with {target_unit}")
    
    else:
        raise TypeError(f"Value must be numeric or astropy Quantity, got {type(value)}")


def sanitize_wavelength(wavelength: Union[float, u.Quantity]) -> float:
    """
    Convert wavelength to internal units (Angstroms).
    
    Parameters
    ----------
    wavelength : float or astropy.units.Quantity
        Wavelength value
        
    Returns
    -------
    float
        Wavelength in Angstroms
    """
    if isinstance(wavelength, u.Quantity):
        return sanitize_quantity(wavelength, INTERNAL_WAVELENGTH_UNIT)
    else:
        # Assume Angstroms if no units provided
        return float(wavelength)


def sanitize_wavelength_array(wavelengths: Union[np.ndarray, u.Quantity]) -> jnp.ndarray:
    """
    Convert wavelength array to internal units (Angstroms).
    
    Parameters
    ----------
    wavelengths : array-like or astropy.units.Quantity
        Wavelength array
        
    Returns
    -------
    jax.numpy.ndarray
        Wavelength array in Angstroms
    """
    if isinstance(wavelengths, u.Quantity):
        converted = wavelengths.to(INTERNAL_WAVELENGTH_UNIT).value
        return jnp.asarray(converted)
    else:
        return jnp.asarray(wavelengths)


def sanitize_velocity(velocity: Union[float, u.Quantity]) -> float:
    """
    Convert velocity to internal units (km/s).
    
    Parameters
    ----------
    velocity : float or astropy.units.Quantity
        Velocity value
        
    Returns
    -------
    float
        Velocity in km/s
    """
    return sanitize_quantity(velocity, INTERNAL_VELOCITY_UNIT, allow_dimensionless=True)


def sanitize_mass(mass: Union[float, u.Quantity]) -> float:
    """
    Convert mass to internal units (solar masses).
    
    Parameters
    ----------
    mass : float or astropy.units.Quantity
        Mass value
        
    Returns
    -------
    float
        Mass in solar masses
    """
    return sanitize_quantity(mass, INTERNAL_MASS_UNIT, allow_dimensionless=True)


def sanitize_age(age: Union[float, u.Quantity]) -> float:
    """
    Convert age to internal units (Gyr).
    
    Parameters
    ----------
    age : float or astropy.units.Quantity
        Age value
        
    Returns
    -------
    float
        Age in Gyr
    """
    return sanitize_quantity(age, INTERNAL_AGE_UNIT, allow_dimensionless=True)


def sanitize_flux_density(flux: Union[float, np.ndarray, u.Quantity], 
                         flux_type: str = 'fnu') -> Union[float, jnp.ndarray]:
    """
    Convert flux density to standard units.
    
    Parameters
    ----------
    flux : float, array-like, or astropy.units.Quantity
        Flux density value(s)
    flux_type : str
        Type of flux density ('fnu' or 'flam')
        
    Returns
    -------
    float or jax.numpy.ndarray
        Flux density in standard units:
        - fnu: µJy (microjanskys)
        - flam: erg/s/cm²/Å
    """
    if flux_type == 'fnu':
        target_unit = u.uJy
    elif flux_type == 'flam':
        target_unit = u.erg / u.s / u.cm**2 / u.angstrom
    else:
        raise ValueError(f"Unknown flux_type: {flux_type}. Must be 'fnu' or 'flam'")
    
    if isinstance(flux, u.Quantity):
        converted = flux.to(target_unit).value
        if np.isscalar(converted):
            return float(converted)
        else:
            return jnp.asarray(converted)
    else:
        # Return as-is, assuming correct units
        if np.isscalar(flux):
            return float(flux)
        else:
            return jnp.asarray(flux)


class SEDUnits:
    """
    Class to handle SED unit conversions and manage internal SED unit standard.
    
    All models internally work with SEDs in L_sun/Å. This class provides
    methods to convert to/from observational units.
    """
    
    INTERNAL_UNIT = INTERNAL_SED_UNIT
    
    @staticmethod
    def to_observed_flux(sed_lsun_per_angstrom: jnp.ndarray,
                        wavelength_angstrom: jnp.ndarray,
                        redshift: float,
                        distance: u.Quantity = None,
                        flux_type: str = 'fnu') -> jnp.ndarray:
        """
        Convert internal SED units to observed flux units.
        
        Parameters
        ----------
        sed_lsun_per_angstrom : jax.numpy.ndarray
            SED in internal units (L_sun/Å)
        wavelength_angstrom : jax.numpy.ndarray
            Rest-frame wavelengths in Angstroms
        redshift : float
            Redshift of the source
        distance : astropy.units.Quantity, optional
            Luminosity distance. If None, uses flat ΛCDM cosmology.
        flux_type : str
            Output flux type ('fnu' or 'flam')
            
        Returns
        -------
        jax.numpy.ndarray
            Observed flux in requested units
        """
        # Import cosmology here to avoid circular imports
        from .. import config
        
        # Calculate luminosity distance if not provided
        if distance is None:
            # Use cosmology from config
            d_L = config.cosmo.luminosity_distance(redshift)
        else:
            d_L = distance
            
        # Convert to observed frame
        observed_wavelength = wavelength_angstrom * (1 + redshift)  # Å
        
        # Convert L_sun/Å to flux units
        # F_λ = L_λ / (4π d_L²) × (1+z)  [the (1+z) accounts for time dilation]
        flux_factor = L_sun.to(u.erg/u.s) / (4 * np.pi * d_L.to(u.cm)**2) * (1 + redshift)
        
        # sed_lsun_per_angstrom is already per Angstrom, so result is in erg/s/cm²/Å
        flam = sed_lsun_per_angstrom * flux_factor.value  # erg/s/cm²/Å
        
        if flux_type == 'flam':
            return flam
        elif flux_type == 'fnu':
            # Convert F_λ to F_ν using F_ν = F_λ × λ²/c
            c_angstrom_per_s = (u.c).to(u.angstrom/u.s).value
            fnu_cgs = flam * observed_wavelength**2 / c_angstrom_per_s  # erg/s/cm²/Hz
            # Convert to µJy
            fnu_ujy = fnu_cgs * 1e6 * 1e23  # µJy
            return fnu_ujy
        else:
            raise ValueError(f"Unknown flux_type: {flux_type}")
    
    @staticmethod
    def from_observed_flux(flux: jnp.ndarray,
                          wavelength_angstrom: jnp.ndarray,
                          redshift: float,
                          distance: u.Quantity = None,
                          flux_type: str = 'fnu') -> jnp.ndarray:
        """
        Convert observed flux to internal SED units (L_sun/Å).
        
        Parameters
        ----------
        flux : jax.numpy.ndarray
            Observed flux
        wavelength_angstrom : jax.numpy.ndarray
            Observed-frame wavelengths in Angstroms
        redshift : float
            Redshift of the source
        distance : astropy.units.Quantity, optional
            Luminosity distance
        flux_type : str
            Input flux type ('fnu' or 'flam')
            
        Returns
        -------
        jax.numpy.ndarray
            SED in internal units (L_sun/Å)
        """
        # This is the inverse of to_observed_flux
        # Implementation would convert flux back to luminosity
        raise NotImplementedError("Conversion from observed flux to SED not yet implemented")


def get_default_units() -> dict:
    """
    Get a dictionary of default units for common quantities.
    
    Returns
    -------
    dict
        Dictionary mapping quantity names to default units
    """
    return {
        'wavelength': INTERNAL_WAVELENGTH_UNIT,
        'sed': INTERNAL_SED_UNIT,
        'velocity': INTERNAL_VELOCITY_UNIT,
        'mass': INTERNAL_MASS_UNIT,
        'age': INTERNAL_AGE_UNIT,
        'flux_fnu': u.uJy,
        'flux_flam': u.erg / u.s / u.cm**2 / u.angstrom,
        'redshift': u.dimensionless_unscaled
    }


def validate_units_compatible(value: u.Quantity, expected_unit: u.Unit) -> bool:
    """
    Check if a quantity's units are compatible with expected units.
    
    Parameters
    ----------
    value : astropy.units.Quantity
        Quantity to check
    expected_unit : astropy.units.Unit
        Expected unit
        
    Returns
    -------
    bool
        True if units are compatible
    """
    try:
        value.to(expected_unit)
        return True
    except u.UnitConversionError:
        return False