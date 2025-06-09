"""
Emission line database for astronomical spectroscopy.
Contains rest-frame wavelengths and properties of common emission lines.
"""

import jax.numpy as jnp

# Emission line database
# Format: {name: {'wavelength': rest_wavelength_angstrom, 'species': element/ion, 'transition': description}}
EMISSION_LINES = {
    # Hydrogen Balmer series
    'halpha': {
        'wavelength': 6562.8,
        'species': 'H I',
        'transition': 'Balmer alpha (3→2)',
        'aliases': ['ha', 'h_alpha', 'h-alpha']
    },
    'hbeta': {
        'wavelength': 4861.3,
        'species': 'H I', 
        'transition': 'Balmer beta (4→2)',
        'aliases': ['hb', 'h_beta', 'h-beta']
    },
    'hgamma': {
        'wavelength': 4340.5,
        'species': 'H I',
        'transition': 'Balmer gamma (5→2)',
        'aliases': ['hg', 'h_gamma', 'h-gamma']
    },
    'hdelta': {
        'wavelength': 4101.7,
        'species': 'H I',
        'transition': 'Balmer delta (6→2)',
        'aliases': ['hd', 'h_delta', 'h-delta']
    },
    
    # Hydrogen Lyman series
    'lyman_alpha': {
        'wavelength': 1215.7,
        'species': 'H I',
        'transition': 'Lyman alpha (2→1)',
        'aliases': ['lya', 'ly_alpha', 'ly-alpha']
    },
    
    # Hydrogen Paschen series
    'paschen_alpha': {
        'wavelength': 18751.0,
        'species': 'H I',
        'transition': 'Paschen alpha (4→3)',
        'aliases': ['pa', 'pas_alpha']
    },
    
    # Forbidden oxygen lines [O II] and [O III]
    'oii_3727': {
        'wavelength': 3727.1,
        'species': '[O II]',
        'transition': '²D₅/₂ → ⁴S₃/₂',
        'aliases': ['oii_3727', '[oii]_3727']
    },
    'oii_3729': {
        'wavelength': 3729.9,
        'species': '[O II]',
        'transition': '²D₃/₂ → ⁴S₃/₂', 
        'aliases': ['oii_3729', '[oii]_3729']
    },
    'oiii_4959': {
        'wavelength': 4958.9,
        'species': '[O III]',
        'transition': '¹D₂ → ³P₁',
        'aliases': ['oiii_4959', '[oiii]_4959']
    },
    'oiii_5007': {
        'wavelength': 5006.8,
        'species': '[O III]',
        'transition': '¹D₂ → ³P₂',
        'aliases': ['oiii_5007', '[oiii]_5007', 'oiii']
    },
    
    # Forbidden oxygen [O I]
    'oi_6300': {
        'wavelength': 6300.3,
        'species': '[O I]',
        'transition': '¹D₂ → ³P₂',
        'aliases': ['oi_6300', '[oi]_6300']
    },
    'oi_6364': {
        'wavelength': 6363.8,
        'species': '[O I]',
        'transition': '¹D₂ → ³P₁',
        'aliases': ['oi_6364', '[oi]_6364']
    },
    
    # Forbidden nitrogen [N II]
    'nii_6548': {
        'wavelength': 6548.0,
        'species': '[N II]',
        'transition': '¹D₂ → ³P₁',
        'aliases': ['nii_6548', '[nii]_6548']
    },
    'nii_6584': {
        'wavelength': 6583.5,
        'species': '[N II]',
        'transition': '¹D₂ → ³P₂',
        'aliases': ['nii_6584', '[nii]_6584', 'nii']
    },
    
    # Forbidden sulfur [S II]
    'sii_6717': {
        'wavelength': 6716.4,
        'species': '[S II]',
        'transition': '²D₅/₂ → ⁴S₃/₂',
        'aliases': ['sii_6717', '[sii]_6717']
    },
    'sii_6731': {
        'wavelength': 6730.8,
        'species': '[S II]',
        'transition': '²D₃/₂ → ⁴S₃/₂',
        'aliases': ['sii_6731', '[sii]_6731']
    },
    
    # Forbidden sulfur [S III]
    'siii_9069': {
        'wavelength': 9068.6,
        'species': '[S III]',
        'transition': '¹D₂ → ³P₁',
        'aliases': ['siii_9069', '[siii]_9069']
    },
    'siii_9532': {
        'wavelength': 9531.1,
        'species': '[S III]',
        'transition': '¹D₂ → ³P₂',
        'aliases': ['siii_9532', '[siii]_9532']
    },
    
    # He I lines
    'hei_3889': {
        'wavelength': 3888.6,
        'species': 'He I',
        'transition': '²³S → ²³P',
        'aliases': ['hei_3889', 'he_3889']
    },
    'hei_5876': {
        'wavelength': 5875.6,
        'species': 'He I',
        'transition': '²³S → ²³P',
        'aliases': ['hei_5876', 'he_5876']
    },
    
    # He II lines
    'heii_4686': {
        'wavelength': 4685.7,
        'species': 'He II',
        'transition': '4→3',
        'aliases': ['heii_4686', 'he2_4686']
    },
    
    # Carbon lines
    'ciii_1909': {
        'wavelength': 1908.7,
        'species': 'C III]',
        'transition': '²P₃/₂ → ²P₁/₂',
        'aliases': ['ciii_1909', 'c3_1909']
    },
    'civ_1549': {
        'wavelength': 1548.2,
        'species': 'C IV',
        'transition': '²P₃/₂ → ²S₁/₂',
        'aliases': ['civ_1549', 'c4_1549']
    },
    
    # Mg II line
    'mgii_2798': {
        'wavelength': 2798.0,
        'species': 'Mg II',
        'transition': '²P₃/₂ → ²S₁/₂',
        'aliases': ['mgii_2798', 'mg2_2798']
    },
    
    # Common AGN lines
    'broad_halpha': {
        'wavelength': 6562.8,
        'species': 'H I (broad)',
        'transition': 'Balmer alpha (3→2)',
        'aliases': ['broad_ha', 'halpha_broad']
    },
    'broad_hbeta': {
        'wavelength': 4861.3,
        'species': 'H I (broad)',
        'transition': 'Balmer beta (4→2)',
        'aliases': ['broad_hb', 'hbeta_broad']
    },
}

def get_line_wavelength(line_name: str) -> float:
    """
    Get the rest-frame wavelength of an emission line.
    
    Parameters
    ----------
    line_name : str
        Name of the emission line or alias
        
    Returns
    -------
    float
        Rest-frame wavelength in Angstroms
        
    Raises
    ------
    ValueError
        If the line name is not found in the database
    """
    # Direct lookup
    if line_name.lower() in EMISSION_LINES:
        return EMISSION_LINES[line_name.lower()]['wavelength']
    
    # Search through aliases
    for name, properties in EMISSION_LINES.items():
        if line_name.lower() in [alias.lower() for alias in properties.get('aliases', [])]:
            return properties['wavelength']
    
    raise ValueError(f"Emission line '{line_name}' not found in database. "
                     f"Available lines: {list(EMISSION_LINES.keys())}")

def get_line_info(line_name: str) -> dict:
    """
    Get full information about an emission line.
    
    Parameters
    ----------
    line_name : str
        Name of the emission line or alias
        
    Returns
    -------
    dict
        Dictionary containing wavelength, species, transition, and aliases
    """
    # Direct lookup
    if line_name.lower() in EMISSION_LINES:
        return EMISSION_LINES[line_name.lower()]
    
    # Search through aliases
    for name, properties in EMISSION_LINES.items():
        if line_name.lower() in [alias.lower() for alias in properties.get('aliases', [])]:
            return properties
    
    raise ValueError(f"Emission line '{line_name}' not found in database.")

def list_available_lines() -> list:
    """
    Get a list of all available emission line names and aliases.
    
    Returns
    -------
    list
        List of all available line names and aliases
    """
    all_names = list(EMISSION_LINES.keys())
    for properties in EMISSION_LINES.values():
        all_names.extend(properties.get('aliases', []))
    return sorted(all_names)

def get_lines_by_species(species: str) -> dict:
    """
    Get all lines from a specific ion/species.
    
    Parameters
    ----------
    species : str
        Ion/species name (e.g., 'H I', '[O III]', 'He I')
        
    Returns
    -------
    dict
        Dictionary of lines from that species
    """
    return {name: props for name, props in EMISSION_LINES.items() 
            if props['species'] == species}

def get_lines_in_range(min_wavelength: float, max_wavelength: float) -> dict:
    """
    Get all lines within a wavelength range.
    
    Parameters
    ----------
    min_wavelength : float
        Minimum wavelength in Angstroms
    max_wavelength : float
        Maximum wavelength in Angstroms
        
    Returns
    -------
    dict
        Dictionary of lines within the wavelength range
    """
    return {name: props for name, props in EMISSION_LINES.items()
            if min_wavelength <= props['wavelength'] <= max_wavelength}