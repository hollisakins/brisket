'''
Package-level configuration for BRISKET. 

This module handles the loading and validation of configuration parameters for the BRISKET package.
It reads the default configuration from a TOML file and performs consistency checks on the parameters.
It also provides a mechanism for users to override default parameters through environment variables (TODO). 
'''

import toml
import os
from .utils import exceptions
root_dir = os.path.dirname(os.path.abspath(__file__))

config = toml.load(os.path.join(root_dir, 'config.toml'))
'''
Default configuration parameters for BRISKET.
'''

def _get_config_value(key, default_value, value_type=str):
    """Get configuration value from environment variable or use default."""
    env_key = f'BRISKET_{key.upper()}'
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if value_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(env_value)
        elif value_type == float:
            return float(env_value)
        else:
            return env_value
    return default_value

grid_dir = _get_config_value('grid_dir', config['grid_dir'])
'''
Directory containing the grid files for BRISKET. Can be overridden with BRISKET_GRID_DIR environment variable.
'''

# Check if grid_dir is provided as an absolute path or relative
# if not os.path.isabs(grid_dir):
#     grid_dir = os.path.join(root_dir, grid_dir)

# Check if grid_dir exists
# if not os.path.exists(grid_dir):
#     raise exceptions.InconsistentParameter(f'The specified grid_dir "{grid_dir}" does not exist.')

filter_dir = _get_config_value('filter_dir', config['filter_dir'])
'''
Directory containing the filter files for BRISKET. Can be overridden with BRISKET_FILTER_DIR environment variable.
'''

# Check if grid_dir is provided as an absolute path or relative
if not os.path.isabs(filter_dir):
    filter_dir = os.path.join(root_dir, filter_dir)
filter_directory = f'{filter_dir}/filter_directory.toml'

max_redshift = _get_config_value('max_redshift', config['max_redshift'], float)
'''
Maximum redshift for model evaluation. Can be overridden with BRISKET_MAX_REDSHIFT environment variable.
'''

oversample_wavelengths = _get_config_value('oversample_wavelengths', config['oversample_wavelengths'], int)
'''
Oversampling of model wavelength grid. Can be overridden with BRISKET_OVERSAMPLE_WAVELENGTHS environment variable.
'''

resolution_smoothing_kernel_size = _get_config_value('resolution_smoothing_kernel_size', config['resolution_smoothing_kernel_size'], int)
'''
Size of the smoothing kernel for resolution curve. Can be overridden with BRISKET_RESOLUTION_SMOOTHING_KERNEL_SIZE environment variable.
'''

cosmology = _get_config_value('cosmology', config['cosmology'])
'''
Cosmology to use for calculations. Can be overridden with BRISKET_COSMOLOGY environment variable.
'''

import astropy.cosmology 
from .utils.misc import check_spelling_against_list

if cosmology not in astropy.cosmology.available:
    alternative = check_spelling_against_list(cosmology, astropy.cosmology.available)
    if alternative is None:
        raise exceptions.InconsistentParameter(f'Configured cosmology "{cosmology}" not recognized by astropy.cosmology.')
    else:
        raise exceptions.InconsistentParameter(f'Configured cosmology "{cosmology}" not recognized by astropy.cosmology. Did you mean "{alternative}"?')
cosmo = getattr(astropy.cosmology, cosmology)


sfh_age_log_sampling = _get_config_value('sfh_age_log_sampling', config['sfh_age_log_sampling'], float)
'''
Sampling for star formation history ages in log space. Can be overridden with BRISKET_SFH_AGE_LOG_SAMPLING environment variable.
'''



default_resolution = _get_config_value('default_resolution', config['default_resolution'], int)
'''
Default spectral resolution for models. Can be overridden with BRISKET_DEFAULT_RESOLUTION environment variable.
'''
if default_resolution <= 0:
    raise exceptions.InconsistentParameter(f'Default resolution must be positive, got {default_resolution}.')


max_wavelength = _get_config_value('max_wavelength', config['max_wavelength'], float)
'''
Maximum wavelength for model evaluation (in Angstroms). Can be overridden with BRISKET_MAX_WAVELENGTH environment variable.
'''
if max_wavelength <= 0:
    raise exceptions.InconsistentParameter(f'Max wavelength must be positive, got {max_wavelength}.')