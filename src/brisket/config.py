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
print(root_dir)

config = toml.load(os.path.join(root_dir, 'config.toml'))
'''
Default configuration parameters for BRISKET.
'''

grid_dir = config['grid_dir']
'''
Directory containing the grid files for BRISKET. Overrides can be set through environment variables (TODO).
'''
# TODO check if user overwrites grid_dir from environment variables

# Check if grid_dir is provided as an absolute path or relative
if not os.path.isabs(grid_dir):
    grid_dir = os.path.join(root_dir, grid_dir)

# Check if grid_dir exists
if not os.path.exists(grid_dir):
    raise exceptions.InconsistentParameter(f'The specified grid_dir "{grid_dir}" does not exist.')

max_redshfit = config['max_redshift']
# TODO check if user overwrites max_redshift from environment variables


cosmology = config['cosmology']
# TODO check if user overwrites cosmology from environment variables

import astropy.cosmology 
from .utils.misc import check_spelling_against_list

if cosmology not in astropy.cosmology.available:
    alternative = check_spelling_against_list(cosmology, astropy.cosmology.available)
    if alternative is None:
        raise exceptions.InconsistentParameter(f'Configured cosmology "{cosmology}" not recognized by astropy.cosmology.')
    else:
        raise exceptions.InconsistentParameter(f'Configured cosmology "{cosmology}" not recognized by astropy.cosmology. Did you mean "{alternative}"?')
cosmo = getattr(astropy.cosmology, cosmology)


sfh_age_log_sampling = config['sfh_age_log_sampling']