'''
Stellar models. Combined SFH and CEH models from sfzh.py to create a stellar model.
'''
from __future__ import annotations

import numpy as np
import os
import h5py
from copy import copy
import astropy.units as u

from .. import config
from ..utils import utils
from ..utils.sed import SED
# from ..data.grid_manager import GridManager
from ..grids.grids import Grid
from .base import *
from ..console import setup_logger

class BaseStellarModel:

    def __init__(self, params, verbose=False):
        self.params = params

        self.verbose = verbose
        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        self.validate_params(params)
    
    def validate_params(self, params):
        expected_params = ['grid', 'logMstar', 'zmet', 't_bc']
        if 'grid' not in params:
            if 'grids' in params:
                self.logger.info("Parameter 'grids', should be 'grid', fixing.")
                params['grid'] = params['grids']; del params['grids']
            else:
                raise exceptions.InconsistentParameter('No stellar grid specified.')
        elif str(params['grid']).endswith('.hdf5') or '/' in str(params['grid']):
            raise exceptions.InconsistentParameter("Parameter 'grid' expects the grid name, not the full path.")
        assert Grid.check_if_exists(params['grid']), f"Grid '{params['grid']}' not found."

        if 'logMstar' not in params:
            alt_mass_keys = ['mass', 'massformed', 'logmass', 'stellar_mass', 'mstar', 'Mstar']
            if any(key in params for key in alt_mass_keys):
                k = next(key for key in alt_mass_keys if key in params)
                self.logger.info(f"Parameter '{k}', should be 'logMstar', fixing.")
                params['logMstar'] = params[k]; del params[k]
            else:
                raise BrisketError("Parameter 'logMstar' not specified, cannot create stellar model.")
        
        if 'zmet' not in params:
            alt_zmet_keys = ['metallicity', 'Z', 'Zmet']
            if any(key in params for key in alt_zmet_keys):
                k = next(key for key in alt_zmet_keys if key in params)
                self.logger.info(f"Parameter '{k}', should be 'zmet', fixing.")
                params['zmet'] = params[k]; del params[k]
            else:
                raise BrisketError("Parameter 'zmet' not specified, cannot create stellar model.")
        
        if 't_bc' not in params:
            self.logger.info("Parameter 't_bc' not specified, setting to 10 Myr.")
            params['t_bc'] = 0.01
        
        for key in params.all_param_names:
            if '/' not in key and key not in expected_params:
                self.logger.warning(f"Ignoring unexpected parameter '{key}'.")
                del params[key]

    def validate_components(self, params):
        '''Validate that the SFH components were added correctly.'''

        self.sfh_components = {}
        for comp_name, comp in self.params.components.items():
            if comp.model.type == 'sfh':
                self.sfh_components[comp_name] = comp.model

        if len(self.sfh_components) == 1:
            self.sfh_weights = [1]
        elif len(self.sfh_components) == 0:
            raise BrisketError('No SFH components found.')
        else:
            if 'weight' in comp:
                self.sfh_weights = [float(comp['weight']) for comp in self.params.components.values() if comp.model.type == 'sfh']
            elif 'logweight' in comp:
                self.sfh_weights = [np.power(10., float(comp['logweight'])) for comp in self.params.components.values() if comp.model.type == 'sfh']

    def __repr__(self):
        return f'BaseStellarModel(grid={self.grid.name}, sfh={list(self.sfh_components)})'
    
    def __str__(self):
        return self.__repr__()

    def resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths
        self.grid.resample(self.wavelengths)

    def get_sed(self, params):
        """
        Prototype for child defined get_sed methods.
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            "How did you get here!?"
        )


# class BimodalCompositeStellarPopModel(BaseStellarModel):

class CompositeStellarPopModel(BaseStellarModel):
    '''
    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''

    def __init__(self, params, verbose=False):
        super().__init__(params, verbose)

        # Initialize the SFH and ZH models
        sfh_params = params.get_sfh()
        SFH = params.sfh_model(sfh_params)

        zh_params = params.get_zh()
        ZH = params.zh_model(zh_params)
        
        # Load in the stellar grid
        self.grid = Grid(str(params['grid']))
        self.grid.age[self.grid.age == 0] = 1
        self.grid.age_bins = np.power(10., utils.make_bins(np.log10(self.grid.age), fix_low=-99))
        self.grid.age_widths = self.grid.age_bins[1:] - self.grid.age_bins[:-1]

        self.sfzh = SFZH(SFH, ZH, grid_live_frac)


    def get_sed(self, params):
        """Compute the SED for a given star-formation and chemical enrichment history.
        """
        
        self.grid_weights = self.sfzh.get_weights(params)

        if np.ndim(self.grid_weights) == 2: # not vectorized
            self.grid.collapse(axis=('zmet','age'), weights=self.grid_weights, inplace=True)
        
        return self.grid.data

class SimpleStellarPopModel(BaseGriddedModel, BaseSourceModel):
    '''
    A simple stellar population model, which interpolates from 
    a given stellar grid to the specified age and metallicity (zmet).

    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''
    type = 'source'
    order = 0

    def __init__(self, params):
        self.params = params
        # self._build_defaults(params)
        self.grid = Grid(str(params['grids']))

    def validate_components(self, params):
        pass

    def resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths
        self.grid.resample(self.wavelengths)

    def emit(self, params):
        self.grid.interpolate({'zmet':0, 'age':0}, inplace=True)
        # interpolate live_frac
        # scale to mass: self.grid *= params['massformed']
        
        return self.grid.to_SED()



class BC03StellarModel(CompositeStellarPopModel):
    def validate_params(self, params):
        if 'grids' in params:
            raise BrisketError(f'Cant specify grids with {self.__name__}.')
        else:
            params['grids'] = 'bc03_miles_chabrier'
        super().validate_params(params)


class Starburst25StellarModel(CompositeStellarPopModel):
    def validate_params(self, params):
        if 'grids' in params:
            raise BrisketError(f'Cant specify grids with {self.__name__}.')
        else:
            params['grids'] = 'starburst25'
        super().validate_params(params)