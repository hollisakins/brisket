'''
Stellar models. Combines SFH and CEH models from sfzh.py to create a stellar model.
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
from ..grids.grids import Grid
from .base import *
from ..console import setup_logger

class BaseStellarModel:

    expected_params = ['grid', 'logMstar', 'zmet']

    def __init__(self, params, verbose=False):
        self.params = params

        self.verbose = verbose
        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')
    
    @staticmethod
    def validate_params(params):
        """
        Validate parameters for the stellar model.
        
        This static method should be called before initializing the model, 
        i.e., from within the Params validation method. 
        """
        
        # Validate each component of the model parameters inline
        params = BaseStellarModel._validate_grid(params)
        params = BaseStellarModel._validate_mstar(params)
        params = BaseStellarModel._validate_zmet(params)
        params = BaseStellarModel._validate_sfh(params)
        return params


    @staticmethod
    def _validate_grid(params):
        # Check if grid is specified
        if 'grid' not in params:
            if 'grids' in params:
                raise exceptions.MisspelledParameter("Parameter 'grids' not understood. Did you mean 'grid'?")
            else:
                raise exceptions.MissingParameter("Parameter 'grid' not specified, cannot create stellar model.")
        
        # Make sure the grid is provided as a file name only
        elif params['grid'].endswith('.hdf5') or params['grid'].endswith('h5'):
            raise exceptions.InconsistentParameter("Parameter 'grid' expects the grid name, without any extension.")
        elif '/' in params['grid']:
            raise exceptions.InconsistentParameter("Parameter 'grid' expects the grid name, not the full path.")

        # Check if the grid exists, for this we have a handy static method in Grid
        Grid.assert_exists(params['grid'])
        return params

    @staticmethod
    def _validate_mstar(params):
        if 'logMstar' not in params:
            alt_mass_keys = ['mass', 'massformed', 'logmass', 'stellar_mass', 'mstar', 'Mstar']
            if any(key in params for key in alt_mass_keys):
                k = next(key for key in alt_mass_keys if key in params)
                raise exceptions.MisspelledParameter(f"Parameter '{k}' not understood. Did you mean 'logMstar'?")
            else:
                raise exceptions.MissingParameter("Parameter 'logMstar' not specified, cannot create stellar model.")
        return params

    @staticmethod
    def _validate_zmet(params):
        if 'zmet' not in params:
            alt_zmet_keys = ['metallicity', 'Z', 'Zmet']
            if any(key in params for key in alt_zmet_keys):
                k = next(key for key in alt_zmet_keys if key in params)
                raise exceptions.MisspelledParameter(f"Parameter '{k}' not understood. Did you mean 'zmet'?")
            else:
                raise exceptions.MissingParameter("Parameter 'zmet' not specified, cannot create stellar model.")
        return params
        
    @staticmethod
    def _validate_sfh(params):
        pass


        for key in params.all_param_names:
            if '/' not in key and key not in expected_params:
                self.logger.warning(f"Ignoring unexpected parameter '{key}'.")
                del params[key]

        return params

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

class CompositeStellarPopulationModel(BaseStellarModel):
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

class SimpleStellarPopulationModel(BaseStellarModel):
    '''
    A simple stellar population model. 
    
    Interpolates over a given stellar grid to the 
    specified age and metallicity.

    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''

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

    grid_file: str = 'bc03_miles_{imf}.hdf5'


    @staticmethod
    def validate_params(params):
        """
        Validate parameters for the BC03 model.
        """
        if 'grids' in params:
            raise exceptions.InconsistentParameter(f'Cant specify grids with {self.__name__}.')
        else:
            params['grids'] = 'bc03_miles_chabrier'
        

        if not 'imf' in params:
            params['imf'] = 'chabrier'

        
