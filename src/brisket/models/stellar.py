'''
Stellar models. Combines SFH and CEH models from sfzh.py to create a stellar model.
'''
from __future__ import annotations

import numpy as np
import os
import h5py
from copy import copy
import astropy.units as u

from ..config import grid_dir
from ..utils.misc import make_bins
from ..utils.console import setup_logger
from ..utils import exceptions
from ..parameters import Params
# from ..grid import Grid

from .sfzh import SFZHModel
from .base import *


class BaseStellarModel(EmitterModel):

    name = 'stars'

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        self.logger = setup_logger(__name__, self.verbose)
        self.params = self.validate(kwargs)
    
    def validate(self, kwargs):
        """
        Validate/parse parameters for the stellar model.
        """
        params = super().validate(kwargs)

        # Check if grid is specified
        if 'grid' not in kwargs:
            if 'grids' in kwargs:
                raise exceptions.MisspelledParameter("Parameter 'grids' not understood. Did you mean 'grid'?")
            else:
                raise exceptions.MissingParameter("Parameter 'grid' not specified, cannot create stellar model.")
        
        params['grid'] = kwargs['grid']
        del kwargs['grid']

        if 'logMstar' not in kwargs:
            alt_mass_keys = ['mass', 'massformed', 'logmass', 'stellar_mass', 'mstar', 'Mstar']
            if any(key in kwargs for key in alt_mass_keys):
                k = next(key for key in alt_mass_keys if key in kwargs)
                raise exceptions.MisspelledParameter(f"Parameter '{k}' not understood. Did you mean 'logMstar'?")
            else:
                raise exceptions.MissingParameter("Parameter 'logMstar' not specified, cannot create stellar model.")
        params['logMstar'] = kwargs['logMstar']
        del kwargs['logMstar']
        
        if 'sfh' not in kwargs:
            raise exceptions.MissingParameter("Parameter 'sfh' not specified, cannot create stellar model.")
        self.sfh = kwargs['sfh']
        del kwargs['sfh']
        if any([k.startswith('sfh/') for k in self.sfh.params.all_params]):
            raise exceptions.UnimplementedFunctionality("Cannot use the same SFH instance more than once. Did you forget to initialize a second?.")
        params += self.sfh.params.withprefix('sfh')

        if 'zh' not in kwargs:
            raise exceptions.MissingParameter("Parameter 'zh' not specified, cannot create stellar model.")
        self.zh = kwargs['zh']
        del kwargs['zh']
        if any([k.startswith('zh/') for k in self.zh.params.all_params]):
            raise exceptions.UnimplementedFunctionality("Cannot use the same ZH instance more than once. Did you forget to initialize a second?.")
        params += self.zh.params.withprefix('zh')

        for kwarg in kwargs:
            self.logger.warning(f"Ignoring unexpected parameter '{kwarg}'.")

        return params

    # def add_sfh(self, **kwargs):
    #     if self.has_sfh or len(kwargs) > 1:
    #         raise exceptions.InconsistentParameter('Only one SFH can be added.')

    #     name = list(kwargs)[0]
    #     sfh = kwargs[name]
    
    #     if name in self.params:
    #         raise exceptions.InconsistentParameter(f'SFH with name {name} already exists')

    #     self.sfh = sfh
    #     self.sfh.name = name
    #     self.params.add_child(name, params=sfh.params)
    #     self.params[name].model = sfh # note to self, this is just so the params obj knows the model name

        # if name in ['continuity', 'bursty_continuity']:
        #     if not 'n_bins' in kwargs:
        #         raise exceptions.MissingParameter('n_bins must be specified for continuity and bursty_continuity SFHs')
        #     sfh['n_bins'] = kwargs['n_bins']

        #     if 'bin_edges' in kwargs:
        #         sfh['bin_edges'] = kwargs['bin_edges']

        #     if 'z_max' in kwargs:
        #         sfh['z_max'] = kwargs['z_max']

        #     df = 2.0
        #     if name == 'continuity': scale = 0.3
        #     if name == 'bursty_continuity': scale = 1.0
        #     for i in range(sfh['n_bins']):
        #         sfh[f'dsfr{i}'] = priors.StudentsT(low=-10, high=10, loc=0, scale=scale, df=df)

        # if name == 'constant':
        #     if 'age_min' in kwargs:
        #         sfh['age_min'] = kwargs['age_min']
        #     else:
        #         raise exceptions.MissingParameter("Parameter 'age_min' must be specified for ConstantSFH")
        #     if 'age_max' in kwargs:
        #         sfh['age_max'] = kwargs['age_max'] 
        #     else:
        #         raise exceptions.MissingParameter("Parameter 'age_max' must be specified for ConstantSFH")

        # if name in sfh_models:
        #     self.sfh_model = sfh_models[name]
        # else:
        #     raise exceptions.InconsistentParameter(f'SFH model {name} not recognized')

        return sfh

    # def add_zh(self, **kwargs):
    #     if self.has_zh or len(kwargs) > 1:
    #         raise exceptions.InconsistentParameter('Only one ZH can be added.')

    #     name = list(kwargs)[0]
    #     zh = kwargs[name]
    
    #     if name in self.params:
    #         raise exceptions.InconsistentParameter(f'ZH with name {name} already exists')

    #     self.zh = zh
    #     self.zh.name = name
    #     self.params.add_child(name, params=zh.params)
    #     self.params[name].model = zh

    @property
    def has_sfh(self):
        return hasattr(self, 'sfh')

    @property
    def has_zh(self):
        return hasattr(self, 'zh')

    def __repr__(self):
        return f'BaseStellarModel()'
    
    def __str__(self):
        return self.__repr__()

    def _load_grid(self):
        self.grid = Grid(self.params['grid'], grid_dir)
        self.grid.age_bins = np.power(10., make_bins(self.grid.log10age, fix_low=-99))
        self.grid.age_widths = self.grid.age_bins[1:] - self.grid.age_bins[:-1]

    def _resample_grid_wavelengths(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        if not hasattr(self, 'grid'):
            self._load_grid()
            
        self.wavelengths = wavelengths
        self.grid.interp_spectra(self.wavelengths*angstrom)

    def _assign_sfh_ages(self):
        """ Assigns grid ages to the SFH model. """
        if not hasattr(self, 'grid'):
            self._load_grid()
        
        self.sfh._assign_ages(self.grid.log10age)

    def _assign_zh_metallicities(self):
        """ Assigns grid metallicities to the ZH model. """
        if not hasattr(self, 'grid'):
            self._load_grid()
        
        self.zh._assign_metallicities(self.grid.metallicity)

    def prepare(self, wavelengths):
        """
        Prototype for child defined "prepare" methods.
        In general, should call _load_grid, _resample_grid_wavelengths and/or _assign_sfh_ages. 
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent. "
            "How did you get here!?"
        )

    def get_sed(self, redshift, params):
        """
        Prototype for child defined get_sed methods.
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent. "
            "How did you get here!?"
        )


# class BimodalCompositeStellarPopModel(BaseStellarModel):

class CompositeStellarPopulationModel(BaseStellarModel):
    '''
    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''


    def __init__(self, verbose=False, **kwargs):
        BaseStellarModel.__init__(self, verbose=verbose, **kwargs)
        self.params = self.validate(kwargs)

    def validate(self, kwargs):
        params = BaseStellarModel.validate(self, kwargs)
        return params


    def prepare(self, wavelengths):
        self._load_grid()
        self._resample_grid_wavelengths(wavelengths)
        self._assign_sfh_ages()
        self._assign_zh_metallicities()

    def get_sed(self, redshift, params):
        """Compute the SED for a given star-formation and chemical enrichment history.
        """

        log10age = np.array(self.grid.log10age)
        metallicity = np.array(self.grid.metallicity)
        grid_live_frac = np.ones((len(log10age),len(metallicity)))
        
        self.sfzh = SFZHModel(self.sfh, self.zh, grid_live_frac)

        grid_weights = self.sfzh.get_weights(params)
        if np.ndim(grid_weights) == 2: # Not vectorized
            self.grid.collapse(axis=('ages','metallicities'), weights=grid_weights)
        elif np.ndim(grid_weights) == 3: # Vectorized
            self.grid.collapse(axis=(None, 'ages','metallicities'), weights=grid_weights)

        # TODO check if nebular emission is requested, handle additonal parameters if so 

        return self.grid.spectra['incident']

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



# class BC03StellarModel(CompositeStellarPopModel):

#     grid_file: str = 'bc03_miles_{imf}.hdf5'


#     @staticmethod
#     def validate_params(params):
#         """
#         Validate parameters for the BC03 model.
#         """
#         if 'grids' in params:
#             raise exceptions.InconsistentParameter(f'Cant specify grids with {self.__name__}.')
#         else:
#             params['grids'] = 'bc03_miles_chabrier'
        

#         if not 'imf' in params:
#             params['imf'] = 'chabrier'

        
