'''
Stellar models. Combines SFH and CEH models from sfzh.py to create a stellar model.
'''
from __future__ import annotations

import numpy as np
import os, sys
import h5py
from copy import copy
import astropy.units as u

from ..utils.misc import check_spelling_against_list
from ..utils.console import setup_logger
from ..utils import exceptions
from ..parameters import Params
from .linelist import linelist

class SpectralLineModel:

    def __init__(self, verbose=False, **kwargs):
        
        self.verbose = verbose
        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')
        
        self.params = self.validate(kwargs)
    
    def validate(self, kwargs):
        """
        Validate and parse parameters for the spectral line model.
        """
        params = Params()

        self.lines = {}
        
        for name,value in kwargs.items():
            
            # Check for global line width parameter
            if name == 'fwhm':
                params[name] = value
            # if 'fwhm' in name:
            #
            elif 'fwhm' in name or 'dv' in name:
                pass

            # Check if its a line flux
            elif name.startswith('f_') or name.startswith('flux_'):

                # Expand shortcut line flux naming scheme
                if name.startswith('f_'):
                    new_name = name.replace('f_','flux_')
                    self.logger.warning(f'Detected parameter {name}, expanding to {new_name}')
                    name = new_name

                # Check if it has a suffix
                suffix = None
                if len(name.split('_')) > 2:
                    suffix = name.split('_')[-1]

                # Check if line is in the line list, raise error if it isn't
                line_name = name.split('_')[1]
                if not line_name in linelist.names:
                    alternative = check_spelling_against_list(line_name, linelist.names)
                    if alternative is None:
                        msg = f'Line "{line_name}" not found in linelist!'
                    else:
                        msg = f'Line "{line_name}" not found in linelist! Did you mean "{alternative}"?'
                    self.logger.error(msg)
                    raise exceptions.MisspelledParameter(msg)
                params[name] = value
                
                if suffix is not None:
                    if f'fwhm_{line_name}_{suffix}' in kwargs:
                        fwhm_key = f'fwhm_{line_name}_{suffix}'
                    else:
                        fwhm_key = 'fwhm'
                else:
                    if f'fwhm_{line_name}' in kwargs:
                        fwhm_key = f'fwhm_{line_name}'
                    else:
                        fwhm_key = 'fwhm'

                # Check for corresponding FWHM, dv, values

                self.logger.info(f'Detected line {line_name}, flux {name}')
                self.lines[line_name] = {'flux_key': name, 'dv_key': None, 'fwhm_key': fwhm_key}

            else:
                msg = f"Parameter {name} not understood"
                self.logger.error(msg)
                raise exceptions.InconsistentParameter(msg)


        return params

    @property
    def __name__(self):
        return 'harmonizer.models.generic.SpectralLineModel'

    def __repr__(self):
        return f'harmonizer.models.generic.SpectralLineModel()'
    
    def __str__(self):
        return self.__repr__()

    def resample(self, wavelengths):
        self.wavelengths = wavelengths

    @staticmethod
    def _gaussian(x, L, mu, fwhm):
        sigma = mu * fwhm/2.998e5 / 2.355
        norm = sigma*np.sqrt(2*np.pi)
        y = np.exp(-(x-mu)**2/(2*sigma**2))
        y *= L/norm
        return y

    def get_sed(self, redshift, params):
        f2l = flux_to_lum(redshift)
        

        sed = np.zeros(len(self.wavelengths))
        
        self.line_grid = {}
        # line_names, line_wavs, line_fwhms, line_fluxes = [], [], [], []
        for line in self.lines:
            wav = linelist[line].wav

            flux_key = self.lines[line]['flux_key']
            flux = params[flux_key] # in erg/s/cm2

            fwhm_key = self.lines[line]['fwhm_key']
            fwhm = params[fwhm_key] # in erg/s/cm2

            dv_key = self.lines[line]['dv_key']
            if dv_key is not None:
                dv = params[dv_key]
                wav *= 1+dv/2.998e5

            lum = flux * f2l * (1+redshift) / 3.826e33
            g = self._gaussian(self.wavelengths, lum, wav, fwhm)
            # self.line_grid[key.replace('f_','')] = g
            sed += g

        return sed
