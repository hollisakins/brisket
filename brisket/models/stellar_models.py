from __future__ import print_function, division, absolute_import

import astropy.units as u
import numpy as np
import os
from brisket import config
from brisket import utils
from brisket.utils.sed import SED
from brisket.utils import utils
import h5py

from brisket.data.grid_manager import GridManager
from brisket.models.base_models import *

class GriddedStellarModel(BaseGriddedModel, BaseSourceModel):
    type = 'source'
    order = 0
    def __init__(self, params):
        self.params = params
        
        self._build_defaults(params)

        grid_file_name = str(params['grids'])
        if not grid_file_name.endswith('.hdf5'):
            grid_file_name += '.hdf5'

        gm = GridManager()
        gm.check_grid(grid_file_name)
        grid_path = os.path.join(config.grid_dir, grid_file_name)

        self._load_hdf5_grid(grid_path)
    
    def _build_defaults(self, params):
        if 'grids' not in params:
            params['grids'] = 'bc03_miles_chabrier'
        if 't_bc' not in params:
            params['t_bc'] = 0.01
        # if 'logMstar' not in params:
            # params['logMstar'] = 10
        # if 'zmet' not in params:
            # params['zmet'] = 0.02

    def _validate_components(self, params):
        # TODO initialize SFHs
        # stack all the SFH components into one, there's no real point in keeping them separate
        self.sfh_components = {}
        for comp_name, comp in self.params.components.items():
            if comp.model.type == 'sfh':
                self.sfh_components[comp_name] = comp.model
        if len(self.sfh_components) == 1:
            self.sfh_weights = [1]
        else:
            if 'weight' in comp:
                self.sfh_weights = [float(comp['weight']) for comp in self.params.components.values() if comp.model.type == 'sfh']
            elif 'logweight' in comp:
                self.sfh_weights = [np.power(10., float(comp['logweight'])) for comp in self.params.components.values() if comp.model.type == 'sfh']

    def _load_hdf5_grid(self, grid_path):
        """ Load the grid from an HDF5 file. """

        with h5py.File(grid_path,'r') as f:
            self.grid_wavelengths = np.array(f['wavs'])
            self.grid_metallicities = np.array(f['metallicities'][:])
            self.grid_ages = np.array(f['ages'])
            self.grid = np.array(f['grid'])
            self.grid_live_frac = np.array(f['live_frac'][:])

        self.grid_ages[self.grid_ages == 0] = 1
        self.grid_age_bins = np.power(10., utils.make_bins(np.log10(self.grid_ages), fix_low=-99))
        self.grid_age_widths = self.grid_age_bins[1:] - self.grid_age_bins[:-1]


    
    def __repr__(self):
        return f'BaseStellarModel'
    
    def __str__(self):
        return self.__repr__()

    def _resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths
        self.grid = SED(self.grid_wavelengths, Llam=self.grid, verbose=False, units=False)
        self.grid.resample(self.wavelengths)


    def emit(self, params):
    
        """ Obtain a split 1D spectrum for a given star-formation and
        chemical enrichment history, one for ages lower than t_bc, one
        for ages higher than t_bc. This allows extra dust to be applied
        to the younger population still within its birth clouds.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        t_bc : float
            The age at which to split the spectrum in Gyr.
        """

        # TODO compute sfh_ceh from input SFH parameters
        sfh_ceh = np.zeros((len(self.grid_metallicities), len(self.grid_ages)))
        for (sfh_name, sfh), sfh_weight in zip(self.sfh_components.items(), self.sfh_weights):
            sfh.update(params[sfh_name], weight=sfh_weight)
            sfh_ceh += sfh.combined_weights


        t_bc = float(params['t_bc']) * 1e9
        young = SED(self.wavelengths, verbose=False, units=False)
        old = SED(self.wavelengths, verbose=False, units=False)

        index = self.grid_ages[self.grid_ages < t_bc].shape[0]
        old_weight = (self.grid_ages[index] - t_bc)/self.grid_age_widths[index-1]

        if index == 0:
            index += 1

        for i in range(len(self.grid_metallicities)):
            if sfh_ceh[i, :index].sum() > 0.:
                sfh_ceh[:, index-1] *= (1. - old_weight)
                # print(type(self.grid[i, :index, :].T))
                # print(type(sfh_ceh))
                young += np.sum(self.grid[i, :index, :].T * sfh_ceh[i, :index].T, axis=1)
                sfh_ceh[:, index-1] /= (1. - old_weight)

            if sfh_ceh[i, index-1:].sum() > 0.:
                sfh_ceh[:, index-1] *= old_weight
                old += np.sum(self.grid[i, index-1:, :].T * sfh_ceh[i, index-1:].T, axis=1)
                sfh_ceh[:, index-1] /= old_weight

        # if t_bc == 0.:
            # return spectrum

        return young+old
