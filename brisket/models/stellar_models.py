from __future__ import print_function, division, absolute_import

import numpy as np
import os
from brisket import config
from brisket import utils
from brisket.utils.sed import SED
from brisket.utils import utils

from brisket.data.grid_manager import GridManager
from brisket.models.base_models import *

# Set up edge positions for age bins for stellar + nebular models.
age_bins = 10**utils.make_bins(config.age_sampling, make_rhs=True)[0]
age_bins[0] = 0.
age_bins[-1] = 10**9*cosmo.age(0.).value

# Set up widths for the age bins for the stellar + nebular models.
age_widths = age_bins[1:] - age_bins[:-1]

# Convert the age sampling from log10(Gyr) to Gyr.
age_sampling = 10**age_sampling


class GriddedStellarModel(BaseGriddedModel, BaseSourceModel):
    def __init__(self, params):
        self.params = params
        grid_file_name = params['grids']
        if not grid_file_name.endswith('.hdf5'):
            grid_file_name += '.hdf5'

        gm = GridManager()
        gm.check_grid(grid_file_name)
        grid_path = os.path.join(config.grid_dir, grid_file_name)

        self._load_hdf5_grid(grid_path)

        self.sfh_components = {}
        for comp_name, comp in self.params.components.items():
            if comp.model.type == 'sfh':
                self.sfh_components[comp_name] = comp.model

        # TODO initialize SFHs
        # stack all the SFH components into one, there's no real point in keeping them separate

    def _load_hdf5_grid(self, grid_path):
        """ Load the grid from an HDF5 file. """

        with h5py.File(grid_path,'r') as f:
            self.grid_wavelengths = np.array(f['wavs'])
            self.grid_metallicities = np.array(f['metallicities'][:])
            self.grid_ages = np.array(f['ages'])
            self.grid = np.array(f['grid'])
            # self.live_frac = np.array(f['live_frac'][:])

        self.grid_age_bins = np.power(10., utils.make_bins(np.log10(self.grid_ages), make_rhs=True)[0])
        self.grid_age_bins[0] = 0.
        self.grid_age_bins[-1] = config.cosmo.age(0.).value * 1e9
        self.grid_age_widths = self.grid_age_bins[1:] - self.grid_age_bins[:-1]


    
    def __repr__(self):
        return f'BaseStellarModel'
    
    def __str__(self):
        return self.__repr__()

    def _resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths

        self.grid = SED(seld.grid_wavelengths*u.angstrom, Llam=self.grid*u.Lsun/u.AA)
        self.grid.resample(self.wavelengths*u.angstrom).value


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

        t_bc *= 10**9
        spectrum_young = np.zeros_like(self.wavelengths)
        spectrum = np.zeros_like(self.wavelengths)

        index = self.grid_ages[self.grid_ages < t_bc].shape[0]
        old_weight = (self.grid_ages[index] - t_bc)/self.grid_age_widths[index-1]

        if index == 0:
            index += 1

        for i in range(len(self.grid_metallicities)):
            if sfh_ceh[i, :index].sum() > 0.:
                sfh_ceh[:, index-1] *= (1. - old_weight)

                spectrum_young += np.sum(self.grid[:, i, :index]
                                         * sfh_ceh[i, :index], axis=1)

                sfh_ceh[:, index-1] /= (1. - old_weight)

            if sfh_ceh[i, index-1:].sum() > 0.:
                sfh_ceh[:, index-1] *= old_weight

                spectrum += np.sum(self.grid[:, i, index-1:]
                                   * sfh_ceh[i, index-1:], axis=1)

                sfh_ceh[:, index-1] /= old_weight

        if t_bc == 0.:
            return spectrum

        return spectrum_young, spectrum
