from __future__ import annotations
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import toml, sys, os
from .. import config

from .resample import make_bins

class Filters:
    """Class for loading and manipulating sets of filter curves. 

    Parameters
    ----------

    filters : list
        List of names of filters, as defined in filter_directory.toml
    """

    def __init__(self, names, verbose=False):
        self.wavelengths = None
        self.verbose = verbose
        self.names = names
        self._load_filter_curves()
        self._calculate_effective_wavelengths()

    def _load_filter_curves(self):
        """ Loads filter files for the specified filter_names and truncates
        any zeros from either of their edges. """
        
        all_nicknames = {}
        
        self.filt_db = toml.load(config.filter_directory)
        for key in self.filt_db:
            for n in self.filt_db[key]['nicknames']:
                all_nicknames[n] = key
        
        self.filt_dict = {}
        self.nicknames = []
        
        if self.verbose: print("Loading filters")
        if self.verbose: print("-" * 80)
        l = np.max([len(i) for i in self.names])+3
        if self.verbose: print(f"Nickname".rjust(l) + ' -> ' + "Filter ID")
        for filt in self.names:
            if filt in all_nicknames:
                self.nicknames.append(all_nicknames[filt])
                self.filt_dict[filt] = np.loadtxt(os.path.join(os.path.split(config.filter_directory)[0], f'{all_nicknames[filt]}'))
                if self.verbose: print(f"{filt}".rjust(l) + ' -> ' +  f"{all_nicknames[filt]}") #+ f"{self.filt_db[all_nicknames[filt]]['description']}".ljust(32))
            else:
                raise ValueError(f"""Failed to match {filt} to any filter curve or nickname in database. Make sure it is named properly, or add it to the filter database using `brisket-filters add`""")
                

        if self.verbose: print("-" * 80)

    def _calculate_effective_wavelengths(self):
        """ Calculates effective wavelengths for each filter curve. """

        self.wav = np.zeros(self.__len__())
        self.wav_min = np.zeros(self.__len__())
        self.wav_max = np.zeros(self.__len__())

        for i in range(self.__len__()):
            filt = self.names[i]
            dlambda = make_bins(self.filt_dict[filt][:, 0])[1]
            filt_weights = dlambda*self.filt_dict[filt][:, 1]
            self.wav[i] = np.sqrt(np.sum(filt_weights*self.filt_dict[filt][:, 0])
                                       / np.sum(filt_weights
                                       / self.filt_dict[filt][:, 0]))
            self.wav_min[i] = np.min(self.filt_dict[filt][:,0][self.filt_dict[filt][:,1]/np.max(self.filt_dict[filt][:,1])>0.5])
            self.wav_max[i] = np.max(self.filt_dict[filt][:,0][self.filt_dict[filt][:,1]/np.max(self.filt_dict[filt][:,1])>0.5])

    def resample_filter_curves(self, wavelengths):
        """ Resamples the filter curves onto a new set of wavelengths
        and creates a 2D array of filter curves on this sampling. """

        self.wavelengths = wavelengths  # Wavelengths for new sampling

        # Array containing filter profiles on new wavelength sampling
        self.filt_array = np.zeros((wavelengths.shape[0], len(self.names)))

        # Array containing the width in wavelength space for each point
        self.widths = np.diff(make_bins(wavelengths))

        for i in range(len(self.names)):
            filt = self.names[i]
            self.filt_array[:, i] = np.interp(wavelengths,
                                              self.filt_dict[filt][:, 0],
                                              self.filt_dict[filt][:, 1],
                                              left=0, right=0)

    def get_photometry(self, y, redshift):
        """ Calculates photometric fluxes. The filters are first re-
        sampled onto the same wavelength grid with transmission values
        blueshifted by (1+z). This is followed by an integration over
        the observed spectrum in the rest frame:

        flux = integrate[(f_lambda*lambda*T(lambda*(1+z))*dlambda)]
        norm = integrate[(lambda*T(lambda*(1+z))*dlambda))]
        photometry = flux/norm

        lambda:            rest-frame wavelength array
        f_lambda:          observed spectrum
        T(lambda*(1+z)):   transmission of blueshifted filters
        dlambda:           width of each wavelength bin

        The integrals over all filters are done in one array operation
        to improve the speed of the code.
        """

        if self.wavelengths is None:
            raise ValueError("Please use resample_filter_curves method to set"
                             + " wavelengths before calculating photometry.")

        redshifted_wavs = self.wavelengths*(1. + redshift)

        # Array containing blueshifted filter curves
        filters_z = jnp.zeros_like(self.filt_array)

        # blueshift filter curves to sample right bit of rest frame spec
        for i in range(len(self.names)):
            new_filter = jnp.interp(
                redshifted_wavs, 
                self.wavelengths,
                self.filt_array[:, i],
                left=0, right=0)

            filters_z = filters_z.at[:, i].set(new_filter)

        # Calculate numerator of expression
        flux = jnp.expand_dims(y * self.widths * self.wavelengths, axis=1)
        flux = jnp.sum(flux * filters_z, axis=0)

        # Calculate denominator of expression
        norm = filters_z * jnp.expand_dims(self.widths * self.wavelengths, axis=1)
        norm = jnp.sum(norm, axis=0)

        photometry = jnp.squeeze(flux/norm)

        # # This is a little dodgy as pointed out by Ivo, it should depend
        # # on the spectral shape however only currently used for UVJ mags
        # if unit_conv == "cgs_to_mujy":
        #     photometry /= (10**-29*2.9979*10**18/self.eff_wavs**2)

        return photometry


    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return f"Filters({', '.join(self.nicknames)})"