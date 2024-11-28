import numpy as np
import sys, os
import astropy.units as u
# from brisket import filters
# from brisket import utils

class Data:
    """
    A container for observational data loaded into BRISKET.
    """

    def __init__(self, 
                 ID, 
                 
                 filters = None,
                 phot = None,
                 phot_err = None,
                 phot_units=u.uJy,

                 spec_wavs = None,
                 spec = None, 
                 spec_err = None,
                 wav_units=u.angstrom,
                 spec_units=u.erg/u.s/u.cm**2/u.angstrom,

                 spectral_indices = None # currently not implemented
                 ):
    
        self.ID = ID
        self.load_phot = load_phot
        self.phot_units = phot_units
        self.filt_list = filt_list
        
        
        self.load_spec = load_spec
        self.wav_units = wav_units
        self.spec_units = spec_units
        self.spec_wavs = None

        self.spectrum_exists = True
        if not self.load_spec:
            self.spectrum_exists = False
        self.photometry_exists = True
        if not self.load_phot:
            self.photometry_exists = False

        if self.spectrum_exists:
            self.spectrum = load_spec(self.ID)

        if self.photometry_exists:
            phot_nowavs = load_phot(self.ID)

        # If photometry is provided, add filter effective wavelengths to array
        if self.photometry_exists:
            self.filter_set = filters.filter_set(filt_list, logger=logger)
            self.photometry = np.c_[self.filter_set.eff_wavs, phot_nowavs]

        # self._convert_units()

        # Mask the regions of the spectrum specified in masks/[ID].mask
        if self.spectrum_exists:
            self.spectrum = self._mask(self.spectrum)
            self.spec_wavs = self.spectrum[:, 0]

            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.spectrum[startn, 1] == 0.:
                startn += 1

            endn = 0
            while self.spectrum[-endn-1, 1] == 0.:
                endn += 1

            if endn == 0:
                self.spectrum = self.spectrum[startn:, :]

            else:
                self.spectrum = self.spectrum[startn:-endn, :]

            self.spec_wavs = self.spectrum[:, 0]

        # Deal with any spectral index calculations.
        # if load_indices is not None:
        #     self.index_names = [ind["name"] for ind in self.index_list]

        #     if callable(load_indices):
        #         self.indices = load_indices(self.ID)

        #     elif load_indices == "from_spectrum":
        #         self.indices = np.zeros((len(self.index_list), 2))
        #         for i in range(self.indices.shape[0]):
        #             self.indices[i] = measure_index(self.index_list[i],
        #                                             self.spectrum,
                                                    # self.index_redshift)
# try[:, 2] *= conversion

#     def _mask(self, spec):
#         """ Set the error spectrum to infinity in masked regions. """

#         if not os.path.exists(f'masks/{self.ID}_mask'):
#             return spec

#         # if self.spec_cov is not None:
#         #     raise ValueError("Automatic masking not supported where covariance"
#         #                      " matrix is specified, please do this manually.")

#         mask = np.loadtxt(f'masks/{self.ID}_mask')
#         if len(mask.shape) == 1:
#             wl_mask = (spec[:, 0] > mask[0]) & (spec[:, 0] < mask[1])
#             if spec[wl_mask, 2].shape[0] != 0:
#                 spec[wl_mask, 2] = 9.9*10**99.

#         if len(mask.shape) == 2:
#             for i in range(mask.shape[0]):
#                 wl_mask = (spec[:, 0] > mask[i, 0]) & (spec[:, 0] < mask[i, 1])
#                 if spec[wl_mask, 2].shape[0] != 0:
#                     spec[wl_mask, 2] = 9.9*10**99.

#         return spec

#     # def plot(self, show=True, return_y_scale=False, y_scale_spec=None):
#     #     return plotting.plot_galaxy(self, show=show,
#     #                                 return_y_scale=return_y_scale,
#     #                                 y_scale_spec=y_scale_spec)
