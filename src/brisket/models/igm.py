'''
IGM models
'''
from __future__ import annotations

import numpy as np
import warnings, os, h5py
from astropy.io import fits
from ..parameters import Params
from ..utils.console import setup_logger

from scipy.interpolate import interp1d
# from synthesizer.emission_models.transformers import Inoue14
# from unyt import angstrom

def miralda_escude_eq12(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.power(x, 9/2)/(1-x) + 9/7*np.power(x, 7/2) + 9/5*np.power(x, 5/2) + 3*np.power(x, 3/2) + 9*np.power(x, 1/2) - 9/2*np.log((1+np.power(x,1/2))/(1-np.power(x,1/2)))

# The Voigt-Hjerting profile based on the numerical approximation by Garcia
def H(a,x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5*x**(-2)
    return H0 - a / np.sqrt(np.pi) /\
    P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0 )

def interp_discont(x, xp, fp, xdiscont, left=None, right=None):
    """Interpolates separately on both sides of a discontinuity (not over it)"""
    i  = np.searchsorted(x, xdiscont)
    ip = np.searchsorted(xp, xdiscont)
    y1 = np.interp(x[:i], xp[:ip], fp[:ip], left=left)
    y2 = np.interp(x[i:], xp[ip:], fp[ip:], right=right)
    y  = np.concatenate([y1, y2])
    return y

class BaseIGMModel:

    model_type = 'absorber'

    def __init__(self, verbose=False, **kwargs):

        self.verbose = verbose
        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')
        
        self.params = self.validate(kwargs)
    
    def validate(self, kwargs):
        """
        Validate/parse parameters for the IGM model.
        """
        params = Params()
        return params

    def prepare(self, wavelengths):
        """
        Prototype for child defined "prepare" methods.
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


class Inoue2014IGMModel(BaseIGMModel):
    """ 
    IGM model of Inoue et al. 2014
    """

    def __init__(self, verbose=False, **kwargs):
        BaseIGMModel.__init__(self, verbose=verbose, **kwargs)

        self.igm_grid_path = os.path.join(utils.data_dir, 'igm_inoue14_grid.hdf5')
        self.igm_redshifts = np.arange(0.0, utils.max_redshift + 0.01, 0.01)
        self.igm_wavelengths = np.arange(1.0, 1225.01, 1.0)

        self.params = self.validate(kwargs)

    def validate(self, kwargs):
        params = BaseIGMModel.validate(self, kwargs)

        if 'scale_tau' in kwargs:
            params['scale_tau'] = kwargs['scale_tau']
        else:
            params['scale_tau'] = 1.

        return params

    @property
    def grid_exists(self):
        return os.path.exists(self.igm_grid_path)

    def prepare(self, wavelengths):
        """ Resample the raw grid to the input wavelengths. """
        
        # If the IGM grid has not yet been calculated, calculate it now.
        if not self.grid_exists:
            _, _, igm_grid = self._generate_grid(self.igm_redshifts, self.igm_wavelengths)
        else:
            # open igm_grid_path, check for max_redshift parameter
            igm_redshifts, igm_wavelengths, igm_grid = self._load_grid()

            z_check = np.all(igm_redshifts == self.igm_redshifts)
            wav_check = np.all(igm_wavelengths == self.igm_wavelengths)

            if not wav_check or not z_check:
                _, _, igm_grid = self._generate_grid(self.igm_redshifts, self.igm_wavelengths)


        self.logger.info('Resampling IGM absorption grid to input wavelengths')
        self.wavelengths = wavelengths
        new_igm_grid = np.zeros((len(self.wavelengths), len(self.igm_redshifts)))

        for i in range(len(self.igm_redshifts)):
            new_igm_grid[:, i] = interp_discont(
                self.wavelengths,
                self.igm_wavelengths,
                igm_grid[:, i], 
                1215.67,
                left=0., right=1.)
            
        # Make sure the pixel containing Lya is always IGM attenuated
        # lya_ind = np.abs(self.wavelengths - 1215.67).argmin()
        # if self.wavelengths[lya_ind] > 1215.67:
        #     grid[lya_ind, :] = grid[lya_ind-1, :]

        self.grid = new_igm_grid

    def _load_grid(self):
        with h5py.File(self.igm_grid_path, 'r') as f:
            redshifts = f["redshifts"][:]
            wavelengths = f["wavelengths"][:]
            transmission = f["transmission"][:]

        return redshifts, wavelengths, transmission

    def _generate_grid(self, redshifts, wavelengths):

        self.logger.info('Generating IGM absorption grid')
        mod = Inoue14(scale_tau=self.params['scale_tau'])
        
        transmission = np.zeros((len(wavelengths), len(redshifts)))
        for i,z in enumerate(redshifts):
            transmission[:,i] = mod.get_transmission(z, wavelengths*(1+z)*angstrom)

        with h5py.File(self.igm_grid_path, 'w') as f:
            d0 = f.create_dataset("redshifts", data=redshifts)
            d1 = f.create_dataset("wavelengths", data=wavelengths)
            d2 = f.create_dataset("transmission", data=transmission)

        return redshifts, wavelengths, transmission

    def get_transmission(self, redshift, params):
        """ Get the IGM transmission at the given redshift."""

        # If the interpolator doesn't exist, create it
        if not hasattr(self, '_interpolator'):
            self._interpolator = interp1d(self.igm_redshifts, self.grid, bounds_error=False, fill_value=None)

        return self._interpolator(redshift).T


        # if np.ndim(redshift) == 0: # Not vectorized
        #     redshift_mask = (self.igm_redshifts < redshift)
        #     zred_ind = self.igm_redshifts[redshift_mask].shape[0]

        #     zred_fact = ((redshift - self.igm_redshifts[zred_ind-1])
        #                 / (self.igm_redshifts[zred_ind]
        #                     - self.igm_redshifts[zred_ind-1]))

        #     if zred_ind == 0:
        #         zred_ind += 1
        #         zred_fact = 0.

        #     weights = np.array([1. - zred_fact, zred_fact])
        #     igm_trans = np.sum(weights*self.grid[:, zred_ind-1:zred_ind+1], axis=1)










    #     if 'xhi' in params:
    #         # apply IGM damping wing 
    #         self.tdmp = self.damping_wing(redshift, float(params['xhi']))
    #         igm_trans *= self.tdmp
    #     if 'logNH' in params:
    #         # apply DLA 
    #         igm_trans *= self.dla(redshift, params['logNH'])

    #     return sed_incident * igm_trans

    # def damping_wing(self, redshift, xhi):
    #     zn = 8.8
    #     if redshift < zn:
    #         return np.ones(len(self.wavelengths))
    #     else:
    #         tau0 = 3.1e5
    #         Lambda = 6.25e8 # /s
    #         nu_alpha = 2.47e15 # Hz
    #         R_alpha = Lambda/(4*np.pi*nu_alpha)
    #         dwav = (self.wavelengths-1215.67)*(1+redshift)
    #         delta = dwav/(1215.67*(1+redshift))
    #         x2 = np.power(1+delta, -1)
    #         x1 = (1+zn)/((1+redshift)*(1+delta))
    #         tau = tau0 * xhi * R_alpha / np.pi * np.power(1+delta, 3/2) * (miralda_escude_eq12(x2)-miralda_escude_eq12(x1))
    #         trans = np.exp(-tau)
    #         trans[np.isnan(trans)] = 0
    #         return trans

    # def dla(self, logNH):
    #     # Constants
    #     m_e = 9.1095e-28
    #     e = 4.8032e-10
    #     c = 2.998e10
    #     lamb = 1215.67
    #     f = 0.416
    #     gamma = 6.265e8
    #     broad = 1

    #     NH = np.power(10., logNH)
    #     C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
    #     a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
    #     dl_D = broad/c * lamb
    #     x = (self.wavelengths - lamb)/dl_D+0.01
    #     tau = np.array(C_a * NH * H(a,x), dtype=np.float64)
    #     return np.exp(-tau)

