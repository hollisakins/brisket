from __future__ import print_function, division, absolute_import

import numpy as np

from brisket import config
import warnings
from brisket.models.base_models import *
from astropy.io import fits


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


class InoueIGMModel(BaseGriddedModel, BaseAbsorberModel):
    """ Allows access to and manipulation of the IGM attenuation models
    of Inoue (2014).

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the DL07 models.
    """

    type = 'absorber'  # Define the type of model
    order = 10  # Define the order of operations for this model
    # TODO could define defaults as class variables

    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        self.igm_redshifts = np.arange(0.0, config.max_redshift + 0.01, 0.01)
        self.igm_wavelengths = np.arange(1.0, 1225.01, 1.0)
        self.raw_igm_grid = fits.open(config.grid_dir + "/d_igm_grid_inoue14.fits")[1].data

    def _resample(self, wavelengths):
        """ Resample the raw grid to the input wavelengths. """
        self.wavelengths = wavelengths

        grid = np.zeros((self.wavelengths.shape[0],
                         self.igm_redshifts.shape[0]))

        for i in range(self.igm_redshifts.shape[0]):
            grid[:, i] = interp_discont(self.wavelengths,
                                        self.igm_wavelengths,
                                        self.raw_igm_grid[i, :], 1215.67,
                                        left=0., right=1.)
                                   
        # Make sure the pixel containing Lya is always IGM attenuated
        lya_ind = np.abs(self.wavelengths - 1215.67).argmin()
        if self.wavelengths[lya_ind] > 1215.67:
            grid[lya_ind, :] = grid[lya_ind-1, :]

        self.grid = grid

    def absorb(self, sed_incident, params):
        """ Apply the IGM attenuation to the input SED."""
        redshift = float(params['redshift'])
        redshift_mask = (self.igm_redshifts < redshift)
        zred_ind = self.igm_redshifts[redshift_mask].shape[0]

        zred_fact = ((redshift - self.igm_redshifts[zred_ind-1])
                     / (self.igm_redshifts[zred_ind]
                        - self.igm_redshifts[zred_ind-1]))

        if zred_ind == 0:
            zred_ind += 1
            zred_fact = 0.

        weights = np.array([1. - zred_fact, zred_fact])
        igm_trans = np.sum(weights*self.grid[:, zred_ind-1:zred_ind+1], axis=1)

        if 'xhi' in params:
            print('applying IGM damping wing')
            # apply IGM damping wing 
            self.tdmp = self.damping_wing(redshift, float(params['xhi']))
            igm_trans *= self.tdmp
        if 'logNH' in params:
            # apply DLA 
            igm_trans *= self.dla(redshift, params['logNH'])

        return sed_incident * igm_trans

    def damping_wing(self, redshift, xhi):
        zn = 8.8
        if redshift < zn:
            return np.ones(len(self.wavelengths))
        else:
            tau0 = 3.1e5
            Lambda = 6.25e8 # /s
            nu_alpha = 2.47e15 # Hz
            R_alpha = Lambda/(4*np.pi*nu_alpha)
            dwav = (self.wavelengths-1215.67)*(1+redshift)
            delta = dwav/(1215.67*(1+redshift))
            x2 = np.power(1+delta, -1)
            x1 = (1+zn)/((1+redshift)*(1+delta))
            tau = tau0 * xhi * R_alpha / np.pi * np.power(1+delta, 3/2) * (miralda_escude_eq12(x2)-miralda_escude_eq12(x1))
            trans = np.exp(-tau)
            trans[np.isnan(trans)] = 0
            return trans

    def dla(self, logNH):
        # Constants
        m_e = 9.1095e-28
        e = 4.8032e-10
        c = 2.998e10
        lamb = 1215.67
        f = 0.416
        gamma = 6.265e8
        broad = 1

        NH = np.power(10., logNH)
        C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
        a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
        dl_D = broad/c * lamb
        x = (self.wavelengths - lamb)/dl_D+0.01
        tau = np.array(C_a * NH * H(a,x), dtype=np.float64)
        return np.exp(-tau)

