from __future__ import print_function, division, absolute_import

import numpy as np

from brisket import config
from brisket import utils


class DustEmissionModel(object):
    """ Allows access to dust emission models. Currently implemented models: 

        - Draine & Li (2007) radiative transfer models
        - Drew & Casey (2021) modified blackbody + mid-IR power law 

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the dust emission models.
    
    redshift : float
        redshift of the model
    
    model_comp : dict
        Subset of the model_components dictionary with dust emission parameters
    """

    def __init__(self, wavelengths, params, logger=utils.NullLogger):
        self.wavelengths = wavelengths
        self.params = params
        self.logger = logger

        self.flag = True
        if not 'dust_emission' in list(params):
            self.logger.info("Skipping dust emission module")
            self.flag = False; return

        self.redshift =  params['redshift']
        self.type = params['dust_emission']['type']
        if self.type == 'DL07':
            self.spectrum = self.spectrum_DL07
        elif self.type == 'MBBPL':
            self.spectrum = self.spectrum_MBBPL


    def spectrum_DL07(self, params):
        """ Get the Draine & Li (2007) model for a given set of model 
        parameters, given as keys in model_comp.

        Parameters
        ----------

        qpah : float
            The PAH fraction
        umin : float
            The minimum of the starlight intensity distribution
        gamma : float 
            The fraction of the dust heated by starlight (0-1)
        """

        qpah = params['dust_emission']['qpah']
        umin = params['dust_emission']['umin']
        gamma = params['dust_emission']['gamma']

        qpah_ind = config.qpah_vals[config.qpah_vals < qpah].shape[0]
        umin_ind = config.umin_vals[config.umin_vals < umin].shape[0]

        qpah_fact = ((qpah - config.qpah_vals[qpah_ind-1])
                     / (config.qpah_vals[qpah_ind]
                        - config.qpah_vals[qpah_ind-1]))

        umin_fact = ((umin - config.umin_vals[umin_ind-1])
                     / (config.umin_vals[umin_ind]
                        - config.umin_vals[umin_ind-1]))

        umin_w = np.array([(1 - umin_fact), umin_fact])

        lqpah_only = config.dust_grid_umin_only[qpah_ind]
        hqpah_only = config.dust_grid_umin_only[qpah_ind+1]
        tqpah_only = (qpah_fact*hqpah_only[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_only[:, umin_ind:umin_ind+2])

        lqpah_umax = config.dust_grid_umin_umax[qpah_ind]
        hqpah_umax = config.dust_grid_umin_umax[qpah_ind+1]
        tqpah_umax = (qpah_fact*hqpah_umax[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_umax[:, umin_ind:umin_ind+2])

        interp_only = np.sum(umin_w*tqpah_only, axis=1)
        interp_umax = np.sum(umin_w*tqpah_umax, axis=1)

        model = gamma*interp_umax + (1 - gamma)*interp_only

        spectrum = np.interp(self.wavelengths,
                             config.dust_grid_umin_only[1][:, 0],
                             model, left=0., right=0.)

        return spectrum


    def spectrum_MBBPL(self, params):
        """ Compute the Drew & Casey (2021) model for a given 
        set of model parameters, given as keys in model_comp.

        Parameters
        ----------

        Tdust : float
            The dust temperature 
        beta : float
            ...
        alpha : float 
            ...
        lam0 : float 
            ...
        """

        T = params['dust_emission']['Tdust']
        beta = params['dust_emission']['beta']
        alpha = params['dust_emission']['alpha']
        lam0 = params['dust_emission']['lam0'] # in micron

        hck = 143877687.75039333 # Kelvin*angstrom
        c = 2.9979e18 # angstrom/s
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            z = self.redshift
            T_CMB_z0 = 2.73
            T_CMB = T_CMB_z0*(1+z)
            Tdust_z = (T**(beta+4) + T_CMB_z0**(beta+4) * ((1+z)**(beta+4)-1))**(1/(beta+4))
            CMB = 1 - (np.power(c/self.wavelengths, 3) / (np.exp(hck/(self.wavelengths*T_CMB))-1)) / (np.power(c/self.wavelengths, 3) / (np.exp(hck/(self.wavelengths*Tdust_z))-1))
            T = Tdust_z
            CMB = np.where(self.wavelengths < 1e4, 1, CMB)

            lam_fine = np.logspace(3, 6.5, 1000)
            if lam0 == 'ot': MBB = np.power(c/lam_fine, beta) * np.power(c/lam_fine, 3) / (np.exp(hck/(lam_fine*T))-1) # optically thin 
            else: MBB = (1-np.exp(-(lam0*1e4/lam_fine)**beta)) * np.power(c/lam_fine, 3) / (np.exp(hck/(lam_fine*T))-1) # general opacity
            delta_y = np.diff(np.log10(MBB))
            delta_x = np.diff(np.log10(lam_fine))
            deriv = delta_y / delta_x
            
            lam_fine = 0.5*(lam_fine[1:] + lam_fine[:-1])
            lam_int = lam_fine[np.nanargmin(np.abs(deriv-alpha))]
            
            Npl = MBB[np.argmin(np.abs(lam_fine-lam_int))] * lam_int**(-alpha)
            PL = Npl * self.wavelengths**alpha

            if lam0 == 'ot': MBB = np.power(c/self.wavelengths, beta) * np.power(c/self.wavelengths, 3) / (np.exp(hck/(self.wavelengths*T))-1) # optically thin 
            else: MBB = (1-np.exp(-(lam0*1e4/self.wavelengths)**beta)) * np.power(c/self.wavelengths, 3) / (np.exp(hck/(self.wavelengths*T))-1) # general opacity
            spectrum =  np.where(self.wavelengths < lam_int, PL, MBB)
            # spectrum = np.where(self.wavelengths < 1.6e4, 0, spectrum)

            # convert from F_nu to F_lambda
            spectrum = spectrum / self.wavelengths**2
            # spectrum *= np.power(10., 1/(1+np.exp(-20*(np.log10(self.wavelengths)-4.8))))
        

            spectrum = np.power(10., np.log10(spectrum) * 1/(1+np.exp(-20*(np.log10(self.wavelengths)-3.8))))
            # renormalize to 1
            spectrum = spectrum / np.trapz(spectrum, x=self.wavelengths)

        return spectrum * CMB

    def __bool__(self):
        return self.flag
