import numpy as np
from unyt import Msun

from . import sfzh
from ..console import setup_logger

from synthesizer import Grid
from synthesizer.parametric import Stars, Galaxy
from synthesizer.emission_models import (
    EmissionModel,
    IntrinsicEmission
)

from astropy.cosmology import Planck18 as cosmo

grid_dir = '/Users/hba423/codes/synthesizer/tests/test_grid/'

class Model(object):
    """
    Model galaxy generation with `harmonizer`.
    Uses `synthesizer` as the engine to generate galaxy SEDs.

    Args:
        parameters (harmonizer.parameters.Params)
            Model parameters.
    
        verbose (bool, optional)
            Whether to print log messages (default: True).
    """

    def __init__(self, 
                 params, 
                 instrument = None,
                 verbose: bool = False):

        self.params = params
        self.params.validate()
        self.logger = setup_logger(__name__, verbose)

        if self.params.ndim > 0:
            e = ValueError('Cannot create a model with free parameters. Please provide a set of fixed parameters.')
            self.logger.error(e)
            raise e

        # Calculate optimal wavelength sampling for the model
        self.logger.info('Calculating optimal wavelength sampling for the model')
        self.wavelengths, self.R = self.get_wavelength_sampling()

        # Initialize the observables
        self._prepare_observables()
        
        # Initialize the various models and resample to the internal, optimized wavelength grid
        self.logger.info('Initializing the models')
        
        if 'stars' in self.params:
            # Load the stellar grid and resample to the internal wavelength grid
            file = self.params['stars']['grid']
            if not file.endswith('.hdf5'):
                file += '.hdf5'
            self.stellar_grid = Grid(file, grid_dir=grid_dir, new_lam=self.wavelengths)

            if not self.params['stars'].has_sfh:
                raise ValueError('Must specify an SFH for the stellar component')

            if 'nebular' in self.params['stars']:
                pass
            if 'dust_atten' in self.params['stars']:
                pass
            if 'dust_emission' in self.params['stars']:
                pass
        
        if 'agn' in self.params:
            # Load the agn grid and resample to the internal wavelength grid
            file = self.params['agn']['grid']
            if not file.endswith('.hdf5'):
                file += '.hdf5'
            self.agn_grid = Grid(file, grid_dir=grid_dir, new_lam=self.wavelengths)

            if 'nebular' in self.params['agn']:
                pass
            if 'dust_atten' in self.params['agn']:
                pass
            if 'dust_emission' in self.params['agn']:
                pass

        self.logger.info('Computing the SED')
        self._compute_sed() 

        # Compute observables
        self.compute_observables()

    def _prepare_observables(self):

        #### Do some brunt work to prepare the instrument models 
        # # Create a spectral calibration object to handle spectroscopic calibration and resolution.
        # self.calib = False
        # if self.spec_output and 'calib' in self.components: 
        #     self.calib = SpectralCalibrationModel(self._spec_wavs, self.params, logger=logger)
        pass


    def update(self, params):
        """ Update the model outputs (spectra, photometry) to reflect 
        new parameter values in the parameters dictionary. Note that 
        only the changing of numerical values is supported."""
        self.params.update(params)

        # Compute the internal full SED 
        self.compute_sed()

        # Compute observables
        self.compute_observables()


    def compute_sed(self):

        redshift = self.params['redshift']

        emission_models = []
        if 'stars' in self.params:
            star_params = self.params['stars']
            
            if 'logZ' in star_params:
                # If `logZ` is given, assume the user wants a delta function metallicity history
                zh = sfzh.Normal(mean=0.01, sigma=0.05)
            else:
                # look for a metallicity function in the parameters
                pass

            sfh_name = star_params.sfh_name
            sfh_model = sfzh.get(sfh_name)
            sfh = sfh_model(**star_params[sfh_name])

            initial_mass = np.power(10., star_params['logMstar']) * Msun
            
            stars = Stars (
                self.stellar_grid.log10age,
                self.stellar_grid.metallicity,
                sf_hist = sfh, 
                metal_dist = zh, 
                initial_mass = initial_mass
            )

            # If nebular emission is included, we may need to collapse the grid
            if 'nebular' in star_params:
                params_has_U = 'U' in star_params['nebular'] or 'logU' in star_params['nebular']
                grid_has_U = 'U' in self.stellar_grid.axes or 'logU' in self.stellar_grid.axes
                if params_has_U and not grid_has_U:
                    errmsg = 'Nebular emission w/ logU requested, but grid does not have ionization parameter.'
                    logger.error(errmsg)
                    raise ValueError(errmsg)
                if not params_has_U and grid_has_U:
                    errmsg = 'Grid has ionization parameter, please specify U or logU in the parameters.'
                    logger.error(errmsg)
                    raise ValueError(errmsg)

                if params_has_U:
                    if 'logU' in self.stellar_grid.axes:
                        self.stellar_grid.collapse(
                            'logU', 
                            star_params['nebular']['logU'], 
                            method='interpolate'
                        )
                    elif 'U' in self.stellar_grid.axes:
                        self.stellar_grid.collapse(
                            'U', 
                            star_params['nebular']['U'], 
                            method='interpolate', 
                            pre_interp_function=np.log10
                        )

                fesc = star_params['nebular'].get('fesc', 0.0)
                fesc_ly_alpha = star_params['nebular'].get('fesc_lya', 1.0)

            # next, handle emission models for the stars
            if ('nebular' in star_params and 
                'dust_attenuation' in star_params and 
                'dust_emission' in star_params): 
                # use TotalEmission model
                print('Will use totalemission model')
                pass
            
            elif ('photoionization' in star_params and 
                  'dust_attenuation' in star_params):
                # use EmergentEmission model
                print('Will use EmergentEmission model')
                pass

            elif 'nebular' in star_params:
                # use IntrisicEmission model
                print('Will use IntrinsicEmission model')
                model = IntrinsicEmission(self.stellar_grid, 
                    fesc=fesc, 
                    fesc_ly_alpha=fesc_ly_alpha, 
                    emitter="galaxy")

            elif (star_params.include_dust_attenuation and 
                  star_params.include_dust_emission):
                print('Will use IncidentEmission model + DustAttenuation model + DustEmission model')
                # use IncidentEmission + DustAttenuation + DustEmission
                pass
        
            elif star_params.include_dust_attenuation:
                print('Will use IncidentEmission model + DustAttenuation model')
                # use IncidentEmission + DustAttenuation
                pass

            else:
                print('Will use IncidentEmission model')
                # use IncidentEmission
                pass

            emission_models.append(model)
                
        if 'agn' in self.params:
            pass


        # # now that we've assigned all the emission models, we can combine them
        # galaxy = Galaxy(
        #     stars = stars,
        #     black_holes = blackholes,
        #     redshift = redshift,
        # )

        # total_emission_model = EmissionModel(
        #     label="total",
        #     combine=emission_models,
        #     emitter="galaxy",
        # )
        # total_emission_model.save_spectra("total")#, "dust_emission", "total_attenuated", "total_intrinsic")

        if 'igm' in self.params:
            # do something to figure out which IGM model the user wants to use
            pass
        else:
            igm_model = None

        # galaxy.get_spectra(total_emission_model)
        # galaxy.get_observed_spectra(cosmo, igm=igm_model)
        # self.sed = galaxy.spectra['total']

