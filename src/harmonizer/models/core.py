
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

        if self.params.ndim > 0:
            e = ValueError('Cannot create a model with free parameters. Please provide a set of fixed parameters.')
            self.logger.error(e)
            raise e

        assert 'redshift' in self.params, "Redshift must be specified for any model"
        # self.redshift = float(self.params['redshift'])
        # if self.redshift > config.max_redshift:
        #     raise ValueError("""Attempted to create a model with too high redshift. 
        #                     Please increase max_redshift in brisket/config.py 
        #                     before making this model.""")

        #### Do some brunt work to prepare the instrument models 
        # # Create a spectral calibration object to handle spectroscopic calibration and resolution.
        # self.calib = False
        # if self.spec_output and 'calib' in self.components: 
        #     self.calib = SpectralCalibrationModel(self._spec_wavs, self.params, logger=logger)
        
        # Calculate optimal wavelength sampling for the model
        self.logger.info('Calculating optimal wavelength sampling for the model')
        self.wavelengths, self.R = self.get_wavelength_sampling()
        
        # Initialize the various models and resample to the internal, optimized wavelength grid
        self.logger.info('Initializing the model components')
        self.components = self.params.components
        for comp_name, comp_params in self.components.items(): 
            # initialize the model
            comp_params.model = comp_params.model_func(comp_params, verbose=self.verbose) 
            comp_params.model.resample(self.wavelengths) # resample the model

            subcomps = comp_params.components
            for subcomp_name, subcomp_params in subcomps.items():
                subcomp_params.model = subcomp_params.model_func(subcomp_params, parent=comp_params.model, verbose=self.verbose)
                subcomp_params.model.resample(self.wavelengths)
            
            # then validate that sub-components were added correctly (only used by certain models)
            comp_params.model.validate_components(comp_params)

        # self.sources = [n for n,t in self.component_types.items() if t == 'source']
        # self.reprocessors = [n for n,t in self.component_types.items() if t == 'reprocessor']
        # self.absorbers = [n for n,t in self.component_types.items() if t == 'absorber']
        

        self.logger.info('Computing the SED')
        # Compute the main SED 
        self.compute_sed() 

        # Compute observables
        self.compute_observables()


    def compute_sed():


        from synthesizer.emission_models import TotalEmission
        from synthesizer.emission_models import UnifiedAGN


params = harmonizer.Params()
params['redshift'] = 7

stars = params.add_emitter('stars', emitter_function=synthesizer.parametric.Stars)
stars['grid'] = 'bc03'
stars['logMstar'] = 10 # Uniform(8, 11)
stars['zmet'] = 1
stars.add_photoionization()
stars.add_dust_attenuation()
stars.add_dust_emission(dust_emission_model=synthesizer.emission_models.dust.emission.Blackbody)
sfh = stars.add_sfh('constant', sfh_function=synthesizer.parametric.SFH.Constant)
sfh['age_min'] = 0.1 # Uniform(0.05, 0.15)
sfh['age_max'] = 0.4 # Uniform(0.4, 0.6)


params.add_emitter('agn', ) # UnifiedAGN
params.stars['nlr_grid'] = ...
params.stars['blr_grid'] = ...
params.stars.add_photoionization() # whether to include BLR/NLR emission
params.stars.add_dust_attenuation() # -> separate attenuation emissionmodel
params.stars.add_dust_emission() # -> torus, +reprocessing of attenuating dust