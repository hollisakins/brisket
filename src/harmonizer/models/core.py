
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

        redshift = self.params['redshift']
        emission_models = []

        if self.params.has_emitter('stars'):
            star_params = self.params['stars']
            
            # load the stellar grid
            grid = Grid(emitter['grid_file'])
            if len(grid.axes) > 2:
                raise ValueError('Grid must have only two axes (log10age, metallicity)')

            # define the metallicity history
            zh = ZDist.DeltaConstant(metallicity = star_params['zmet'])

            # Define the star formation history
            sfh_p = {"max_age": 100 * Myr}
            sfh = SFH.Constant(**sfh_p)

            stars = ParametricStars(
                grid.log10age,
                grid.metallicity,
                sf_hist = sfh, 
                metal_dist = zh, 
                initital_mass = np.power(10., star_params['logMstar']) * Msun,
            )
            print(stars.total_mass)

            # next, handle emission models for the stars
            if (star_params.include_photoionization and 
                star_params.include_dust_attenuation and 
                star_params.include_dust_emission): 
                # use TotalEmission model
                pass
            
            elif (star_params.include_photoionization and 
                  star_params.include_dust_attenuation):
                # use EmergentEmission model
                pass

            elif star_params.include_photoionization:
                # use IntrisicEmission model
                pass

            elif (star_params.include_dust_attenuation and 
                  star_params.include_dust_emission):
                # use IncidentEmission + DustAttenuation + DustEmission
                pass
        
            elif star_params.include_dust_attenuation:
                # use IncidentEmission + DustAttenuation
                pass

            else:
                # use IncidentEmission
                pass

            emission_models.append(None)
        else:
            stars = None
                
        if self.params.has_emitter('agn'):
            pass

        else: 
            blackholes = None


        # now that we've assigned all the emission models, we can combine them
        galaxy = Galaxy(
            stars = stars,
            black_holes = blackholes,
            redshift = self.params['redshift'],
        )

        total_emission_model = EmissionModel(
            label="total",
            combine=emission_models,
            emitter="galaxy",
        )
        total_emission_model.save_spectra("total")#, "dust_emission", "total_attenuated", "total_intrinsic")

        if self.params.include_igm:
            # do something to figure out which IGM model the user wants to use
            pass
        else:
            igm_model = None

        spectra = galaxy.get_spectra(total_emission_model)
        obs_spectra = galaxy.get_observed_spectra(cosmo, igm=igm_model)





        from synthesizer.emission_models import UnifiedAGN


        if params.has_emitter('agn'):
            agn = params.get_emitter('agn')
        
            bh = agn.emitter_model(
                mass=np.power(10., agn['logMBH']) * Msun,
                inclination=agn['inclination'] * deg,
                accretion_rate=agn['acc_rate'] * Msun / yr,
                metallicity=agn['zmet'], # metallicity for the gas around the black hole
            )

            if agn['model'] == 'UnifiedAGN':
                uniagn = UnifiedAGN(
                    nlr_grid,
                    blr_grid,
                    covering_fraction_nlr=0.1,
                    covering_fraction_blr=0.1,
                    torus_emission_model=Greybody(1000 * K, 1.5),
                    ionisation_parameter=0.1,
                    hydrogen_density=1e5,
                )

                spectra = bh.get_spectra(uniagn)

