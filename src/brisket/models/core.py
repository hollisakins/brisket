import numpy as np
from rich.tree import Tree

from ..parameters import Params
from ..config import cosmo
from ..utils import exceptions
from ..utils.console import setup_logger, rich_str
from ..fitting.priors import Prior
from .base import *

class Formula(EmitterModel):
    """ 
    Stores a "formula" -- a combination of multiple models via a set of operations, without actually evaluating them. 
    """
    def __init__(self, base, add=None, mul=None, mod=None):
        self.operations = ['+']
        self.models = [base]
        
        if add is not None:
            self.operations.append('+')
            self.models.append(add)
        if mul is not None:
            self.operations.append('*')
            self.models.append(mul)
        if mod is not None:
            self.operations.append('%')
            self.models.append(mod)

    def _add_to_tree(self, tree):
        operations, models = self.operations, self.models
        for i in range(len(operations)):
            if isinstance(models[i], Formula):
                sub = tree.add(f"[bold italic white]{operations[i]} {self.__class__.__name__}[/bold italic white]()")
                models[i]._add_to_tree(sub)
            else:
                sub = tree.add(f"[bold #FFE4B5 not italic]{operations[i]} {models[i].name}:[/bold #FFE4B5 not italic] {models[i].__class__.__name__}")
                for param,value in models[i].params.items():
                    sub.add(f'[bold #FFE4B5 not italic]{param}[white]: [italic not bold #c9b89b]{value}')

    def __repr__(self):
        self.params # force params to be created
        tree = Tree(f"[bold italic white]{self.__class__.__name__}[/bold italic white]()")
        self._add_to_tree(tree)
        return rich_str(tree)


    def _add_to_params(self, params):
        operations, models = self.operations, self.models
        for i in range(len(operations)):
            if isinstance(models[i], Formula):
                models[i]._add_to_params(params)
            else:
                name = models[i].name
                all_keys = list(params.all_params.keys())
                j = 1
                while any([k.startswith(name+"/") for k in all_keys]):
                    name = f'{name}{j}'
                    j += 1

                models[i].name = name
                params += models[i].params.withprefix(name)

    @property
    def params(self):
        params = Params()
        self._add_to_params(params)
        return params

    def _add_to_sed(self, params, sed):
        operations, models = self.operations, self.models
        for i in range(len(operations)):
            model = models[i]
            operation = operations[i]

            if isinstance(model, Formula):
                model._add_to_sed(params, sed)
            else:
                params_i = params.getprefix(model.name)
                if operation == '+':
                    sed += model.get_sed(params_i)
                elif operation == '*':
                    sed *= model.get_transmission(params_i)
                elif operation == '%':
                    sed = model.process(sed, params_i)

    def get_sed(self, wavelengths, params):
        """
        Returns the SED for the given parameters.
        """
        # after all models have been prepared (e.g., wavelengths assigned/resampled, precomputed values calculated, etc.)
        # we can call the models in the order they were added
        # and apply the specified operations

        sed = np.zeros_like(wavelengths)
        self._add_to_sed(params, sed)
        return sed
        


class Model:
    """
    """
    # R_default = 800
    # max_wavelength = 1e9

    def __init__(self, 
                 redshift: float | Prior,
                 formula: Formula,
                 verbose: bool = False, 
                 **kwargs
        ):

        self.obs = None
        self.logger = setup_logger(__name__, verbose)

        if type(formula) is not Formula:
            # e.g., if formula is a single EmitterModel
            formula = Formula(formula)

        self.params = formula.params
        self.params['redshift'] = redshift
        print(formula)
        print(self.params)
        print(self.params.getprefix('stars'))
        # for operation, model_component in zip(formula.operations, formula.models):
        #     if operation == '+':
        #     else:
        #         raise ValueError(f"Unknown operation: {operation}")


        # Initialize the various models and resample to the internal, optimized wavelength grid
        # self.logger.info('Preparing component models')

        # self.params = Params(verbose=verbose, model=self)
        # self.params['redshift'] = redshift
        # for name, model_component in kwargs.items():
        #     self.params.add_child(name, params=model_component.params)
        #     self.params[name].model = model_component
        
        # if verbose:
        #     self.params.print_tree()

        # # Compute the SED
        # self._compute_sed()

        # # Compute the observables
        # self._compute_observables()


    def predict(self, obs):
        if obs is None:
            self._prepare_models()
        
        else:

            if self.obs == obs:
                # Models are already prepared
                # and the observation is the same
                self.logger.info('Using existing observation')
            else:
                # Observation has changed; need to re-prepare models
                # and update observation
                self.obs = obs
                self._prepare_models()
        
        self._compute_sed()

        if self.obs is not None:
            self._compute_observables()
        
    def _prepare_models(self):
        # Calculate optimal wavelength sampling for the model
        self.logger.info('Calculating optimal wavelength sampling for the model')
        self.wavelengths = self._get_wavelength_sampling()

        # Prepare the SED models
        self.logger.info('Resampling/assigning models to optimized wavelength grid')
        for child_name, child in self.params.children.items():
            model = child.model
            model.prepare(self.wavelengths)
        
    

    def _get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """
              
        max_wav = self.max_wavelength
        R_default = self.R_default
        wavelengths = [1.]
        if self.obs is None:
            while wavelengths[-1] < max_wav:
                w = wavelengths[-1]
                wavelengths.append(w*(1.+0.5/R_default))
            wavelengths = np.array(wavelengths)
            R = np.zeros_like(wavelengths) + R_default

        else:
            sig = 3
            R_wav = np.logspace(0, np.log10(config.max_wavelength.to(u.angstrom).value), 1000)
            R = np.zeros_like(R_wav) + R_default

            for phot in self.obs.phot_list:
                wav_range = (phot.wav_range[0].to(u.angstrom).value, phot.wav_range[1].to(u.angstrom).value)
                in_phot_range = (R_wav > wav_range[0]/(1+config.max_redshift) * (1-25*sig/len(R_wav))) & (R_wav < wav_range[1]/(1+config.min_redshift) * (1+25*sig/len(R_wav)))
                R[(in_phot_range)&(R<phot.R*20)] = phot.R*20
            for spec in self.obs.spec_list:
                in_spec_range = (R_wav > spec.wav_range[0]/(1+config.max_redshift) * (1-25*sig/len(R_wav))) & (R_wav < spec.wav_range[1]/(1+config.min_redshift) * (1+25*sig/len(R_wav)))
                Rspec = np.min([spec.R*5, config.R_max])
                R[(in_spec_range)&(R<Rspec)] = Rspec

            from astropy.convolution import convolve
            w = np.arange(-5*sig, 5*sig+1)
            kernel = np.exp(-0.5*(w/sig)**2)
            R = convolve(R, kernel)
            R[R<R_default] = R_default
            
            while wavelengths[-1] < max_wav:
                w = wavelengths[-1]
                r = np.interp(w, R_wav, R)
                wavelengths.append(w*(1.+0.5/r))
                
            wavelengths = np.array(wavelengths)
            R = np.interp(wavelengths, R_wav, R)

            self.logger.info('Resampling the filter curves onto model wavelength grid')
            for phot in self.obs.phot_list:
                phot.filters.resample_filter_curves(wavelengths)

        return wavelengths

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
        self.predict(self.obs)

    def _compute_sed(self):

        redshift = self.params['redshift']
        _sed = np.zeros_like(self.wavelengths)

        for child_name, child in self.params.children.items():
            p = self.params[child_name]
            model = child.model

            if model.model_type == 'emitter':
                _sed_i = child.model.get_sed(redshift, p)
                if np.ndim(_sed_i) == 2:
                    _sed = _sed[np.newaxis,:] + _sed_i
                else:
                    _sed += _sed_i

            elif model.model_type == 'absorber':
                t = model.get_transmission(redshift, p)
                _sed *= t


            # if 'stars' in self.params:
            #     star_params = self.params['stars']
            #     sed = self.stellar_model.get_sed(star_params)

            
            # # If nebular emission is included, we may need to collapse the grid
            # if 'nebular' in star_params:
            #     params_has_U = 'U' in star_params['nebular'] or 'logU' in star_params['nebular']
            #     grid_has_U = 'U' in self.stellar_grid.axes or 'logU' in self.stellar_grid.axes
            #     if params_has_U and not grid_has_U:
            #         errmsg = 'Nebular emission w/ logU requested, but grid does not have ionization parameter.'
            #         logger.error(errmsg)
            #         raise ValueError(errmsg)
            #     if not params_has_U and grid_has_U:
            #         errmsg = 'Grid has ionization parameter, please specify U or logU in the parameters.'
            #         logger.error(errmsg)
            #         raise ValueError(errmsg)

            #     if params_has_U:
            #         if 'logU' in self.stellar_grid.axes:
            #             self.stellar_grid.collapse(
            #                 'logU', 
            #                 star_params['nebular']['logU'], 
            #                 method='interpolate'
            #             )
            #         elif 'U' in self.stellar_grid.axes:
            #             self.stellar_grid.collapse(
            #                 'U', 
            #                 star_params['nebular']['U'], 
            #                 method='interpolate', 
            #                 pre_interp_function=np.log10
            #             )

            #     fesc = star_params['nebular'].get('fesc', 0.0)
            #     fesc_ly_alpha = star_params['nebular'].get('fesc_lya', 1.0)

            # # next, handle emission models for the stars
            # if ('nebular' in star_params and 
            #     'dust_attenuation' in star_params and 
            #     'dust_emission' in star_params): 
            #     # use TotalEmission model
            #     print('Will use totalemission model')
            #     pass
            
            # elif ('photoionization' in star_params and 
            #       'dust_attenuation' in star_params):
            #     # use EmergentEmission model
            #     print('Will use EmergentEmission model')
            #     pass

            # elif 'nebular' in star_params:
            #     # use IntrisicEmission model
            #     print('Will use IntrinsicEmission model')
            #     model = IntrinsicEmission(self.stellar_grid, 
            #         fesc=fesc, 
            #         fesc_ly_alpha=fesc_ly_alpha, 
            #         emitter="galaxy")

            # elif (star_params.include_dust_attenuation and 
            #       star_params.include_dust_emission):
            #     print('Will use IncidentEmission model + DustAttenuation model + DustEmission model')
            #     # use IncidentEmission + DustAttenuation + DustEmission
            #     pass
        
            # elif star_params.include_dust_attenuation:
            #     print('Will use IncidentEmission model + DustAttenuation model')
            #     # use IncidentEmission + DustAttenuation
            #     pass

            # else:
            #     print('Will use IncidentEmission model')
            #     # use IncidentEmission
            #     pass

            # if 'agn' in self.params:
            #     pass

            # if 'igm' in self.params:
            #     # do something to figure out which IGM model the user wants to use
            #     pass
            # else:
            #     igm_model = None

        # galaxy.get_spectra(total_emission_model)
        # galaxy.get_observed_spectra(cosmo, igm=igm_model)
        # self.sed = galaxy.spectra['total']

        # Convert from luminosity to observed flux at redshift z.
        # _sed *= utils.lum_to_flux(redshift)        
        self._sed = _sed

    def _compute_observables(self, obs):
        """ Compute the observables for the model. This is done by 
        convolving the SED with the filter curves and/or spectroscopic 
        resolution. """
        if obs is None:
            self.logger.warning('No observation provided, skipping observable computation')
            return

        # Compute the photometry
        for phot in obs.phot_list:
            phot.compute(self.wavelengths, self._sed, redshift, self.params)

        # Compute the spectroscopy
        for spec in obs.spec_list:
            spec.predict(self.wavelengths, self._sed, redshift, self.params)

    @property 
    def sed(self):
        """
        Synthesizer Sed object representing the full internal model SED. 
        Provided for convenience (internal routines use private attribute _sed)
        """
        lam = self.wavelengths * angstrom
        lnu = self._sed * erg/s/Hz
        # lnu = llam_to_lnu(lam, llam)
        sed = Sed(lam=lam, lnu=lnu)
        fnu = sed.get_fnu(cosmo, self.params['redshift'])
        return sed

    @property
    def __name__(self):
        return 'harmonizer.models.Model'


    def __repr__(self):
        return 'harmonizer.models.Model()'

