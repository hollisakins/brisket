import numpy as np
import warnings
import os, time
from copy import deepcopy
from dotmap import DotMap
import spectres

from astropy.constants import c as speed_of_light
import logging
import astropy.units as u

from brisket import utils
from brisket import config
from brisket import filters
# from .. import plotting

from brisket.models.stellar_model import StellarModel
from brisket.models.nebular_model import NebularModel
from brisket.models.star_formation_history import StarFormationHistoryModel
from brisket.models.dust_emission_model import DustEmissionModel
from brisket.models.dust_attenuation_model import DustAttenuationModel
from brisket.models.accretion_disk_model import AccretionDiskModel
from brisket.models.agn_line_model import AGNLineModel
from brisket.models.igm_model import IGMModel
from brisket.models.calibration import SpectralCalibrationModel
from brisket.parameters import Params

class ModelGalaxy(object):
    """ Model galaxy generation with BRISKET.

    Parameters
    ----------
    
    parameters, 
    filt_list=None,
    spec_wavs=None,
    wav_units=u.um,
    sed_units=u.uJy,
    spec_units=u.erg/u.s/u.cm**2/u.angstrom,
    phot_units=u.uJy,
    logger=utils.NullLogger

    parameters : brisket.parameters.Params object or dict
        Model parameters. If provided as dict, it will be converted 
        to brisket.parameters.Params object to validate the input. 

    filt_list : list - optional
        A list of filter curves as defined in brisket.filters. 
        Only needed if photometric output is desired (internal 
        model SED will be generated regardless).
        Default: None

    spec_wavs : list - optional
        An array of spectroscopic wavelengths.
        Only needed if spectroscopic output is desired (internal 
        model SED will be generated regardless).
        Default: None

    wav_units : astropy.units.Unit - optional
        Desired wavelength units for spectroscopic/SED output. 
        Default: micron

    sed_units : astropy.units.Unit - optional
        Desired flux units for model SED output. Can specify units in 
        f_nu or f_lambda, or nu*f_nu / lambda*f_lambda. 
        Default: uJy
    
    spec_units : astropy.units.Unit - optional
        Desired flux units for spectroscopic output. Can specify units in 
        f_nu or f_lambda. 
        Default: erg/s/cm**2/angstrom

    phot_units : astropy.units.Unit - optional
        Desired flux units for photometric output. Can specify units in 
        f_nu or f_lambda. 
        Default: uJy
    """
    def __init__(self, 
                 parameters, 
                 filt_list=None,
                 spec_wavs=None,
                 wav_units=u.micron,
                 sed_units=u.uJy,
                 spec_units=u.erg/u.s/u.cm**2/u.angstrom,
                 phot_units=u.uJy,
                 logger=utils.NullLogger):

        self.logger = logger
        self.spec_wavs = spec_wavs
        self.filt_list = filt_list
        self.wav_units = wav_units
        self.sed_units = sed_units
        self.spec_units = spec_units
        self.phot_units = phot_units

        self.phot_output, self.spec_output = False, False
        if self.filt_list is not None: self.phot_output = True
        if self.spec_wavs is not None: self.spec_output = True

        # Handl the input parameters, whether provided in dictionary form or in Params object.
        if isinstance(parameters, Params):
            self.logger.debug(f'Parameters loaded')            
            self.parameters = parameters
        elif isinstance(parameters, dict):
            self.logger.info(f'Loading parameter dictionary')           
            self.parameters = Params(parameters)
        else:
            msg = "Input `parameters` must be either a brisket.parameters.Params object or python dictionary"
            self.logger.error(msg)
            raise TypeError(msg)

        self.components = self.parameters.components

        # if self.index_list is not None:
        #     self.spec_wavs = self._get_index_spec_wavs(model_components)

        
        # Initialize unit conversion logic
        if isinstance(self.sed_units, str): self.sed_units = utils.unit_parser(self.sed_units)
        if isinstance(self.wav_units, str): self.wav_units = utils.unit_parser(self.wav_units)
        if 'spectral flux density' in list(self.sed_units.physical_type):
            self.logger.debug(f"Converting SED flux units to f_nu ({self.sed_units})")
            self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2 * (1 * self.wav_units)**2 / speed_of_light).to(self.sed_units).value
            self.flam = False
        elif 'spectral flux density wav' in list(self.sed_units.physical_type):
            self.logger.debug(f"Keeping SED flux units in f_lam ({self.sed_units})")
            self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2).to(self.sed_units).value
            self.flam = True
        else:
            self.logger.error(f"Could not determine units for final SED -- input astropy.units ")
            sys.exit()

        if self.spec_output:
            self.spec_wavs_internal = (self.spec_wavs*self.wav_units).to(u.angstrom).value # store an internal spec_wavs array in angstroms
            if isinstance(self.spec_units, str): self.spec_units = utils.unit_parser(self.spec_units)
            if 'spectral flux density' in list(self.spec_units.physical_type): self.spec_flam = False
            elif 'spectral flux density wav' in list(self.spec_units.physical_type): self.spec_flam = True
            if (self.flam and self.spec_flam) or (not self.flam and not self.spec_flam): # easy conversion, just scaling
                self.spec_unit_conv = (1*self.sed_units).to(self.spec_units).value
            elif self.flam and not self.spec_flam: # if SED is in f_lam but spectrum is in f_nu... 
                self.spec_unit_conv = (1*self.sed_units * (1 * self.wav_units)**2 / speed_of_light).to(self.spec_units).value
            elif not self.flam and self.spec_flam: # if SED is in f_nu but spectrum is in f_lam...
                self.spec_unit_conv = (1*self.sed_units / (1 * self.wav_units)**2 * speed_of_light).to(self.spec_units).value
            else:
                self.logger.error(f"Could not determine units for spectrum -- input astropy.units ")
                sys.exit()

        # Create a spectral calibration object to handle spectroscopic calibration and resolution.
        self.calib = False
        if self.spec_output and 'calib' in self.components: 
            self.calib = SpectralCalibrationModel(self.spec_wavs_internal, self.parameters, logger=logger)
        
        # Create a filter_set object to manage the filter curves.
        if self.phot_output:
            if type(filt_list) == filters.filter_set:
                self.filter_set = filt_list
            else:
                self.filter_set = filters.filter_set(filt_list, logger=logger)
        

        # Calculate optimal wavelength sampling for the model
        self.logger.debug('Calculating optimal wavelength sampling for the model...')
        self.wavelengths = self._get_wavelength_sampling()
        self.wav_rest = (self.wavelengths*u.angstrom).to(self.wav_units).value

        if self.phot_output:
            self.logger.debug('Resampling the filter curves onto model wavelength grid...')
            self.filter_set.resample_filter_curves(self.wavelengths)

        self.logger.debug('Initializing IGM absorption model...')
        self.igm = IGMModel(self.wavelengths, self.parameters['igm'])
        
        # Initialize the base parameters -- redshift, igm transmission, luminosity distance, etc
        self._define_base_params_at_redshift() #TODO should take redshift as argument? 
        
        # Initialize the various physical models
        if 'galaxy' in self.components: 
            params = self.parameters['galaxy']
            params['redshift'] = self.parameters['redshift']

            self.galaxy = DotMap(sfh=StarFormationHistoryModel(params, logger=logger), 
                                 stellar=StellarModel(self.wavelengths, params, logger=logger),
                                 nebular=NebularModel(self.wavelengths, params, logger=logger),
                                 dust_atten=DustAttenuationModel(self.wavelengths, params, logger=logger),
                                 dust_emission=DustEmissionModel(self.wavelengths, params, logger=logger))
        if 'agn' in self.components:
            params = self.parameters['agn']
            params['redshift'] = self.parameters['redshift']
            self.agn = DotMap(accdisk=AccretionDiskModel(self.wavelengths, params, logger=logger), 
                              nebular=AGNLineModel(self.wavelengths, params, logger=logger),
                              dust_atten=DustAttenuationModel(self.wavelengths, params, logger=logger))

        if 'nebular' in self.components: # stand-alone nebular model
            self.nebular = NebularModel(self.wavelengths, self.parameters, logger=logger)


        # Compute the full internal model SED
        self._compute_sed() 

        # Compute observables
        if self.phot_output:
            self.photometry = self._compute_photometry(self.sed)

        if self.spec_output:
            self.spectrum = self._compute_spectrum(self.sed)
    
        # if self.prop_output:
            # self._compute_properties()


    def _define_base_params_at_redshift(self):

        ########################## Configure base-level parameters. ##########################
        self.redshift = self.parameters['redshift']
        if self.redshift > config.max_redshift:
            raise ValueError("""Attempted to create a model with too high redshift. 
                                Please increase max_redshift in brisket/config.py 
                                before making this model.""")

        # Compute IGM transmission at the given redshift
        self.igm_trans = self.igm.trans(self.redshift, self.parameters['igm'])
        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.
        if self.redshift > 0.:
            dL = config.cosmo.luminosity_distance(self.redshift).to(u.cm).value
            self.lum_flux = 4*np.pi*dL**2

        # self.damping = damping(self.wavelengths, parameters['base']['damping'])
        # self.MWdust = MWdust(self.wavelengths, components['base']['MWdust'])

        self.wav_obs = self.wav_rest * (1 + self.redshift)



    def update(self, parameters):
        """ Update the model outputs (spectra, photometry) to reflect 
        new parameter values in the parameters dictionary. Note that 
        only the changing of numerical values is supported."""
        self._define_base_params_at_redshift()

        if 'galaxy' in self.components: 
            params = parameters['galaxy']
            params['redshift'] = parameters['redshift']
            self.galaxy.sfh.update(params)
            if self.galaxy.dust_atten:
                self.galaxy.dust_atten.update(params["dust_atten"])
        
        if 'agn' in self.components:
            params = parameters['agn']
            params['redshift'] = parameters['redshift']
            if self.agn.dust_atten:
                self.agn.dust_atten.update(params["dust_atten"])
    
        # Compute the internal full SED 
        self._compute_sed()

        # If photometric output, compute photometry
        if self.phot_output:
            self.photometry = self._compute_photometry(self.sed)

        # If spectroscopic output, compute spectrum
        if self.spec_output:
            self.spectrum = self._compute_spectrum(self.sed)


    def _compute_sed(self):
        """ This method is the primary workhorse for ModelGalaxy. It combines the 
        models for the various emission, attenuation, and absorption processes to 
        generate the internal full galaxy SED held within the class. 
        The _compute_photometry and compute_spectrum methods generate observables 
        using this internal full spectrum. """

        self.sed = np.zeros(len(self.wavelengths))

        if 'nebular' in self.components: 
            self.sed_nebular = self.nebular.spectrum(self.parameters)
            self.sed_nebular[self.wavelengths < 912] = 0
            self.sed += self.sed_nebular

        if 'galaxy' in self.components: 
            params = self.parameters['galaxy']
            #self.galaxy.sfh.update(params) not needed?

            sed_bc, sed = self.galaxy.stellar.spectrum(self.galaxy.sfh.ceh.grid, params['t_bc'])
            em_lines = np.zeros(config.line_wavs.shape)

            if self.galaxy.nebular:
                grid = np.copy(self.galaxy.sfh.ceh.grid)
                if "metallicity" in list(params["nebular"]):
                    nebular_metallicity = params["nebular"]["metallicity"]
                    self.galaxy.neb_sfh.update(params['nebular'])
                    grid = self.galaxy.neb_sfh.ceh.grid

                if "fesc" not in list(params['nebular']):
                    params['nebular']["fesc"] = 0

                # em_lines += self.galaxy.nebular.line_fluxes(grid, params['t_bc'],
                #                                        params['nebular']["logU"])*(1-params['nebular']["fesc"])

                # All stellar emission below 912A goes into nebular emission
                sed_bc[self.wavelengths < 912.] = sed_bc[self.wavelengths < 912.] * params['nebular']["fesc"]
                sed_bc += self.galaxy.nebular.spectrum(params, sfh_ceh=grid)*(1-params['nebular']["fesc"])

            if self.galaxy.dust_atten:
                # Add attenuation due to the diffuse ISM.
                dust_flux = 0.  # Total attenuated flux for energy balance.
                sed_atten = sed * self.galaxy.dust_atten.trans_cont
                dust_flux += np.trapz(sed - sed_atten, x=self.wavelengths)
                sed = sed_atten

                sed_bc_atten = sed_bc * self.galaxy.dust_atten.trans_bc
                dust_flux += np.trapz(sed_bc - sed_bc_atten, x=self.wavelengths)
                sed_bc = sed_bc_atten

                # Add (extra) attenuation to nebular emission lines
                em_lines *= self.galaxy.dust_atten.trans_line


            sed += sed_bc  # We're done treating birthclouds separately -- add birth cloud SED to full SED. 

            if self.galaxy.dust_atten and self.galaxy.dust_emission:
                # TODO: add logic for ignoring energy balance if e.g. L_IR is present in params['dust_emission']
                sed += dust_flux*self.galaxy.dust_emission.spectrum(params)

            # self.line_fluxes = dict(zip(config.line_names, em_lines))

            # Apply IGM and redshifting
            self.sed_galaxy = sed
            self.sed += self.sed_galaxy
        
        if 'agn' in self.components: 
            params = self.parameters['agn']

            sed = self.agn.accdisk.spectrum(params) * (1+self.redshift)**2
            self.sed_accdisk = sed

            if self.agn.nebular:
                # line_names = list(self.nebular.line_names)
                # em_lines = self.nebular.line_fluxes(model_comp['nebular'])
                i_norm = np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))
                nebular_sed = self.agn.nebular.spectrum(params) * sed[i_norm]
                # blr_spectrum = self.agn_lines.blr * spectrum[np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))]
                # nlr_spectrum = self.agn_lines.nlr * spectrum[np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))]
                sed += nebular_sed

            # Add attenuation
            if self.agn.dust_atten:
                sed_atten = sed * self.agn.dust_atten.trans_cont
                sed = sed_atten
                # if self.nebular:
                #     trans2 = 10**(-0.4*Alam*model_comp['dust_atten']['eta_nlr'])
                #     spectrum = (spectrum-nlr_spectrum) * trans + nlr_spectrum*trans2 + scat
                # else:
            
                # Add dust emission.

            self.sed_agn = sed
            self.sed += self.sed_agn

        # # Optionally divide the model by a polynomial for calibration.
        # if "calib" in list(self.fit_instructions):
        #     self.calib = calib_model(self.model_components["calib"],
        #                              self.galaxy.spectrum,
        #                              self.model_galaxy.spectrum)

        #     model = self.model_galaxy.spectrum[:, 1]/self.calib.model

        # else:
        #     model = self.model_galaxy.spectrum[:, 1]


        if self.flam: 
            unit_conv = self.sed_unit_conv 
        else:
            unit_conv = self.sed_unit_conv * self.wav_obs**2
        
        self.sed *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))
        if 'galaxy' in self.components: 
            self.sed_galaxy *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))
        if 'agn' in self.components: 
            self.sed_accdisk *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))
            self.sed_agn *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))
        if 'nebular' in self.components: 
            self.sed_nebular *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))
            for line in self.nebular.line_grid:
                self.nebular.line_grid[line] *= unit_conv * self.igm_trans / (self.lum_flux * (1+self.redshift))

    def _get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """

        R_spec = config.R_spec
        if self.calib:
            # we don't want to generate a model on a coarser grid than we are observing it
            self.R_curve = self.calib.R_curve
            if self.R_curve is not None:
                R_spec = int(4*np.max(self.R_curve[:,1]))

        R_phot, R_other = config.R_phot, config.R_other

        max_z = config.max_redshift
        max_wav = config.max_wavelength.to(u.angstrom).value

        # if neither spectral or photometric output is desired, just compute the full spectrum at resolution R_other
        if not self.spec_output and not self.phot_output:
            self.max_wavs = [max_wav]
            self.R = [R_other]
            
        # if only photometric output is desired, compute spectrum at resolution R_phot in the range of the photometric data, and R_other elsewhere
        elif not self.spec_output:
            self.max_wavs = [self.filter_set.min_phot_wav/(1+max_z), 1.01*self.filter_set.max_phot_wav, max_wav]
            self.R = [R_other, R_phot, R_other]

        # if only spectral output is desired, compute spectrum at resolution R_spec in the range of the spectrum, and R_other elsewhere
        elif not self.phot_output:
            self.max_wavs = [self.spec_wavs_internal[0]/(1+max_z), self.spec_wavs_internal[-1], max_wav]
            self.R = [R_other, R_spec, R_other]

        # if both are desired, more complicated logic is necessary
        else:
            if (self.spec_wavs_internal[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs_internal[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs_internal[0]/(1.+max_z),
                                 self.spec_wavs_internal[-1],
                                 self.filter_set.max_phot_wav, max_wav]

                self.R = [R_other, R_phot, R_spec,
                          R_phot, R_other]

            elif (self.spec_wavs_internal[0] < self.filter_set.min_phot_wav
                  and self.spec_wavs_internal[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.spec_wavs_internal[0]/(1.+max_z),
                                 self.spec_wavs_internal[-1],
                                 self.filter_set.max_phot_wav, max_wav]

                self.R = [R_other, R_spec,
                          R_phot, R_other]

            elif (self.spec_wavs_internal[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs_internal[-1] > self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs_internal[0]/(1.+max_z),
                                 self.spec_wavs_internal[-1], max_wav]

                self.R = [R_other, R_phot,
                          R_spec, R_other]
            
            elif (self.spec_wavs_internal[0] < self.filter_set.min_phot_wav
                    and self.spec_wavs_internal[-1] > self.filter_set.max_phot_wav):
                self.max_wavs = [self.spec_wavs_internal[0]/(1+max_z), self.spec_wavs_internal[-1], max_wav]
                self.R = [R_other, R_spec, R_other]

        # Generate the desired wavelength sampling.
        x = [1.]

        for i in range(len(self.R)):
            if i == len(self.R)-1 or self.R[i] > self.R[i+1]:
                while x[-1] < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

            else:
                while x[-1]*(1.+0.5/self.R[i]) < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

        return np.array(x)


    def _compute_photometry(self, sed):
        """ This method generates predictions for observed photometry.
        It resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """

        # if uvj:
        #     phot = self.uvj_filter_set.get_photometry(self.spectrum_full,
        #                                               redshift,
        #                                               unit_conv=unit_conv)

        # else:
        phot = self.filter_set.get_photometry(sed, self.redshift)#output_units=self.phot_units)
        return phot

    def _compute_spectrum(self, sed):
        """ This method generates predictions for observed spectroscopy.
        It optionally applies a Gaussian velocity dispersion then
        resamples onto the specified set of observed wavelengths. """

        # if "veldisp" in self.parameters:
        #     vres = 3*10**5/config.R_spec/2.
        #     sigma_pix = model_comp["veldisp"]/vres
        #     k_size = 4*int(sigma_pix+1)
        #     x_kernel_pix = np.arange(-k_size, k_size+1)

        #     kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
        #     kernel /= np.trapz(kernel)  # Explicitly normalise kernel

        #     spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
        #     wav_obs = (1+self.redshift) * self.wavelengths[k_size:-k_size]

        # else:
        wav_obs = (1+self.redshift) * self.wavelengths

        if self.calib:
            if self.calib.R_curve is not None:
                self.logger.debug(f"Convolving output spectrum using provided R_curve")
                wav_obs, sed = self.calib.convolve_R_curve(wav_obs, sed, self.parameters['calib']['f_LSF'])

        spectrum = spectres.spectres(self.spec_wavs_internal, wav_obs, sed, fill=0)


        if (self.flam and self.spec_flam) or (not self.flam and not self.spec_flam): # easy conversion, just scaling
            spectrum *= self.spec_unit_conv
        elif self.flam and not self.spec_flam: # if SED is in f_lam but spectrum is in f_nu... 
            spectrum *= self.spec_unit_conv * self.spec_wavs**2
        elif not self.flam and self.spec_flam: # if SED is in f_nu but spectrum is in f_lam...
            spectrum *= self.spec_unit_conv / self.spec_wavs**2

        # if self.spec_units == "mujy":
        #     fluxes /= ((10**-29*2.9979*10**18/self.spec_wavs**2))

        # self.spectrum = np.c_[self.spec_wavs, fluxes]
        return spectrum

    def _calculate_uvj_mags(self):
        """ Obtain (unnormalised) rest-frame UVJ magnitudes. """

        self.uvj = -2.5*np.log10(self.compute_photometry(0., uvj=True))

    # def plot(self, show=True):
    #     return plotting.plot_model_galaxy(self, show=show)

    # def plot_full_spectrum(self, show=True):
    #     return plotting.plot_full_spectrum(self, show=show)

    def _compute_properties(self): 

        self.properties = {}

        self.properties['redshift'] = self.redshift
        self.properties['t_hubble'] = config.age_at_z(self.redshift)

        # full SEDs (and spec, phot) for each component
        self.properties['SED'] = self.sed
        if self.phot_output: self.properties['photometry'] = self._compute_photometry(self.sed)
        if self.spec_output: self.properties['spectrum'] = self._compute_spectrum(self.sed)

        if len(self.components) >= 2:
            if 'galaxy' in self.components: 
                self.properties['SED_galaxy'] = self.sed_galaxy
                if self.phot_output: self.properties['phot_galaxy'] = self._compute_photometry(self.sed_galaxy)
                if self.spec_output: self.properties['spec_galaxy'] = self._compute_spectrum(self.sed_galaxy)
            if 'agn' in self.components: 
                self.properties['SED_accdisk'] = self.sed_accdisk
                self.properties['SED_AGN'] = self.sed_agn
                if self.phot_output: self.properties['phot_AGN'] = self._compute_photometry(self.sed_agn)
                if self.spec_output: self.properties['spec_AGN'] = self._compute_spectrum(self.sed_agn)
            if 'nebular' in self.components: 
                self.properties['SED_nebular'] = self.sed_nebular
                for line in self.nebular.line_grid:
                    self.properties[f'SED_nebular_{line}'] = self.nebular.line_grid[line]
                if self.phot_output: self.properties['phot_nebular'] = self._compute_photometry(self.sed_nebular)
                if self.spec_output: 
                    self.properties['spec_nebular'] = self._compute_spectrum(self.sed_nebular)
                    for line in self.nebular.line_grid:
                        self.properties[f'spec_nebular_{line}'] = self._compute_spectrum(self.nebular.line_grid[line])


        if 'galaxy' in self.components: 
            for q in ['stellar_mass', 'formed_mass', 'SFR_10', 'sSFR_10', 'nSFR_10', 'SFR_100', 'sSFR_100', 'nSFR_100', 'mass_weighted_age', 't_form', 't_quench']:
                self.properties[q] = getattr(self.galaxy.sfh, q)

            self.properties['SFH_ages'] = self.galaxy.sfh.ages
            self.properties['SFH_redshifts'] = config.z_at_age(self.properties['t_hubble']-self.properties['SFH_ages'])
            self.properties['SFH'] = self.galaxy.sfh.sfh


        # emission line fluxes, equivalent widths 
        # self.line_fluxes
        # self.line_EWs

        tophat = np.array((self.wavelengths > 1450)&(self.wavelengths < 1550),dtype=bool)
        if self.flam: 
            sed_flam = (self.sed*self.sed_units).to(u.erg/u.s/u.cm**2/u.angstrom).value
            sed_fnu = (self.sed*self.sed_units * (self.wav_obs * self.wav_units)**2 / speed_of_light).to(u.Jy).value
        else: 
            sed_flam = (self.sed*self.sed_units * speed_of_light / (self.wav_obs * self.wav_units)**2).to(u.erg/u.s/u.cm**2/u.angstrom).value
            sed_fnu = (self.sed*self.sed_units).to(u.Jy).value

        mUV = -2.5*np.log10(np.mean(sed_fnu[tophat])/(1+self.redshift)/3631)
        dL = config.cosmo.luminosity_distance(self.redshift).to(u.pc).value
        MUV = mUV - 5*(np.log10(dL)-1)
        self.properties['m_UV'] = round(mUV,4)
        self.properties['M_UV'] = round(MUV,4)
        
        # calzetti 1994 wavelength windows 
        windows = np.zeros(len(self.wavelengths),dtype=bool)
        windows |= (self.wavelengths>=1268)&(self.wavelengths<=1284)
        windows |= (self.wavelengths>=1309)&(self.wavelengths<=1316)
        windows |= (self.wavelengths>=1342)&(self.wavelengths<=1371)
        windows |= (self.wavelengths>=1407)&(self.wavelengths<=1515)
        windows |= (self.wavelengths>=1562)&(self.wavelengths<=1583)
        windows |= (self.wavelengths>=1677)&(self.wavelengths<=1740)
        windows |= (self.wavelengths>=1760)&(self.wavelengths<=1833)
        windows |= (self.wavelengths>=1866)&(self.wavelengths<=1890)
        windows |= (self.wavelengths>=1930)&(self.wavelengths<=1950)
        windows |= (self.wavelengths>=2400)&(self.wavelengths<=2580)
        p = np.polyfit(np.log10(self.wavelengths[windows]), np.log10(sed_flam[windows]), deg=1)
        self.properties['beta_UV'] = round(p[0],4)
        # beta_opt?

        # dust_curve
        # UVJ mags

        # # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        # uvj_filt_list = np.loadtxt(utils.install_dir
        #                            + "/filters/UVJ.filt_list", dtype="str")
        # self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        # self.uvj_filter_set.resample_filter_curves(self.wavelengths)


    def save_output(self, outfile, overwrite=True):
        assert hasattr(self, properties)

        if outfile.endswith('.fits'):
            from astropy.io import fits
            self.logger.info(f'Saving model output to {outfile}')
            tables = [fits.PrimaryHDU(header=fits.Header({'EXTEND':True}))]

            columns = []
            columns.append(fits.Column(name='wav_rest', array=self.wavelengths, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='wav_obs', array=self.wavelengths_obs, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='flux', array=self.spectrum_full, format='D', unit=str(self.sed_units)))
            tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'SED'})))

            if self.phot_output:
                columns = []
                columns.append(fits.Column(name='wav_obs', array=self.filter_set.eff_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_min', array=self.filter_set.min_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_max', array=self.filter_set.max_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='flux', array=self.photometry, format='D', unit=str(self.phot_units)))
                # tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'PHOT'})))

            t = fits.HDUList(tables)
            t.writeto(outfile, overwrite=overwrite)