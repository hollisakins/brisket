import numpy as np
import warnings
import os, time
from copy import deepcopy
from dotmap import DotMap

from astropy.constants import c as speed_of_light

import logging

from .. import utils
from .. import config
from .. import filters
# from .. import plotting

from .stellar_model import stellar
from .dust_emission_model import dust_emission
from .dust_attenuation_model import dust_attenuation
from .nebular_model import nebular
from .accretion_disk_model import accretion_disk
from .agn_line_model import agn_lines


from .igm_model import igm
from .star_formation_history import star_formation_history
# from ..input.spectral_indices import measure_index
import astropy.units as u

class galaxy_component(object):
    def __init__(self, wavelengths, parameters):
        pass

class agn_component(object):
    def __init__(self, wavelengths, parameters):
        pass

class model_galaxy(object):
    def __init__(self, parameters, 
                 filt_list=None,
                 spec_wavs=None,
                 wav_units=u.um,
                 sed_units=u.uJy,
                 spec_units=u.erg/u.s/u.cm**2/u.angstrom,
                 phot_units=u.uJy,
                 logger=utils.NullLogger):

        ## separate function to parse parameters/parameter file?
        self.logger = logger
        self.spec_wavs = spec_wavs
        self.filt_list = filt_list
        self.wav_units = wav_units
        self.sed_units = sed_units
        self.spec_units = spec_units
        self.phot_units = phot_units
        
        if type(parameters)==str:
            self.logger.info(f'Loading parameters from file {parameters}')
            # parameter file input
            import toml
            parameters = toml.load(os.path.join(config.working_dir, parameters))

        elif type(parameters)==dict:
            self.logger.info(f'Loading parameter dictionary')            
            pass
        else:
            self.logger.error("Input `parameters` must be either python dictionary or str path to TOML parameter file")
            raise TypeError("Input `parameters` must be either python dictionary or str path to TOML parameter file")


        # if self.index_list is not None:
        #     self.spec_wavs = self._get_index_spec_wavs(model_components)

        # Create a filter_set object to manage the filter curves.
        if filt_list is not None:
            if type(filt_list) == list:
                self.filter_set = filters.filter_set(filt_list, logger=logger)
            elif type(filt_list) == filters.filter_set:
                self.filter_set = filt_list

        self.logger.info('Calculating optimal wavelength sampling for the model...')
        self.wavelengths = self._get_wavelength_sampling()

        self.logger.info('Resampling the filter curves onto model wavelength grid...')
        if filt_list is not None:
            self.filter_set.resample_filter_curves(self.wavelengths)

        self.igm = igm(self.wavelengths)#, parameters['base']['igm'])
        self.define_base_params_at_redshift(parameters)
        # # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        # uvj_filt_list = np.loadtxt(utils.install_dir
        #                            + "/filters/UVJ.filt_list", dtype="str")
        # self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        # self.uvj_filter_set.resample_filter_curves(self.wavelengths)

        ######################################################################################
        ########################## Create relevant physical models. ##########################
        ######################################################################################
        
        # initialize each component
        self.components = [c for c in list(parameters.keys()) if 'galaxy' in c or 'agn' in c]
        for component in self.components:
            params = parameters[component]
            params['redshift'] = self.redshift

            if 'galaxy' in component:
                setattr(self, component, DotMap(sfh=star_formation_history(params, logger=logger), 
                                                stellar=stellar(self.wavelengths, params, logger=logger),
                                                nebular=nebular(self.wavelengths, params, logger=logger),
                                                dust_atten=dust_attenuation(self.wavelengths, params, logger=logger),
                                                dust_emission=dust_emission(self.wavelengths, params, logger=logger)))
            if 'agn' in component:
                setattr(self, component, DotMap(accdisk=accretion_disk(self.wavelengths, params, logger=logger), 
                                                nebular=agn_lines(self.wavelengths, params, logger=logger),
                                                dust_atten=dust_attenuation(self.wavelengths, params, logger=logger)))
                    

        self.compute_sed(parameters) 

        if self.filt_list is not None:
            self.compute_photometry(self.redshift)

        if self.spec_wavs is not None:
            self.compute_spectrum(parameters)

    def define_base_params_at_redshift(self, parameters):
        ########################## Configure base-level parameters. ##########################
        self.redshift = parameters['redshift']
        # Compute IGM transmission at the given redshift
        self.igm_trans = self.igm.trans(self.redshift)
        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.
        if self.redshift > 0.:
            dL = config.cosmo.luminosity_distance(self.redshift).to(u.cm).value
            self.lum_flux = 4*np.pi*dL**2

        # self.damping = damping(self.wavelengths, parameters['base']['damping'])
        # self.MWdust = MWdust(self.wavelengths, components['base']['MWdust'])

        if self.redshift > config.max_redshift:
            raise ValueError("""Attempted to create a model with too high redshift. 
                                Please increase max_redshift in brisket/config.py 
                                before making this model.""")


    def update(self, parameters):
        """ Update the model outputs (spectra, photometry) to reflect 
        new parameter values in the parameters dictionary. Note that 
        only the changing of numerical values is supported."""
        self.define_base_params_at_redshift(parameters)

        for component in self.components:
            comp = getattr(self, component)
            parameters[component]['redshift'] = parameters['redshift']
            if 'galaxy' in component:
                comp.sfh.update(parameters[component])
                if comp.dust_atten:
                    comp.dust_atten.update(parameters[component]["dust_atten"])

                # # If the SFH is unphysical do not caclulate the full spectrum
                # if comp.sfh.unphysical:
                #     warnings.warn("The requested model includes stars which formed "
                #                 "before the Big Bang, no spectrum generated.",
                #                 RuntimeWarning)

                #     self.spectrum_full = np.zeros_like(self.wavelengths)
                #     # self.uvj = np.zeros(3)
                # else:
                #     self.compute_sed(parameters[component])
                #     # self._calculate_uvj_mags()
        
            elif 'agn' in component:
                if comp.dust_atten:
                    comp.dust_atten.update(parameters[component]["dust_atten"])
        
        self.compute_sed(parameters)

        if self.filt_list is not None:
            self.compute_photometry(parameters['redshift'])

        if self.spec_wavs is not None:
            self.compute_spectrum(parameters)


    def compute_sed(self, parameters):
        """ This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        spectrum held within the class. The compute_photometry and
        compute_spectrum methods generate observables using this
        internal full spectrum. """

        spectrum_full = np.zeros(len(self.wavelengths))

        for component in self.components:
            comp = getattr(self, component)
            params = parameters[component]

            if 'galaxy' in component: 
                comp.sfh.update(params)

                t_bc = 0.01
                if "t_bc" in list(params):
                    t_bc = params["t_bc"]


                spectrum_bc, spectrum = comp.stellar.spectrum(comp.sfh.ceh.grid, t_bc)
                em_lines = np.zeros(config.line_wavs.shape)

                if comp.nebular:
                    grid = np.copy(comp.sfh.ceh.grid)
                    if "metallicity" in list(params["nebular"]):
                        nebular_metallicity = params["nebular"]["metallicity"]
                        comp.neb_sfh.update(params['nebular'])
                        grid = comp.neb_sfh.ceh.grid

                    if "fesc" not in list(params['nebular']):
                        params['nebular']["fesc"] = 0

                    em_lines += comp.nebular.line_fluxes(grid, t_bc,
                                                        params['nebular']["logU"])*(1-params['nebular']["fesc"])

                    # All stellar emission below 912A goes into nebular emission
                    spectrum_bc[self.wavelengths < 912.] = spectrum_bc[self.wavelengths < 912.] * params['nebular']["fesc"]
                    spectrum_bc += comp.nebular.spectrum(grid, t_bc,
                                                        params['nebular']["logU"])*(1-params['nebular']["fesc"])

                if comp.dust_atten:
                    # Add attenuation due to the diffuse ISM.
                    dust_flux = 0.  # Total attenuated flux for energy balance.
                    trans = 10**(-params["dust_atten"]["Av"]*comp.dust_atten.A_cont/2.5)
                    dust_spectrum = spectrum*trans
                    dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)
                    
                    scat = 0
                    if 'logfscat' in list(params['dust_atten']):
                        scat = spectrum * np.power(10., params['dust_atten']['logfscat'])
                    dust_spectrum += scat

                    spectrum = dust_spectrum

                    # Add (extra) attenuation to stellar birth clouds.
                    eta = 1.
                    if "eta" in list(params["dust_atten"]):
                        eta = params["dust_atten"]["eta"]

                    bc_Av = eta*params["dust_atten"]["Av"]
                    bc_trans = 10**(-bc_Av*comp.dust_atten.A_cont/2.5)
                    spectrum_bc_dust = spectrum_bc*bc_trans
                    dust_flux += np.trapz(spectrum_bc - spectrum_bc_dust, x=self.wavelengths)

                    scat = 0
                    if 'logfscat' in list(params['dust_atten']):
                        scat_bc = spectrum_bc * np.power(10., params['dust_atten']['logfscat'])
                        spectrum_bc_dust += scat_bc

                    ### add self.spectrum_bc
                    spectrum_bc = spectrum_bc_dust

                    # Add (extra) attenuation to nebular emission lines
                    em_lines *= 10**(-bc_Av*comp.dust_atten.A_line/2.5)

                spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.
                if comp.dust_atten and comp.dust_emission:
                    spectrum += dust_flux*comp.dust_emission.spectrum(params)

                # if self.dust_atten:
                #     spectrum_bc /= self.lum_flux*(1. + model_comp["redshift"])

                # em_lines /= self.lum_flux

                # # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
                # spectrum *= 3.826*10**33
                # em_lines *= 3.826*10**33

                # self.line_fluxes = dict(zip(config.line_names, em_lines))
            
            elif 'agn' in component: 
                spectrum = comp.accdisk.spectrum(params)*(1+self.redshift)**2
                # accdisk_spectrum = deepcopy(spectrum)
                if comp.nebular:
                    # line_names = list(self.nebular.line_names)
                    # em_lines = self.nebular.line_fluxes(model_comp['nebular'])
                    nebular_spectrum = comp.nebular.spectrum(params) * spectrum[np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))]
                    # blr_spectrum = self.agn_lines.blr * spectrum[np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))]
                    # nlr_spectrum = self.agn_lines.nlr * spectrum[np.argmin(np.abs(self.wavelengths-config.qsogen_wnrm))]
                    spectrum += nebular_spectrum

                # Add attenuation due to the diffuse ISM.
                if comp.dust_atten:
                    Alam = params["dust_atten"]["Av"]*comp.dust_atten.A_cont
                    trans = 10**(-0.4*Alam)
                    scat = 0
                    if 'logfscat' in list(params['dust_atten']):
                        scat = spectrum * np.power(10., params['dust_atten']['logfscat'])
                    # if self.nebular:
                    #     trans2 = 10**(-0.4*Alam*model_comp['dust_atten']['eta_nlr'])
                    #     spectrum = (spectrum-nlr_spectrum) * trans + nlr_spectrum*trans2 + scat
                    # else:
                    spectrum = spectrum * trans + scat
                # Add dust emission.

            #################################################
            ### regardless of what type of component it is, apply IGM and redshifting

            spectrum *= self.igm_trans 
            spectrum /= self.lum_flux * (1+self.redshift)
            spectrum_full += spectrum
        
        self.spectrum_full = spectrum_full
        ######################### Handle unit-conversions for output #########################
        # self.logger.debug(f"Converting wavelength units from angstroms to {self.wav_units}")
        self.wav_rest = (self.wavelengths*u.angstrom).to(self.wav_units).value
        self.wav_obs = self.wav_rest * (1 + self.redshift)
        if 'spectral flux density' in list(self.sed_units.physical_type):
            self.logger.debug(f"Converting flux units to f_nu ({self.sed_units})")
            self.spectrum_full = (self.spectrum_full*u.Lsun/u.angstrom/u.cm**2 * (self.wav_obs * self.wav_units)**2 / speed_of_light).to(self.sed_units).value

        elif 'spectral flux density wav' in list(self.sed_units.physical_type):
            self.logger.debug(f"Keeping flux units in f_lam ({self.sed_units})")
            self.spectrum_full *= (1*u.Lsun/u.angstrom/u.cm**2).to(self.sed_units).value
        else:
            self.logger.error(f"Could not determine units for final SED -- input astropy.units ")
            sys.exit(1)


    def _get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """

        max_z = config.max_redshift
        max_wav = config.max_wavelength.to(u.angstrom).value

        # if neither spectral or photometric output is desired, just compute the full spectrum at resolution R_other
        if self.spec_wavs is None and self.filt_list is None:
            self.max_wavs = [max_wav]
            self.R = [config.R_other]
            
        # if only photometric output is desired, compute spectrum at resolution R_phot in the range of the photometric data, and R_other elsewhere
        elif self.spec_wavs is None:
            self.max_wavs = [self.filter_set.min_phot_wav/(1+max_z), 1.01*self.filter_set.max_phot_wav, max_wav]
            self.R = [config.R_other, config.R_phot, config.R_other]

        # if only spectral output is desired, compute spectrum at resolution R_spec in the range of the spectrum, and R_other elsewhere
        elif self.filt_list is None:
            self.max_wavs = [self.spec_wavs[0]/(1+max_z), self.spec_wavs[-1], max_wav]
            self.R = [config.R_other, config.R_spec, config.R_other]

        # if both are desired, more complicated logic is necessary
        else:
            if (self.spec_wavs[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, max_wav]

                self.R = [config.R_other, config.R_phot, config.R_spec,
                          config.R_phot, config.R_other]

            elif (self.spec_wavs[0] < self.filter_set.min_phot_wav
                  and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, max_wav]

                self.R = [config.R_other, config.R_spec,
                          config.R_phot, config.R_other]

            if (self.spec_wavs[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs[-1] > self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1], max_wav]

                self.R = [config.R_other, config.R_phot,
                          config.R_spec, config.R_other]

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

    def compute_photometry(self, redshift, uvj=False):
        """ This method generates predictions for observed photometry.
        It resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """

        # if uvj:
        #     phot = self.uvj_filter_set.get_photometry(self.spectrum_full,
        #                                               redshift,
        #                                               unit_conv=unit_conv)

        # else:
        phot = self.filter_set.get_photometry(self.spectrum_full, redshift)#output_units=self.phot_units)
        self.photometry = phot

    def compute_spectrum(self, model_comp):
        """ This method generates predictions for observed spectroscopy.
        It optionally applies a Gaussian velocity dispersion then
        resamples onto the specified set of observed wavelengths. """

        zplusone = self.redshift + 1.

        if "veldisp" in list(model_comp):
            vres = 3*10**5/config.R_spec/2.
            sigma_pix = model_comp["veldisp"]/vres
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
            redshifted_wavs = zplusone*self.wavelengths[k_size:-k_size]

        else:
            spectrum = self.spectrum_full
            redshifted_wavs = zplusone*self.wavelengths

        fluxes = np.interp(self.spec_wavs,
                           redshifted_wavs,
                           spectrum, left=0, right=0)

        if self.spec_units == "mujy":
            fluxes /= ((10**-29*2.9979*10**18/self.spec_wavs**2))

        self.spectrum = np.c_[self.spec_wavs, fluxes]

    def _calculate_uvj_mags(self):
        """ Obtain (unnormalised) rest-frame UVJ magnitudes. """

        self.uvj = -2.5*np.log10(self.compute_photometry(0., uvj=True))

    # def plot(self, show=True):
    #     return plotting.plot_model_galaxy(self, show=show)

    # def plot_full_spectrum(self, show=True):
    #     return plotting.plot_full_spectrum(self, show=show)


    def save_output(self, outfile, overwrite=True):
        if outfile.endswith('.fits'):
            from astropy.io import fits
            self.logger.info(f'Saving model output to {outfile}')
            tables = [fits.PrimaryHDU(header=fits.Header({'EXTEND':True}))]

            columns = []
            columns.append(fits.Column(name='wav_rest', array=self.wavelengths, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='wav_obs', array=self.wavelengths_obs, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='flux', array=self.spectrum_full, format='D', unit=str(self.sed_units)))
            tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'SED'})))

            if 'photometry' in dir(self):
                columns = []
                columns.append(fits.Column(name='wav_obs', array=self.filter_set.eff_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_min', array=self.filter_set.min_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_max', array=self.filter_set.max_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='flux', array=self.photometry, format='D', unit=str(self.phot_units)))
                # tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'PHOT'})))

            t = fits.HDUList(tables)
            t.writeto(outfile, overwrite=overwrite)

        # elif outfile.endswith('.h5') or outfile.endswith('.hdf5'):
        #     import h5py
        #     with h5py.File(outfile, mode='a') as db:
        #         ds = db.create_dataset('SED', data=filt)
        #         ds.attrs['description'] = args.description
        #         ds.attrs['nicknames'] = args.nickname
                





        # import deepdish as dd
        # import warnings



        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     dd.io.save(outfile, self.results)