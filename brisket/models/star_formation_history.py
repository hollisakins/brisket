# from __future__ import print_function, division, absolute_import


# try:
#     import dense_basis as db

# except ImportError:
#     pass

import numpy as np
from scipy.optimize import fsolve
from copy import copy, deepcopy

from brisket import utils
from brisket import config
# from .. import plotting

from brisket.models.chemical_enrichment_history import ChemicalEnrichmentHistoryModel


def lognorm_equations(p, consts):
    """ Equations for finding the tau and T0 for a lognormal SFH given
    some tmax and FWHM. Needed to transform variables. """

    tau_solve, T0_solve = p

    xmax, h = consts

    tau = np.exp(T0_solve - tau_solve**2) - xmax
    t0 = xmax*(np.exp(0.5*np.sqrt(8*np.log(2)*tau_solve**2))
               - np.exp(-0.5*np.sqrt(8*np.log(2)*tau_solve**2))) - h

    return (tau, t0)


class StarFormationHistoryModel:
    """ Generate a star formation history.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the star formation
        history you wish to generate.

    log_sampling : float - optional
        the log of the age sampling of the SFH, defaults to 0.0025.
    """

    def __init__(self, model_components, log_sampling=0.0025, logger=utils.NullLogger):

        self.logger = logger
        self.hubble_time = config.age_at_z(0)

        model = model_components['stellar_model']
        self.logger.info(f"Initializing star-formation history module".ljust(50) + f"(sfh: {model_components['sfh']})".rjust(20))
        
        self.template_metallicities = config.stellar_models[model]['metallicities']
        self.template_raw_stellar_ages = config.stellar_models[model]['raw_stellar_ages']
        self.template_live_frac = config.stellar_models[model]['live_frac']

        # Set up the age sampling for internal SFH calculations.
        log_age_max = np.log10(self.hubble_time)+9. + 2*log_sampling
        self.ages = np.arange(6., log_age_max, log_sampling)
        self.age_lhs = utils.make_bins(self.ages, make_rhs=True)[0]
        self.ages = 10**self.ages
        self.age_lhs = 10**self.age_lhs
        self.age_lhs[0] = 0.
        self.age_lhs[-1] = 10**9*self.hubble_time
        self.age_widths = self.age_lhs[1:] - self.age_lhs[:-1]

        # Detect SFH components
        comp_list = list(model_components)
        self.components = ([k for k in comp_list if k in dir(self)]
                           + [k for k in comp_list if k[:-1] in dir(self)])

        self.component_sfrs = {}  # SFR versus time for all components.
        self.component_weights = {}  # SSP weights for all components.

        self._resample_live_frac_grid()

        self.update(model_components)

    def update(self, model_components):

        self.model_components = model_components
        self.redshift = self.model_components["redshift"]

        self.sfh = np.zeros_like(self.ages)  # Star-formation history

        self.unphysical = False
        self.age_of_universe = 10**9 * config.age_at_z(self.redshift)

        # Calculate the star-formation history
        func = model_components['sfh']

        if func not in dir(self):
            raise Exception('Missing SFH')
            #func = name[:-1]

        self.component_sfrs[func] = np.zeros_like(self.ages)
        self.component_weights[func] = np.zeros_like(config.age_sampling)

        getattr(self, func)(self.component_sfrs[func],
                            self.model_components)

        # Normalise to the correct mass.
        mass_norm = np.sum(self.component_sfrs[func]*self.age_widths)
        desired_mass = 10**self.model_components["logMstar"]

        self.component_sfrs[func] *= desired_mass/mass_norm
        self.sfh += self.component_sfrs[func]

        # Sum up contributions to each age bin to create SSP weights
        weights = self.component_sfrs[func]*self.age_widths
        self.component_weights[func] = np.histogram(self.ages,
                                                    bins=config.age_bins,
                                                    weights=weights)[0]

                                                    
        # Check no stars formed before the Big Bang.
        if self.sfh[self.ages > self.age_of_universe].max() > 0.:
            self.unphysical = True

        # ceh: Chemical enrichment history object
        self.ceh = ChemicalEnrichmentHistoryModel(self.model_components,
                                                  self.component_weights)

        self._calculate_derived_quantities()

    def _calculate_derived_quantities(self):
        self.stellar_mass = np.log10(np.sum(self.live_frac_grid*self.ceh.grid))
        self.formed_mass = np.log10(np.sum(self.ceh.grid))

        age_mask = self.ages < 1e8 # 100 Myr
        self.SFR_100 = np.sum(self.sfh[age_mask]*self.age_widths[age_mask])/np.sum(self.age_widths[age_mask])
        self.sSFR_100 = np.log10(self.SFR_100) - self.stellar_mass
        self.nSFR_100 = np.log10(self.SFR_100*self.age_of_universe) - self.formed_mass
        
        age_mask = self.ages < 1e7 # 10 Myr
        self.SFR_10 = np.sum(self.sfh[age_mask]*self.age_widths[age_mask])/np.sum(self.age_widths[age_mask])
        self.sSFR_10 = np.log10(self.SFR_10) - self.stellar_mass
        self.nSFR_10 = np.log10(self.SFR_10*self.age_of_universe) - self.formed_mass

        self.mass_weighted_age = np.sum(self.sfh*self.age_widths*self.ages)
        self.mass_weighted_age /= np.sum(self.sfh*self.age_widths)

        self.t_form = self.age_of_universe - self.mass_weighted_age

        self.t_form *= 10**-9
        self.mass_weighted_age *= 10**-9

        mass_assembly = np.cumsum(self.sfh[::-1]*self.age_widths[::-1])[::-1]
        tunivs = self.age_of_universe - self.ages
        mean_sfrs = mass_assembly/tunivs
        normed_sfrs = np.zeros_like(self.sfh)
        sf_mask = (self.sfh > 0.)
        normed_sfrs[sf_mask] = self.sfh[sf_mask]/mean_sfrs[sf_mask]

        if self.SFR_100 > 0.1*mean_sfrs[0]:
            self.t_quench = 99.

        else:
            quench_ind = np.argmax(normed_sfrs > 0.1)
            self.t_quench = tunivs[quench_ind]*10**-9

    def _resample_live_frac_grid(self):
        self.live_frac_grid = np.zeros((self.template_metallicities.shape[0],
                                        config.age_sampling.shape[0]))

        raw_live_frac_grid = self.template_live_frac

        for i in range(self.template_metallicities.shape[0]):
            self.live_frac_grid[i, :] = np.interp(config.age_sampling,
                                                  self.template_raw_stellar_ages,
                                                  raw_live_frac_grid[:, i])

    def massformed_at_redshift(self, redshift):
        t_hubble_at_z = 10**9 * config.age_at_z(redshift)

        mass_assembly = np.cumsum(self.sfh[::-1]*self.age_widths[::-1])[::-1]

        ind = np.argmin(np.abs(self.ages - (self.age_of_universe - t_hubble_at_z)))

        return np.log10(mass_assembly[ind])

    ###################################################################
    ######### Various choices of star-formation history model #########
    ###################################################################
    def burst(self, sfr, param):
        """ A delta function burst of star-formation. """

        if "age" in list(param):
            age = param["age"]*10**9

        elif "tform" in list(param):
            age = self.age_of_universe - param["tform"]*10**9

        sfr[np.argmin(np.abs(self.ages - age))] += 1

    def constant(self, sfr, param):
        """ Constant star-formation between some limits. """

        if "age_min" in list(param):
            if param["age_max"] == "age_of_universe":
                age_max = self.age_of_universe

            else:
                age_max = param["age_max"]*10**9

            age_min = param["age_min"]*10**9

        else:
            age_max = self.age_of_universe - param["tstart"]*10**9
            age_min = self.age_of_universe - param["tstop"]*10**9

        mask = (self.ages > age_min) & (self.ages < age_max)
        sfr[mask] += 1.

    def exponential(self, sfr, param):

        if "age" in list(param):
            age = param["age"]*10**9

        else:
            age = (param["tstart"] - self.age_of_universe)*10**9

        if "tau" in list(param):
            tau = param["tau"]*10**9

        elif "efolds" in list(param):
            tau = (param["age"]/param["efolds"])*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(-t/tau)


    def rising_exp(self, sfr, param):
        if "age" in list(param):
            age = param["age"]*10**9
        else:
            age = (param["tstart"] - self.age_of_universe)*10**9
        if "tau" in list(param):
            tau = param["tau"]*10**9
        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(t/tau)

    def delayed(self, sfr, param):

        age = param["age"]*10**9
        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = t*np.exp(-t/tau)

    def const_exp(self, sfr, param):

        age = param["age"]*10**9
        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(-t/tau)
        sfr[(self.ages > age) & (self.ages < self.age_of_universe)] = 1.

    def lognormal(self, sfr, param):
        if "tmax" in list(param) and "fwhm" in list(param):
            tmax, fwhm = param["tmax"]*10**9, param["fwhm"]*10**9

            tau_guess = fwhm/(2*tmax*np.sqrt(2*np.log(2)))
            t0_guess = np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)

            tau, t0 = fsolve(lognorm_equations, (tau_guess, t0_guess),
                             args=([tmax, fwhm]))

        else:
            tau, t0 = par_dict["tau"], par_dict["t0"]

        mask = self.ages < self.age_of_universe
        t = self.age_of_universe - self.ages[mask]

        sfr[mask] = ((1./np.sqrt(2.*np.pi*tau**2))*(1./t)
                     * np.exp(-(np.log(t) - t0)**2/(2*tau**2)))

    def dblplaw(self, sfr, param):
        alpha = param["alpha"]
        beta = param["beta"]
        tau = param["tau"]*10**9

        mask = self.ages < self.age_of_universe
        t = self.age_of_universe - self.ages[mask]

        sfr[mask] = ((t/tau)**alpha + (t/tau)**-beta)**-1

        if tau > self.age_of_universe:
            self.unphysical = True

    def iyer2019(self, sfr, param):
        tx = param["tx"]
        iyer_param = np.hstack([10., np.log10(param["sfr"]), len(tx), tx])
        iyer_sfh, iyer_times = db.tuple_to_sfh(iyer_param, self.redshift)
        iyer_ages = self.age_of_universe - iyer_times[::-1]*10**9

        mask = self.ages < self.age_of_universe
        sfr[mask] = np.interp(self.ages[mask], iyer_ages, iyer_sfh[::-1])

    def psb_wild2020(self, sfr, param):
        """
        A 2-component SFH for post-starburst galaxies. An exponential
        component represents the existing stellar population before the
        starburst, while a double power law makes up the burst.
        The weight of mass formed between the two is controlled by a
        fburst factor: thefraction of mass formed in the burst.
        For more detail, see Wild et al. 2020
        (https://ui.adsabs.harvard.edu/abs/2020MNRAS.494..529W/abstract)
        """
        age = param["age"]*10**9
        tau = param["tau"]*10**9
        burstage = param["burstage"]*10**9
        alpha = param["alpha"]
        beta = param["beta"]
        fburst = param["fburst"]

        ind = (np.where((self.ages < age) & (self.ages > burstage)))[0]
        texp = age - self.ages[ind]
        sfr_exp = np.exp(-texp/tau)
        sfr_exp_tot = np.sum(sfr_exp*self.age_widths[ind])

        mask = self.ages < self.age_of_universe
        tburst = self.age_of_universe - self.ages[mask]
        tau_plaw = self.age_of_universe - burstage
        sfr_burst = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
        sfr_burst_tot = np.sum(sfr_burst*self.age_widths[mask])

        sfr[ind] = (1-fburst) * np.exp(-texp/tau) / sfr_exp_tot

        dpl_form = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
        sfr[mask] += fburst * dpl_form / sfr_burst_tot

    # def continuity(self, sfr, param):
    #     bin_edges = np.array(param["bin_edges"])[::-1]*10**6
    #     n_bins = len(bin_edges) - 1
    #     dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

    #     for i in range(1, n_bins+1):
    #         print(self.ages)
    #         print(bin_edges)
    #         mask = (self.ages < bin_edges[i-1]) & (self.ages > bin_edges[i])
    #         sfr[mask] += 10**np.sum(dsfrs[:i])

    def custom(self, sfr, param):
        history = param["history"]
        if isinstance(history, str):
            custom_sfh = np.loadtxt(history)

        else:
            custom_sfh = history

        sfr[:] = np.interp(self.ages, custom_sfh[:, 0], custom_sfh[:, 1],
                           left=0, right=0)

        sfr[self.ages > self.age_of_universe] = 0.

    def delayed_agefrac(self, sfr, param):
        age_frac = param["age_frac"]
        age = self.age_of_universe * age_frac
        if age < 1e7:
            age = 1e7
            
        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = t*np.exp(-t/tau)


    def dblplaw_agefrac(self, sfr, param):
        alpha = param["alpha"]
        beta = param["beta"]
        tau_frac = param["tau_frac"]
        tau = self.age_of_universe * tau_frac
        if tau < 1e8:
            tau = 1e8 # minimum tau of 0.1 Gyr

        mask = self.ages < self.age_of_universe
        t = self.age_of_universe - self.ages[mask]
        sfr[mask] = ((t/tau)**alpha + (t/tau)**-beta)**-1


    def continuity(self, sfr, param):
        '''
        * `bin_edges` specifies the first few bins
        * `n_bins` specifies how many total bins you want
        whatever bins not specified by bin_edges will be log-uniformly spaced from 
        the max age in `bin_edges` to 0.85*t_H at the redshift of the computation
        '''
        bin_edges = np.array(param['bin_edges']) * 1e6

        n_bins_specified = len(bin_edges)-1
        n_bins_even = param['n_bins'] - n_bins_specified
        age_max = 0.85*self.age_of_universe # in yr
        # age_max = 0.85*self.age_of_universe # in yr
        bin_edges = np.append(bin_edges[:-1], np.logspace(np.log10(np.max(bin_edges)), np.log10(age_max), n_bins_even+1))
        bin_edges = np.flip(bin_edges)
        n_bins = len(bin_edges)-1

        sfr[(self.ages < bin_edges[0]) & (self.ages > bin_edges[-1])] = 1
        dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

        for i in range(n_bins):
            mask = (self.ages < bin_edges[i]) & (self.ages > bin_edges[i+1])
            sfr[mask] += 10**np.sum(dsfrs[:i])




    def plot(self, show=True):
        return plotting.plot_sfh(self, show=show)
