# '''
# Wrapper for the SFH and ZDist modules in synthesizer.
# Expands to add various SFH models that observers might want to use, 
# such as continuity SFHs, or SFHs with stochastic bursts.

# '''


# from synthesizer.parametric import ZDist, SFH

# DeltaConstant = ZDist.DeltaConstant
# Normal = ZDist.Normal



# def get(name):
#     '''
#     Get the SFH or ZDist model by name
#     '''
#     if name == 'DeltaConstant':
#         return DeltaConstant

#     elif name == 'constant':
#         return SFH.Constant

#     else:
#         raise ValueError(f"Unknown model: {name}")

# ZH + SFH = SFZH



'''
SFH models
'''
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve
from copy import copy, deepcopy


sfh_age_log_sampling=0.0025

from .. import utils

# from .. import config

# TODO define sfh_log_age_sampling in config.py == 0.0025

def lognorm_equations(p, consts):
    """ 
    Function computing tau and t0 for a lognormal SFH given
    some tmax and fwhm. Needed to transform variables.
    """

    tau_solve, T0_solve = p

    xmax, h = consts

    tau = np.exp(T0_solve - tau_solve**2) - xmax
    t0 = xmax*(np.exp(0.5*np.sqrt(8*np.log(2)*tau_solve**2))
               - np.exp(-0.5*np.sqrt(8*np.log(2)*tau_solve**2))) - h

    return (tau, t0)

class SFZHModel:
    """
    A model that combines a star formation history (SFH) and a metallicity distribution (ZH) model.

    """

    def __init__(self, SFH, ZH, grid_live_frac):
        self.SFH = SFH
        self.ZH = ZH
        self.grid_live_frac = grid_live_frac

    def get_weights(self, params): 
        sfh_params = params.get_sfh()
        zh_params = params.get_zh()
        sfh_weights = self.SFH.get_weights(sfh_params)
        zh_weights = self.ZH.get_weights(zh_params) # TODO ZH could take SFH weights as input, if needed, to do joint models
        combined_weights = np.expand_dims(sfh_weights, axis=-1) * np.expand_dims(zh_weights, axis=-2)
        live_weights = self.grid_live_frac * combined_weights

        # Normalise to 1 solar mass (current)
        Mstar = np.power(10., params['logMstar'])
        if np.ndim(Mstar) == 0: # not vectorized
            Mstar_norm = np.sum(live_weights)
            combined_weights *= Mstar/Mstar_norm
        else:
            Mstar_norm = live_weights.reshape(-1, params.n_vector).sum(axis=0)
            norm_factor = Mstar/Mstar_norm
            combined_weights *= norm_factor[:, np.newaxis, np.newaxis]

        return combined_weights

        # self._calculate_derived_quantities()

class BaseSFHModel:
    """
    Base class for all star formation history (SFH) mdoels.

    Args:
        params (brisket.parameters.Params)
            Model parameters.

    Attributes: 
        ages (np.ndarray)
            Ages of stellar populations in the star formation history, in Gyr.
    """
    def __init__(self, log10age, params, verbose=False):
        self.vectorized = True # whether the model is vectorized 
        
        # Set up the age sampling for internal SFH calculations.
        log_age_min, log_age_max = np.min(log10age), np.max(log10age)
        self.ages = np.arange(6., log_age_max, sfh_age_log_sampling)
        # print(np.min(self.ages), np.max(self.ages))
        # print(np.min(log10age), np.max(log10age))
        self.age_bins = utils.make_bins(self.ages, fix_low=-99)
        self.ages = np.power(10., self.ages)
        self.age_bins = np.power(10., self.age_bins)
        self.age_widths = np.diff(self.age_bins)

        self.grid_age_bins = np.power(10., utils.make_bins(log10age, fix_low=-99))
        # self.sfh = np.zeros_like(self.ages)
        # self.grid = self.parent.grid
        # self.weights = np.zeros_like(self.grid.age)
        # self.ceh = ChemicalEnrichmentHistoryModel(self.parent.params, grid=self.grid)

    def get_weights(self, params):
        # Sum up contributions to each age bin to create SSP weights
        self.sfh = self._sfr(params)
        if np.ndim(self.sfh) == 1: # not vectorized
            weights, _ = np.histogram(self.ages, bins=self.grid_age_bins, weights=self.sfh * self.age_widths)
        else:
            n_vector = np.shape(self.sfh)[0]
            weights = np.zeros((n_vector, len(self.grid_age_bins)-1))

            indices = np.digitize(self.ages, self.grid_age_bins)
            for i in range(n_vector):
                weights[i] = np.bincount(indices-1, weights=self.sfh[i] * self.age_widths)
            
            # for i, index in enumerate(np.unique(indices)):
            #     weights[:,i] = np.sum(indices==index, axis=1)

            # for i in range(n_vector):
            #     weights[i], _ = np.histogram(self.ages, bins=self.grid_age_bins, weights=self.sfh[i] * self.age_widths)
            # ages, sample = np.meshgrid(self.ages, np.arange(n_vector))
            # ages, sample = ages.flatten(), sample.flatten()
            # sample_bins = np.append(np.arange(n_vector)-0.5, n_vector-0.5)
            # weights, _, _ = np.histogram2d(ages, sample, bins=(self.grid_age_bins, sample_bins), weights=(self.sfh * self.age_widths).flatten()) 
            # weights = weights.T
        
        return weights


class BaseZHModel:
    """
    Base class for all metallicity distribution (ZH) models.
    """
    def __init__(self, metallicity, verbose=False):
        # Set up the age sampling for internal SFH calculations.
        self.zmet_vals = metallicity

    def __add__(self, other):
        if not isinstance(other, BaseSFHModel):
            raise ValueError("Can only add SFH models to ZH models.")
        return SFZHModel(self, other)

    def _zmet(self, ages, params):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_weights(self, params):
        """ Delta function metallicity history. Currently the default (and only) implemented chemical enrichment history. """
        zmet = params["zmet"]

        if np.ndim(zmet) == 0:
            weights = np.zeros_like(self.zmet_vals)

            high_ind = np.searchsorted(self.zmet_vals, zmet)
            low_ind = high_ind - 1
            width = (self.zmet_vals[high_ind] - self.zmet_vals[low_ind])
            weights[high_ind] = (zmet - self.zmet_vals[low_ind])/width
            weights[high_ind-1] = 1 - weights[high_ind]

        else:
            # zmet is a 1D array, we need to generate weights /for each value/
            # weights is a 2D array, with each row corresponding to the weights for a given zmet value
            weights = np.zeros((len(zmet), len(self.zmet_vals)))
            high_inds = np.sum(self.zmet_vals[:,np.newaxis] < zmet, axis=0)
            low_inds = high_inds - 1

            widths = (self.zmet_vals[high_inds] - self.zmet_vals[low_inds])
            high_weights = (zmet - self.zmet_vals[low_inds])/widths
            weights[np.arange(len(zmet)), high_inds] = high_weights
            weights[np.arange(len(zmet)), low_inds] = 1 - high_weights
    
        return weights






class BurstSFH(BaseSFHModel):
    """A delta function burst of star-formation."""

    def __init__(self, log10age, params):
        super().__init__(log10age, params)
        self._validate_params(params)

    def _validate_params(self, params):
        if "age" not in params:
            raise ValueError("Burst SFH model requires an age")

    def _sfr(self, params):
        age = params["age"]*1e9
        if np.ndim(age) == 0: # not vectorized
            sfr = np.zeros(len(self.ages))
            sfr[np.argmin(np.abs(self.ages - age))] += 1
        else:
            sfr = np.zeros((len(age),len(self.ages)))
            ind = np.argmin(np.abs(age[:,np.newaxis] - self.ages),axis=1)
            sfr[np.arange(len(age)), ind] += 1
        return sfr

class ConstantSFH(BaseSFHModel):
    """ Constant star-formation between some limits. """
    # def __init__(self, params, parent):
    #     self._build_defaults(params)
    #     super().__init__(params, parent)

    # def _build_defaults(self, params):
    #     pass

    def sfr(self, ages, params):
        sfr = np.zeros_like(ages)
        age_min = float(params['age_min'])*1e9
        age_max = float(params['age_max'])*1e9

        mask = (ages > age_min) & (ages < age_max)
        sfr[mask] += 1.
        return sfr

class ExponentialSFH(BaseSFHModel):
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        pass

    def sfr(self, ages, param):
        sfr = np.zeros_like(ages)
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
        return sfr


class RisingExponentialSFH(BaseSFHModel):
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        pass

    def sfr(self, ages, param):
        sfr = np.zeros_like(ages)
        if "age" in list(param):
            age = param["age"]*10**9
        else:
            age = (param["tstart"] - self.age_of_universe)*10**9
        if "tau" in list(param):
            tau = param["tau"]*10**9
        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(t/tau)
        return sfr


class DelayedSFH(BaseSFHModel):
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        pass

    def sfr(self, ages, param):
        sfr = np.zeros_like(ages)

        age = param["age"]*10**9
        tau = param["tau"]*10**9

        t = age - ages[ages < age]

        sfr[ages < age] = t*np.exp(-t/tau)
        return sfr


    # def const_exp(self, sfr, param):

    #     age = param["age"]*10**9
    #     tau = param["tau"]*10**9

    #     t = age - self.ages[self.ages < age]

    #     sfr[self.ages < age] = np.exp(-t/tau)
    #     sfr[(self.ages > age) & (self.ages < self.age_of_universe)] = 1.

    # def lognormal(self, sfr, param):
    #     if "tmax" in list(param) and "fwhm" in list(param):
    #         tmax, fwhm = param["tmax"]*10**9, param["fwhm"]*10**9

    #         tau_guess = fwhm/(2*tmax*np.sqrt(2*np.log(2)))
    #         t0_guess = np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)

    #         tau, t0 = fsolve(lognorm_equations, (tau_guess, t0_guess),
    #                          args=([tmax, fwhm]))

    #     else:
    #         tau, t0 = par_dict["tau"], par_dict["t0"]

    #     mask = self.ages < self.age_of_universe
    #     t = self.age_of_universe - self.ages[mask]

    #     sfr[mask] = ((1./np.sqrt(2.*np.pi*tau**2))*(1./t)
    #                  * np.exp(-(np.log(t) - t0)**2/(2*tau**2)))

    # def dblplaw(self, sfr, param):
    #     alpha = param["alpha"]
    #     beta = param["beta"]
    #     tau = param["tau"]*10**9

    #     mask = self.ages < self.age_of_universe
    #     t = self.age_of_universe - self.ages[mask]

    #     sfr[mask] = ((t/tau)**alpha + (t/tau)**-beta)**-1

    #     if tau > self.age_of_universe:
    #         self.unphysical = True

    # def iyer2019(self, sfr, param):
    #     tx = param["tx"]
    #     iyer_param = np.hstack([10., np.log10(param["sfr"]), len(tx), tx])
    #     iyer_sfh, iyer_times = db.tuple_to_sfh(iyer_param, self.redshift)
    #     iyer_ages = self.age_of_universe - iyer_times[::-1]*10**9

    #     mask = self.ages < self.age_of_universe
    #     sfr[mask] = np.interp(self.ages[mask], iyer_ages, iyer_sfh[::-1])

    # def psb_wild2020(self, sfr, param):
    #     """
    #     A 2-component SFH for post-starburst galaxies. An exponential
    #     component represents the existing stellar population before the
    #     starburst, while a double power law makes up the burst.
    #     The weight of mass formed between the two is controlled by a
    #     fburst factor: thefraction of mass formed in the burst.
    #     For more detail, see Wild et al. 2020
    #     (https://ui.adsabs.harvard.edu/abs/2020MNRAS.494..529W/abstract)
    #     """
    #     age = param["age"]*10**9
    #     tau = param["tau"]*10**9
    #     burstage = param["burstage"]*10**9
    #     alpha = param["alpha"]
    #     beta = param["beta"]
    #     fburst = param["fburst"]

    #     ind = (np.where((self.ages < age) & (self.ages > burstage)))[0]
    #     texp = age - self.ages[ind]
    #     sfr_exp = np.exp(-texp/tau)
    #     sfr_exp_tot = np.sum(sfr_exp*self.age_widths[ind])

    #     mask = self.ages < self.age_of_universe
    #     tburst = self.age_of_universe - self.ages[mask]
    #     tau_plaw = self.age_of_universe - burstage
    #     sfr_burst = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    #     sfr_burst_tot = np.sum(sfr_burst*self.age_widths[mask])

    #     sfr[ind] = (1-fburst) * np.exp(-texp/tau) / sfr_exp_tot

    #     dpl_form = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    #     sfr[mask] += fburst * dpl_form / sfr_burst_tot

    # def continuity(self, sfr, param):
    #     bin_edges = np.array(param["bin_edges"])[::-1]*10**6
    #     n_bins = len(bin_edges) - 1
    #     dsfrs = [param["dsfr" + str(i)] for i in range(1, n_bins)]

    #     for i in range(1, n_bins+1):
    #         print(self.ages)
    #         print(bin_edges)
    #         mask = (self.ages < bin_edges[i-1]) & (self.ages > bin_edges[i])
    #         sfr[mask] += 10**np.sum(dsfrs[:i])


    # def custom(self, sfr, param):
    #     history = param["history"]
    #     if isinstance(history, str):
    #         custom_sfh = np.loadtxt(history)

    #     else:
    #         custom_sfh = history

    #     sfr[:] = np.interp(self.ages, custom_sfh[:, 0], custom_sfh[:, 1],
    #                        left=0, right=0)

    #     sfr[self.ages > self.age_of_universe] = 0.

    # def delayed_agefrac(self, sfr, param):
    #     age_frac = param["age_frac"]
    #     age = self.age_of_universe * age_frac
    #     if age < 1e7:
    #         age = 1e7
            
    #     tau = param["tau"]*10**9

    #     t = age - self.ages[self.ages < age]

    #     sfr[self.ages < age] = t*np.exp(-t/tau)


    # def dblplaw_agefrac(self, sfr, param):
    #     alpha = param["alpha"]
    #     beta = param["beta"]
    #     tau_frac = param["tau_frac"]
    #     tau = self.age_of_universe * tau_frac
    #     if tau < 1e8:
    #         tau = 1e8 # minimum tau of 0.1 Gyr

    #     mask = self.ages < self.age_of_universe
    #     t = self.age_of_universe - self.ages[mask]
    #     sfr[mask] = ((t/tau)**alpha + (t/tau)**-beta)**-1

    # TcSFH from Endsley+24
# class TcSFH(BaseSFHModel):




class ContinuitySFH(BaseSFHModel):
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        pass

    def sfr(self, ages, params):
        '''
        * `bin_edges` specifies the first few bins (default=[0, 10, 30, 100])
        * `n_bins` specifies how many total bins you want (default=7)
        * `z_max` specifies the redshift at which star-formation begins (default=20)
        
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
        
        return sfr



class ChemicalEnrichmentHistoryModel(object):
    """
    Base class for chemical enrichment history models.
    """
    def __init__(self, params, grid):
        self.zmet_vals = grid.zmet
        # self.zmet_lims = utils.make_bins(self.zmet_vals, fix_low=0, fix_high=10)

    def compute_weights(self, params, sfh_weights):
        self.weights = self.delta(params, sfh_weights)
        return self.weights

    def delta(self, params, sfh_weights):
        """ Delta function metallicity history. Currently the default (and only) implemented chemical enrichment history. """
        zmet = float(params.parent["zmet"])


        weights = np.zeros_like(self.zmet_vals)
        high_ind = len(self.zmet_vals[self.zmet_vals < zmet])

        if high_ind == len(self.zmet_vals):
            weights[-1] = 1.
        elif high_ind == 0:
            weights[0] = 1.
        else:
            low_ind = high_ind - 1
            width = (self.zmet_vals[high_ind] - self.zmet_vals[low_ind])
            weights[high_ind] = (zmet - self.zmet_vals[low_ind])/width
            weights[high_ind-1] = 1 - weights[high_ind]

        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh_weights, axis=0)

    def exp(self, comp, sfh):
        """ P(Z) = exp(-z/z_mean). Currently no age dependency! """

        mean_zmet = comp["metallicity"]

        weights = np.zeros(self.zmet_vals.shape[0])

        vals_hr = np.arange(0., 10., 0.01) + 0.005

        factors_hr = (1./mean_zmet)*np.exp(-vals_hr/mean_zmet)

        for i in range(weights.shape[0]):
            lowmask = (vals_hr > self.zmet_lims[i])
            highmask = (vals_hr < self.zmet_lims[i+1])
            weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def mass_metallicity(self, comp, sfh):
        """ To be implemented. 
        Assign SSP metallicity/age weights according to the mass-metallicity relation at the given redshift, i.e.
        P(Z,t) = P(M=M(t),Z,t)
        TBD how to fix metallicity at t=0. 
        """
        pass
        # zmet = comp["metallicity"]
        # weights = np.zeros(self.zmet_vals.shape[0])
        # return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)


    #TODO add non-delta function metallicity distributions? 