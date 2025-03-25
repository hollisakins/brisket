'''
Misc utility functions
'''
from __future__ import annotations

import sys, os
import numpy as np
# import astropy.units as u

# from ..config import cosmo
# z_array = np.arange(0., 100., 0.01)
# age_array = cosmo.age(z_array).value
# def age_at_z(z):
#     '''Returns the age of the universe, in Gyr, at redshift z'''
#     return cosmo.age(z).value
# def z_at_age(age):
#     '''Returns the redshfit corresponding to a given age of the universe, in Gyr'''
#     return np.interp(age, np.flip(age_array), np.flip(z_array))

def dict_to_str(d):
    # This is necessary for converting large arrays to strings
    np.set_printoptions(threshold=10**7)
    s = str(d)
    np.set_printoptions(threshold=10**4)
    return s
    
def str_to_dict(s):
    s = s.replace("array", "np.array")
    s = s.replace("float", "np.float")
    s = s.replace("np.np.", "np.")
    d = eval(s)
    return d


def parse_fit_params(parameters):
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


# def make_dirs(run="."):
#     working_dir = os.getcwd()
#     """ Make local Bagpipes directory structure in working dir. """

#     if not os.path.exists(working_dir + "/brisket"):
#         os.mkdir(working_dir + "/brisket")

#     if not os.path.exists(working_dir + "/brisket/plots"):
#         os.mkdir(working_dir + "/brisket/plots")

#     if not os.path.exists(working_dir + "/brisket/posterior"):
#         os.mkdir(working_dir + "/brisket/posterior")
    
#     if not os.path.exists(working_dir + "/brisket/models"):
#         os.mkdir(working_dir + "/brisket/models")

#     # if not os.path.exists(working_dir + "/brisket/cats"):
#     #     os.mkdir(working_dir + "/brisket/cats")

#     if run != ".":
#         if not os.path.exists("brisket/posterior/" + run):
#             os.mkdir("brisket/posterior/" + run)

#         if not os.path.exists("brisket/plots/" + run):
#             os.mkdir("brisket/plots/" + run)


def make_bins(midpoints, fix_low=None, fix_high=None):
    """ A general function for turning an array of bin midpoints into an
    array of bin positions. Splits the distance between bin midpoints equally in linear space.

    Parameters
    ----------
    midpoints : numpy.ndarray
        Array of bin midpoint positions

    fix_low : float, optional
        If set, the left edge of the first bin will be fixed to this value

    fix_high : float, optional
        If set, the right edge of the last bin will be fixed to this value
    """

    bins = np.zeros(midpoints.shape[0]+1)
    if fix_low is not None:
        bins[0] = fix_low
    else:
        bins[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
    if fix_high is not None:
        bins[-1] = fix_high
    else:
        bins[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
    bins[1:-1] = (midpoints[1:] + midpoints[:-1])/2

    return bins


def unit_parser(unit_str):
    d = {
        'angstrom': u.angstrom, 'AA': u.angstrom, 'A': u.angstrom, 
        'micron': u.um, 'um':u.um, 
        'nanometer': u.um, 'nm':u.nm, 
        'ergscma': u.erg/u.s/u.cm**2/u.angstrom,
        'erg/s/cm2/a': u.erg/u.s/u.cm**2/u.angstrom,
        'erg/s/cm2/aa': u.erg/u.s/u.cm**2/u.angstrom,
        'ergscm2a': u.erg/u.s/u.cm**2/u.angstrom,
        'nanojy': u.nJy, 'nanoJy': u.nJy, 'njy': u.nJy, 'nJy': u.nJy, 
        'mujy': u.uJy, 'muJy': u.uJy, 'uJy': u.uJy, 'ujy': u.uJy, 
        'mjy': u.mJy, 'mJy': u.mJy,
        'jy': u.Jy, 'Jy': u.Jy,
    }
    return d[unit_str]