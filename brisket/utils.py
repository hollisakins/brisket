from __future__ import print_function, division, absolute_import

import os
import numpy as np
import logging
import astropy.units as u

NullLogger = logging.getLogger('null')
NullLogger.addHandler(logging.NullHandler())


def parse_fit_params(parameters, logger=NullLogger):
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




def unit_parser(unit_str):
    d = {'angstrom': u.angstrom, 'AA': u.angstrom, 'A': u.angstrom, 
         'micron': u.um, 'um':u.um, 
         'nanometer': u.um, 'nm':u.nm, 
         'ergscma': u.erg/u.s/u.cm**2/u.angstrom,
         'erg/s/cm2/a': u.erg/u.s/u.cm**2/u.angstrom,
         'erg/s/cm2/aa': u.erg/u.s/u.cm**2/u.angstrom,
         'ergscm2a': u.erg/u.s/u.cm**2/u.angstrom,
         'nanojy': u.nJy, 'nanoJy': u.nJy, 'njy': u.nJy, 'nJy': u.nJy, 
         'mujy': u.uJy, 'muJy': u.uJy, 'uJy': u.uJy, 'ujy': u.uJy, 
         'mjy': u.mJy, 'mJy': u.mJy}
    return d[unit_str]




def make_dirs(run="."):
    working_dir = os.getcwd()
    """ Make local Bagpipes directory structure in working dir. """

    if not os.path.exists(working_dir + "/brisket"):
        os.mkdir(working_dir + "/brisket")

    if not os.path.exists(working_dir + "/brisket/plots"):
        os.mkdir(working_dir + "/brisket/plots")

    if not os.path.exists(working_dir + "/brisket/posterior"):
        os.mkdir(working_dir + "/brisket/posterior")

    if not os.path.exists(working_dir + "/brisket/cats"):
        os.mkdir(working_dir + "/brisket/cats")

    if run != ".":
        if not os.path.exists("brisket/posterior/" + run):
            os.mkdir("brisket/posterior/" + run)

        if not os.path.exists("brisket/plots/" + run):
            os.mkdir("brisket/plots/" + run)


def make_bins(midpoints, make_rhs=False):
    """ A general function for turning an array of bin midpoints into an
    array of bin left hand side positions and bin widths. Splits the
    distance between bin midpoints equally in linear space.

    Parameters
    ----------

    midpoints : numpy.ndarray
        Array of bin midpoint positions

    make_rhs : bool
        Whether to add the position of the right hand side of the final
        bin to bin_lhs, defaults to false.
    """

    bin_widths = np.zeros_like(midpoints)

    if make_rhs:
        bin_lhs = np.zeros(midpoints.shape[0]+1)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
        bin_lhs[1:-1] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    else:
        bin_lhs = np.zeros_like(midpoints)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[1:] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:]-bin_lhs[:-1]

    return bin_lhs, bin_widths


