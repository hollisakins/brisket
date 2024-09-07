from __future__ import print_function, division, absolute_import

import os
import numpy as np
import logging
import astropy.units as u
import sys

install_dir = os.path.dirname(os.path.realpath(__file__))
grid_dir = install_dir + "/models/grids"
param_template_dir = install_dir + "/defaults/templates/"

basicLogger = logging.getLogger('brisket')
basicLogger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout) # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', "%H:%M:%S")
ch.setFormatter(formatter)
basicLogger.addHandler(ch)

NullLogger = logging.getLogger('null')
NullLogger.addHandler(logging.NullHandler())

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
    if unit_str in ['angstrom','AA','A','ang']: return u.angstrom
    elif unit_str in ['micrometer', 'micron','um']: return u.um
    elif unit_str in ['nanometer','nm']: return u.nm
    elif unit_str in ['ergscma','erg/s/cm2/a','erg/s/cm2/aa',
                      'ergscm2a','erg/s/cm2/A','erg/s/cm^2/A']: return u.erg/u.s/u.cm**2/u.angstrom
    elif unit_str in ['nanojy','nanoJy','njy','nJy']: return u.nJy
    elif unit_str in ['mujy','muJy','uJy','ujy']: return u.uJy
    elif unit_str in ['mjy','mJy']: return u.mJy




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


