from __future__ import annotations
import os, h5py
import numpy as np
from copy import deepcopy
from typing import Tuple

from .. import config
from ..utils.sed import SED
from spectres import spectres

from scipy.interpolate import RegularGridInterpolator

class Grid:

    def __init__(self, 
        grid_name,
        grid_dir=None,
        read_lines=True,
        spectra_to_read=None,
    ):
        """

        Largely adapted from synthesizer.grid.Grid, but with some modifications
        to improve speed and allow for vectorized operations. 

        Args:
            grid_dir (str)
                The file path to the directory containing the grid file.
            spectra_to_read (list)
                A list of spectra to read in. If None then all available
                spectra will be read. Default is None.
        """

        # Get the grid file path data
        self.grid_dir = ""
        self.grid_name = ""
        self.grid_ext = "hdf5"  # can be updated if grid_name has an extension
        self._parse_grid_path(grid_dir, grid_name)
        
        # Prepare lists of available lines and spectra
        self.available_lines = []
        self.available_spectra = []

        # Set up property flags. These will be set when their property methods
        # are first called to avoid reading the file too often.
        self._reprocessed = None
        self._lines_available = None

        # Set up spectra and lines dictionaries (if we don't read them they'll
        # just stay as empty dicts)
        self.spectra = {}
        self.line_lams = {}
        self.line_lums = {}
        self.line_conts = {}

        # Get the axes of the grid from the HDF5 file
        self.axes = []  # axes names
        self._axes_values = {}
        self._axes_units = {}
        self._extract_axes = []
        self._extract_axes_values = {}
        self._get_axes()


    def _parse_grid_path(self, grid_dir, grid_name):
        """Parse the grid path and set the grid directory and filename."""
        # If we haven't been given a grid directory, check the config file 
        # for a user-specified grid directory
        if grid_dir is None:
            grid_dir = config["grid_dir"]

            if grid_dir is None:
                # Otherwise, assume the grids are stored in the package data directory
                grid_dir = os.path.join(os.path.dirname(__file__), "data/grids")

        # Store the grid directory
        self.grid_dir = grid_dir

        # Have we been passed an extension?
        grid_name_split = grid_name.split(".")[-1]
        ext = grid_name_split[-1]
        if ext == "hdf5" or ext == "h5":
            self.grid_ext = ext

        # Strip the extension off the name (harmless if no extension)
        self.grid_name = grid_name.replace(f".{self.grid_ext}", "")

        # Construct the full path
        self.grid_filename = os.path.join(
            f"{self.grid_dir}", f"{self.grid_name}.{self.grid_ext}"
        )




    def _load_from_hdf5(self, path):
        with h5py.File(path, 'r') as f:
            axes = list(f['axes'].asstr(encoding='utf-8'))
            assert axes[-1] == 'wavs'
            self.wavs = f['wavs'][:]
            self.data = f['grid'][:]
            self.axes = axes[:-1]
            self.array_axes = axes
            for axis in self.axes:
                setattr(self, axis, f[axis][:])
            for key in f.keys():
                if key not in axes + ['grid', 'wavs', 'axes']:
                    setattr(self, key, f[key][:])

    @property 
    def shape(self):
        return self.data.shape[:-1] # remove the last axis, which is the SED
    
    @property
    def ndim(self):
        return len(self.shape) # remove the last axis, which is the SED

    @property 
    def wavelengths(self):
        return self.wavs

    def __getitem__(self, indices):
        newgrid = deepcopy(self)
        newgrid.data = newgrid.data[indices]
        return newgrid
    
    def __setitem__(self, indices, values):
        self.data[indices] = values

    def __array__(self, dtype=None, copy=None):
        return self.data

    def __repr__(self):
        return f'Grid({self.name}, shape={self.shape})'


    def _get_spectra_grid(self, spectra_to_read):
        """
        Get the spectra grid from the HDF5 file.

        If using a cloudy reprocessed grid this method will automatically
        calculate 2 spectra not native to the grid file:
            total = transmitted + nebular
            nebular_continuum = nebular - linecont

        Args:
            spectra_to_read (list)
                A list of spectra to read in. If None then all available
                spectra will be read.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            # Are we reading everything?
            if spectra_to_read is None:
                self.available_spectra = self._get_spectra_ids_from_file()
            elif isinstance(spectra_to_read, list):
                all_spectra = self._get_spectra_ids_from_file()
                self.available_spectra = spectra_to_read

                # Check the requested spectra are available
                missing_spectra = set(spectra_to_read) - set(all_spectra)
                if len(missing_spectra) > 0:
                    raise exceptions.MissingSpectraType(
                        f"The following requested spectra are not available"
                        "in the supplied grid file: "
                        f"{missing_spectra}"
                    )
            else:
                raise exceptions.InconsistentArguments(
                    "spectra_to_read must either be None or a list "
                    "containing a subset of spectra to read."
                )

            # Read the wavelengths
            self.lam = hf["spectra/wavelength"][:]

            # Get all our spectra
            for spectra_id in self.available_spectra:
                self.spectra[spectra_id] = hf["spectra"][spectra_id][:]

        # If a full cloudy grid is available calculate some
        # other spectra for convenience.
        if self.reprocessed:
            # The total emission (ignoring any dust reprocessing) is just
            # the transmitted plus the nebular
            self.spectra["total"] = (
                self.spectra["transmitted"] + self.spectra["nebular"]
            )
            self.available_spectra.append("total")

            # The nebular continuum is the nebular emission with the line
            # contribution removed
            self.spectra["nebular_continuum"] = (
                self.spectra["nebular"] - self.spectra["linecont"]
            )
            self.available_spectra.append("nebular_continuum")
            
    @property
    def reprocessed(self):
        """
        Flag for whether grid has been reprocessed through cloudy.

        This will only access the file the first time this property is
        accessed.

        Returns:
            True if reprocessed, False otherwise.
        """
        if self._reprocessed is None:
            with h5py.File(self.grid_filename, "r") as hf:
                old_grids = (
                    True if "cloudy_version" in hf.attrs.keys() else False
                )
                new_grid = True if "CloudyParams" in hf.keys() else False
                self._reprocessed = old_grids or new_grid

        return self._reprocessed
    
    @property
    def lines_available(self):
        """
        Flag for whether line emission exists.

        This will only access the file the first time this property is
        accessed.

        Returns:
            bool:
                True if lines are available, False otherwise.
        """
        if self._lines_available is None:
            with h5py.File(self.grid_filename, "r") as hf:
                self._lines_available = True if "lines" in hf.keys() else False

        return self._lines_available

    @property
    def shape(self):
        """Return the shape of the grid."""
        return self.spectra[self.available_spectra[0]].shape

    @property
    def ndim(self):
        """Return the number of dimensions in the grid."""
        return len(self.shape)

    @property
    def nlam(self):
        """Return the number of wavelengths in the grid."""
        return len(self.lam)
    @property
    def has_spectra(self):
        """Return whether the Grid has spectra."""
        return len(self.spectra) > 0

    @property
    def has_lines(self):
        """Return whether the Grid has lines."""
        return len(self.line_lums) > 0

    def resample(self, new_wavs, fill=0, inplace=True):
        new_data = spectres(new_wavs, self.wavs, self.data, fill=fill, verbose=False) 
        self.wavs = new_wavs
        if inplace:
            self.data = new_data
            return
        else:
            return new_data


    def get_nearest(self, x, return_nearest=False):
        '''
        Returns the value of the grid at the point nearest to the vector x.
        If return_nearest=True, also returns the vector of the nearest point.
        '''
        points = ()
        for axis in self.axes:
            points += (getattr(self, axis),)
        
        index = ()
        for i in range(len(x)):
            index += (np.argmin(np.abs(points[i]-x[i])),)

        if return_nearest:
            return self.data[index], [points[i][j] for i,j in enumerate(index)]
        else:
            return self.data[index]

    def interpolate(self, 
                    params: dict[str, float],
                    inplace: bool = False,
                    ) -> np.ndarray | Grid:
        '''
        Returns the value of the grid at an arbitrary vector x, via linear interpolation.
        '''

        # if the grid has been updated (i.e., collapsed), we need to reinitialize the interpolator
        if hasattr(self, '_interpolator'):
            if self._interpolator_ndim != len(self.axes):
                delattr(self, '_interpolator')

        if not hasattr(self, '_interpolator'):
            points = ()
            for axis in self.axes:
                points += (getattr(self, axis),)
            
            self._interpolator_axes = list(self.axes)
            self._interpolator_ndim = len(points)
            self._interpolator = RegularGridInterpolator(points, self.data, bounds_error=False, fill_value=None)


        x = [params.get(axis, None) for axis in self.axes]

        if inplace:
            for axis in self._interpolator_axes:
                delattr(self, axis)
            self.axes = [a for a in self.axes if a not in self._interpolator_axes]
            self.array_axes = [a for a in self.array_axes if a not in self._interpolator_axes]

            self.data = self._interpolator(x)
            return 
        
        else:
            if len(x) == 1:
                return self._interpolator(x)[0]
            else:
                return self._interpolator(x)


    def collapse(self, 
                 axis: str | int | Tuple[str, ...] | Tuple[int, ...], 
                 weights: np.ndarray = None, 
                 inplace: bool = False):
        '''
        Collapses (i.e., sums) the grid, along a given axis (or axes). 
        Optionally, specify weights.

        Args:
            axis (str | int | Tuple[str, ...] | Tuple[int, ...]): The axis or axes to collapse over.
            weights (np.ndarray): Weights to apply to the grid before collapsing.
            inplace (bool): If True, the grid is updated in place. Otherwise, a new grid is returned.
        '''

        if isinstance(axis, str) or isinstance(axis, int):
            # collapse over a single axis
            axis = (axis,)

        if all(isinstance(a, str) for a in axis):
            axis_indices = [list(self.axes).index(a) for a in axis]
        elif all(isinstance(a, int) for a in axis):
            axis_indices = list(axes)
        else:
            raise ValueError('axis must be a string, an integer, or a tuple of strings or integers')


        collapse_ndim = len(axis_indices)

        if collapse_ndim == 1:
            if weights is None:
                weights = np.ones(self.shape[axis_indices[0]])
            assert weights.shape == self.shape[axis_indices[0]]

        else:
            if weights is None:
                weights = np.ones(self.shape)
            else:

                for i,j in enumerate(axis_indices):
                    assert weights.shape[i] == self.shape[j], f'Cannot apply {weights.shape[i]} weights to axis {axis[j]} with length {self.shape[j]}'

                for dim in range(self.ndim):
                    if dim not in axis_indices:
                        weights = np.expand_dims(weights, axis=dim)                

                # assert weights.shape == self.shape, f'Cannot apply weights of shape {weights.shape} to grid of shape {self.shape}'

        weights = np.expand_dims(weights, axis=-1)



        if inplace:
            for i in axis_indices:
                delattr(self, self.axes[i])
            self.axes = [a for i,a in enumerate(self.axes) if i not in axis_indices]
            self.array_axes = [a for i,a in enumerate(self.array_axes) if i not in axis_indices]

            self.data = np.sum(self.data * weights, axis=tuple(axis_indices))
            return 
        
        else:
            return np.sum(self.data * weights, axis=tuple(axis_indices))
        


