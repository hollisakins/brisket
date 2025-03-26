'''
Provides access to the Grid object, which allows manipulation of 
SED grids. 

Much of this code is adapted from synthesizer.grid.Grid. 
https://synthesizer-project.github.io/synthesizer/
'''

from __future__ import annotations
import os, h5py
import numpy as np
from copy import deepcopy
from typing import Tuple

from .. import config
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
        Initialize the Grid object.

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
        self.axes = [] # list of axes names
        self._axes_values = {}
        self._axes_units = {}
        self._extract_axes = []
        self._extract_axes_values = {}
        self._get_axes()

        # Read in the metadata
        self._weight_var = None
        self._model_metadata = {}
        self._get_grid_metadata()

        # Get the ionising luminosity (if available)
        self._get_ionising_luminosity()

        # We always read spectra, but can read a subset if requested
        self.lam = None
        self.available_spectra = None
        self._get_spectra_grid(spectra_to_read)





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


    def _get_axes(self):
        """Get the grid axes from the HDF5 file."""
        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            # Get list of axes
            axes = list(hf.attrs["axes"])

            # Save the values of each axis to _axes_values
            for axis in axes:
                # What are the units of this axis?
                axis_units = hf["axes"][axis].attrs.get("Units")
                log_axis = hf["axes"][axis].attrs.get("log_on_read")

                if "log10" in axis:
                    raise exceptions.GridError(
                        "Logged axes are no longer supported because "
                        "of ambiguous units. Please update your grid file."
                    )

                # Get the values
                values = hf["axes"][axis][:]

                # Set all the axis attributes 
                self.axes.append(axis)
                self._axes_values[axis] = values
                self._axes_units[axis] = axis_units

                # Now we handle the extractions
                if log_axis:
                    self._extract_axes.append(f"log10{axis}")
                    self._extract_axes_values[f"log10{axis}"] = np.log10(values)
                else:
                    self._extract_axes.append(axis)
                    self._extract_axes_values[axis] = values

            # Number of axes
            self.naxes = len(self.axes)

    def _get_grid_metadata(self):
        """Unpack the grids metadata into the Grid."""
        # Open the file
        with h5py.File(self.grid_filename, "r") as hf:
            # What component variable do we need to weight by for the
            # emission in the grid?
            self._weight_var = hf.attrs.get("WeightVariable")

            # Loop over the Model metadata stored in the Model group
            # and store it in the Grid object
            if "Model" in hf:
                for key, value in hf["Model"].attrs.items():
                    self._model_metadata[key] = value

            # Attach all the root level attribtues to the grid object
            for k, v in hf.attrs.items():
                # Skip the axes attribute as we've already read that
                if k == "axes" or k == "WeightVariable":
                    continue
                setattr(self, k, v)


    def _get_ionising_luminosity(self):
        """Get the ionising luminosity from the HDF5 file."""
        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            # Extract any ionising luminosities
            if "log10_specific_ionising_luminosity" in hf.keys():
                self.log10_specific_ionising_lum = {}
                for ion in hf["log10_specific_ionising_luminosity"].keys():
                    self.log10_specific_ionising_lum[ion] = hf[
                        "log10_specific_ionising_luminosity"
                    ][ion][:]

    def _get_spectra_ids_from_file(self):
        """
        Get a list of the spectra available in a grid file.

        Returns:
            list:
                List of available spectra
        """
        with h5py.File(self.grid_filename, "r") as hf:
            spectra_keys = list(hf["spectra"].keys())

        # Clean up the available spectra list
        spectra_keys.remove("wavelength")

        # Remove normalisation dataset
        if "normalisation" in spectra_keys:
            spectra_keys.remove("normalisation")

        return spectra_keys

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

    # def __getitem__(self, indices):
    #     newgrid = deepcopy(self)
    #     newgrid.data = newgrid.data[indices]
    #     return newgrid
    
    # def __setitem__(self, indices, values):
    #     self.data[indices] = values

    # def __array__(self, dtype=None, copy=None):
    #     return self.data

    def __repr__(self):
        return f'Grid({self.name}, shape={self.shape})'


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
        

