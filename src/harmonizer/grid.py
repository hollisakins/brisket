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

from synthesizer.grid import Grid as SynthesizerGrid

class Grid(SynthesizerGrid):

    def __init__(self, 
        grid_name,
        grid_dir=None,
        read_lines=True,
        spectra_to_read=None,
    ):
        """
        Initialize the Grid object.

        Subclassed from synthesizer.grid.Grid, to allow some some modifications
        to improve speed and allow for vectorized operations. 

        Args:
            grid_dir (str)
                The file path to the directory containing the grid file.
            spectra_to_read (list)
                A list of spectra to read in. If None then all available
                spectra will be read. Default is None.
        """

        SynthesizerGrid.__init__(self, grid_name, grid_dir, read_lines, spectra_to_read)

    # def __repr__(self):
    #     return f'Grid({self.name}, shape={self.shape})'



    # def resample(self, new_wavs, fill=0, inplace=True):
    #     new_data = spectres(new_wavs, self.wavs, self.data, fill=fill, verbose=False) 
    #     self.wavs = new_wavs
    #     if inplace:
    #         self.data = new_data
    #         return
    #     else:
    #         return new_data

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

        # If the interpolator doesn't exist, create it
        # For this we use scipy RegularGridInterpolator, which is quite fast and can interpolate to an array
        if not hasattr(self, '_interpolator'):
            points = ()
            for axis in self.axes:
                points += (getattr(self, axis),)
            
            self._interpolator_axes = list(self.axes)
            self._interpolator_ndim = len(points)
            self._interpolator = RegularGridInterpolator(points, self.data, bounds_error=False, fill_value=None)


        x = [params.get(axis, None) for axis in self.axes] # TODO convert to array...

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
        

