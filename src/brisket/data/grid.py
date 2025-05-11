import boto3
from botocore import UNSIGNED
from botocore.client import Config

import sys, os, h5py
import numpy as np
from .. import config

from spectres import spectres

from scipy.interpolate import RegularGridInterpolator

from rich import print
from rich.prompt import Confirm

from ..utils import exceptions
from .downloader import Downloader

downloader = Downloader(box='brisket-data')

class Grid:

    def __init__(self, name):
        self.name = name
        if not self.name.endswith('.hdf5'):
            self.name += '.hdf5'

        self.path = self.check_exists(os.path.join(config.grid_dir, name))
        self._load_from_hdf5(self.path)
    
    def check_exists(self, path):
        
        if os.path.exists(path):
            return path

        print(f"Grid file [blue]{grid_file_name}[/blue] not found locally at [blue]{config.grid_dir}[/blue].")
        remote_available = downloader.check_exists(self.name)

        if not remote_available:
            print(f"Grid file [blue]{grid_file_name}[/blue] not found on remote bucket [blue]s3://{downloader.bucket}[/blue].")
            print("[red italic]Perhaps the file name is incorrect?")
            sys.exit(1)

        downloader.download_file(grid_file_name, path, prompt=True)

        return path

    def _load_from_hdf5(self, path):

        with h5py.File(path, 'r') as f:
            groups = list(f.keys())

            if not 'axes' in groups:
                raise exceptions.GridParseError(f'Grid file {self.path} does not contain an `axes` group. Please check the file.')
            if not 'continuum' in groups:
                raise exceptions.GridParseError(f'Grid file {self.path} does not contain a `continuum` group. Please check the file.')

            axes = list(f['axes'])

            continuum_datasets = list(f['continuum'].keys())
            if 'lines' in groups and 'transmitted' in continuum_datasets and 'nebular' in transmitted_datasets:
                self.photoionized = True
            else:
                self.photoionized = False
            
            self.axes = {n: np.array(f['axes'][n][:]) for n in axes}

            self.continuum = {
                'wavelengths': np.array(f['continuum']['wavelengths'][:]),
                'incident': np.array(f['continuum']['incident'][:]),
            }
            if self.photoionized:
                self.continuum['transmitted'] = np.array(f['continuum']['transmitted'][:])
                self.continuum['nebular'] = np.array(f['continuum']['nebular'][:])
                self.lines = {
                    'ids': np.array(f['lines']['id'][:]), 
                    'wavelengths': np.array(f['lines']['wavelengths'][:]), 
                    'luminosity': np.array(f['lines']['nebular'][:])
                }

            for axis in list(self.axes):
                setattr(self, axis, self.axes[axis])

    # @property 
    # def shape(self):
    #     return self.data.shape[:-1] 
    
    @property
    def ndim(self):
        return len(self.shape)

    @property 
    def wavelengths(self):
        return self.wavs

    # def __getitem__(self, indices):
    #     newgrid = deepcopy(self)
    #     newgrid.data = newgrid.data[indices]
    #     return newgrid
    
    # def __setitem__(self, indices, values):
    #     self.data[indices] = values

    def __repr__(self):
        return f'Grid({self.name}, photoionized={self.photoionized}, shape={self.shape})'


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
        

    # def to_sed(self, **kwargs):
    #     '''
    #     Converts the grid to a SED object.
    #     '''
    #     redshift = kwargs.get('redshift', None)
    #     verbose = kwargs.get('verbose', False)
    #     return SED(redshift=redshift, verbose=verbose, Llam=self.data*u.Lsun/u.angstrom, wav_rest=self.wavs*u.angstrom)




# from synthesizer.grid import Grid as SynthesizerGrid

# class Grid(SynthesizerGrid):

#     def __init__(self, 
#         grid_name,
#         grid_dir=None,
#         spectra_to_read=None,
#         read_lines=False,
#         new_lam=None,
#         lam_lims=(),
#     ):
#         """
#         Initialize the Grid object.

#         Subclassed from synthesizer.grid.Grid, to allow some some modifications
#         to improve speed and allow for vectorized operations. 

#         Args:
#             grid_dir (str)
#                 The file path to the directory containing the grid file.
#             spectra_to_read (list)
#                 A list of spectra to read in. If None then all available
#                 spectra will be read. Default is None.
#         """

#         SynthesizerGrid.__init__(
#             self,
#             grid_name, 
#             grid_dir=grid_dir, 
#             spectra_to_read=spectra_to_read, 
#             read_lines=read_lines,
#             new_lam=new_lam, 
#             lam_lims=lam_lims,
#         )

#     # def __repr__(self):
#     #     return f'Grid({self.name}, shape={self.shape})'



#     # def resample(self, new_wavs, fill=0, inplace=True):
#     #     new_data = spectres(new_wavs, self.wavs, self.data, fill=fill, verbose=False) 
#     #     self.wavs = new_wavs
#     #     if inplace:
#     #         self.data = new_data
#     #         return
#     #     else:
#     #         return new_data

#     def interpolate(self, 
#             params: dict[str, float],
#             inplace: bool = False,
#         ) -> np.ndarray | Grid:
        
#         '''
#         Returns the value of the grid at an arbitrary vector x, via linear interpolation.
#         '''

#         # if the grid has been updated (i.e., collapsed), we need to reinitialize the interpolator
#         if hasattr(self, '_interpolator'):
#             if self._interpolator_ndim != len(self.axes):
#                 delattr(self, '_interpolator')

#         # If the interpolator doesn't exist, create it
#         # For this we use scipy RegularGridInterpolator, which is quite fast and can interpolate to an array
#         if not hasattr(self, '_interpolator'):
#             points = ()
#             for axis in self.axes:
#                 points += (getattr(self, axis),)
            
#             self._interpolator_axes = list(self.axes)
#             self._interpolator_ndim = len(points)
#             self._interpolator = RegularGridInterpolator(points, self.data, bounds_error=False, fill_value=None)


#         x = [params.get(axis, None) for axis in self.axes] # TODO convert to array...

#         if inplace:
#             for axis in self._interpolator_axes:
#                 delattr(self, axis)
#             self.axes = [a for a in self.axes if a not in self._interpolator_axes]
#             self.array_axes = [a for a in self.array_axes if a not in self._interpolator_axes]

#             self.data = self._interpolator(x)
#             return 
        
#         else:
#             if len(x) == 1:
#                 return self._interpolator(x)[0]
#             else:
#                 return self._interpolator(x)


#     def collapse(self, 
#                  axis: str | int | Tuple[str, ...] | Tuple[int, ...], 
#                  weights: np.ndarray = None):
#         '''
#         Collapses (i.e., sums) the grid, along a given axis (or axes). 
#         Optionally, specify weights.

#         Args:
#             axis (str | int | Tuple[str, ...] | Tuple[int, ...]): The axis or axes to collapse over.
#             weights (np.ndarray): Weights to apply to the grid before collapsing.
#             inplace (bool): If True, the grid is updated in place. Otherwise, a new grid is returned.
#         '''

#         if isinstance(axis, str) or isinstance(axis, int):
#             # collapse over a single axis
#             axis = (axis,)

#         vectorize = False
#         if axis[0] is None:
#             vectorize = True
#             axis = axis[1:]

#         if all(isinstance(a, str) for a in axis):
#             axis_indices = [list(self.axes).index(a) for a in axis]
#         elif all(isinstance(a, int) for a in axis):
#             axis_indices = list(axes)
#         else:
#             raise ValueError('axis must be a string, an integer, or a tuple of strings or integers')

#         collapse_ndim = len(axis_indices)

#         if collapse_ndim == 1:
#             if weights is None:
#                 weights = np.ones(self.shape[axis_indices[0]])
#             assert weights.shape == self.shape[axis_indices[0]]

#         else:
#             if weights is None:
#                 weights = np.ones(self.shape)
            
#             else:
#                 if vectorize:
#                     for i,j in enumerate(axis_indices):
#                         assert weights.shape[i+1] == self.shape[j], f'Cannot apply {weights.shape[i+1]} weights to axis {axis[j]} with length {self.shape[j]}'

#                     for dim in range(self.ndim):
#                         if dim not in axis_indices:
#                             weights = np.expand_dims(weights, axis=dim+1)                

#                 else:
#                     for i,j in enumerate(axis_indices):
#                         assert weights.shape[i] == self.shape[j], f'Cannot apply {weights.shape[i]} weights to axis {axis[j]} with length {self.shape[j]}'

#                     for dim in range(self.ndim):
#                         if dim not in axis_indices:
#                             weights = np.expand_dims(weights, axis=dim)                

#         self.axes = [a for i,a in enumerate(self.axes) if i not in axis_indices]

#         for key in self.available_spectra:
#             spec = self.spectra[key]
#             if vectorize:
#                 spec = np.expand_dims(spec, axis=0)
#                 self.spectra[key] = np.sum(spec * weights, axis=tuple([a+1 for a in axis_indices]))
#             else:
#                 self.spectra[key] = np.sum(spec * weights, axis=tuple(axis_indices))
        

