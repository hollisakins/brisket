'''
This module defines the Params class, which is used to store and manage model parameters. 
'''

from __future__ import annotations
import rich, os, sys
import numpy as np
from copy import deepcopy
from collections.abc import MutableMapping

from .fitting.priors import Prior

from rich.table import Table
from rich.tree import Tree

from .utils.console import console, setup_logger, PathHighlighter, LimitsHighlighter

class Params(MutableMapping):
    '''
    The Params class is used to store and manage model parameters.
    
    Args:
        template (str, optional)
            Name of the parameter template to use, if desired (default: None).
        file (str, optional)
            Path to parameter file, if desired (default: None).
        verbose (bool, optional)
            Whether to print log messages (default: False).
    
    '''

    def __init__(self, verbose=False):

        self.logger = setup_logger(__name__, verbose)
        
        self._data = {}

        self.linked_params = {} # TODO

    # def add_parameter(self, key, value):
    #     self._data[key] = value

    # def add_child(self, child_name, params=None):
    #     if child_name in self.children:
    #         raise Exception(f'{child_name} already exists in Params object')
    #     if params is None:
    #         child = Params(name=child_name, parent=self)
    #     else:
    #         child = params
    #     self.children[child_name] = child
    #     return child
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        if key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        else:
            raise KeyError(f"No key {key} found in {self}")
    
    def __contains__(self, key):
        return dict.__contains__(self.all_params, key)
    
    def get(self, key, default=None):
        if key in self:
            return self.__getitem__(key)
        else:
            return default

    def __iter__(self):
        return self.to_dict().__iter__()
    
    def __len__(self):
        return len(self.to_dict())

    def __delitem__(self, key):
        if key in self._data:
            del self._data[key]
        else:
            raise KeyError(f"No key {key} found in {self}")

    def to_dict(self) -> dict:
        d = {}
        d.update(self._data)
        return d
    
    # def _flattener(self, d, parent=None, sep='/'):
    #     for key, value in d.items():
    #         new_key = parent + sep + key if parent else key
    #         if isinstance(value, dict):
    #             yield from self._flattener(value, parent=key, sep=sep)
    #         else:
    #             yield new_key, value

    # @property
    # def all_params(self) -> dict:
    #     return dict(self._flattener(self.to_dict(), sep='/'))
    
    def items(self):
        """Returns a list of tuples (key, value) for all parameters in the model."""
        return self.to_dict().items()

    @property
    def all_params(self) -> dict:
        return self.to_dict()

    @property
    def free_params(self) -> dict:
        free_params = {}
        for key, value in self.all_params.items():
            if isinstance(value, Prior): 
                free_params[key] = value
        return free_params

    @property 
    def free_param_names(self) -> list:
        """List of names of free parameters in the model."""
        return list(self.free_params.keys())
    
    @property
    def all_param_names(self) -> list:
        """List of names of parameters in the model."""
        return list(self.all_params.keys())
    
    @property 
    def all_param_values(self) -> list:
        """List of values of parameters in the model."""
        return list(self.all_params.values())

    @property
    def nparam(self):
        return len(self.all_param_names)
    
    @property
    def nfree(self):
        return len(self.free_param_names)

    def __repr__(self):
        s = f"Params(nparam={self.nparam}, nfree={self.nfree}):\n"
        for key, value in self.all_params.items():
            s += f"  {key}: {value}\n"
        return s

    def withprefix(self, prefix):
        """Adds a prefix to all parameter names in the model."""
        new = deepcopy(self)
        for key, value in new.all_params.items():
            new_key = f"{prefix}/{key}"
            new._data[new_key] = value
            del new._data[key]
        return new
        
    def getprefix(self, prefix):
        """Returns a new Params object with only the parameters that start with the given prefix, and with said prefix removed."""
        new = deepcopy(self)
        new._data = {}
        for key, value in self.all_params.items():
            if key.startswith(prefix+'/'):
                new_key = key[len(prefix)+1:]
                new._data[new_key] = value
        return new
        
    # def print_table(self):
    #     """Prints a summary of the model parameters, in table form."""
    #     h = PathHighlighter()
    #     l = LimitsHighlighter()
    #     if (self.nfree == 0) or (self.nparam != self.nfree):
    #         if self.nfree == 0:
    #             table = Table(title="")
    #         else:
    #             table = Table(title="Fixed Parameters")

    #         table.add_column("Parameter name", justify="left", no_wrap=True)
    #         table.add_column("Value", style='bold #FFE4B5', justify='left', no_wrap=True)

    #         for i in range(self.nparam): 
    #             n = self.all_param_names[i]
    #             if n in self.free_param_names: continue
    #             table.add_row(h(n), str(self.all_params[n]))

    #         console.print(table)
                     
    #     if self.nfree > 0:
    #         table = Table(title="Free Parameters")
    #         table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
    #         table.add_column("Limits", style=None, justify='left', no_wrap=True)
    #         table.add_column("Prior", style=None, no_wrap=True)
        
    #         for i in range(self.nfree): 
    #             n = self.free_param_names[i]
    #             p = self.free_params[n]
    #             table.add_row(h(n), str(p))
        
    #         console.print(table)

    # def print_tree(self):
    #     """Prints a summary of the model parameters, in tree form."""
    #     tree = Tree(f"[bold italic white]{self.model.__class__.__name__}[/bold italic white](nparam={self.nparam}, nfree={self.nfree})")
    #     children = list(self.children)
    #     names = [n for n in self.all_param_names if '/' not in n]
    #     for name in names:
    #         tree.add('[bold #FFE4B5 not italic]' + name + '[white]: [italic not bold #c9b89b]' + self.all_params[name].__repr__())
    #     for child in children:
    #         source = tree.add(f'[bold #6495ED not italic]{child}[white]: [italic not bold #6480b3]{self.children[child].model.__class__.__name__}[not italic](nparam={self.children[child].nparam}, nfree={self.children[child].nfree})')
    #         params_i = self.children[child]
    #         names_i = [n for n in params_i.all_param_names if '/' not in n]
    #         for name_i in names_i:
    #             source.add('[bold #FFE4B5 not italic]' + name_i + '[white]: [italic not bold #c9b89b]' + params_i.all_params[name_i].__repr__())
    #         children_i = list(params_i.children)
    #         for child_i in children_i:
    #             subsource = source.add(f'[bold #8fbc8f not italic]{child_i}[white]: [italic not bold #869e86]{params_i.children[child_i].model.__class__.__name__}[not italic](nparam={params_i.children[child_i].nparam}, nfree={params_i.children[child_i].nfree})')
    #             params_ii = params_i.children[child_i]
    #             names_ii = [n for n in params_ii.all_param_names if '/' not in n]
    #             for name_ii in names_ii:
    #                 subsource.add('[bold #FFE4B5 not italic]' + name_ii + '[white]: [italic not bold #c9b89b]' + params_ii.all_params[name_ii].__repr__())
    #     console.print(tree)

    def update(self, other):
        if not len(other) == len(self):
            raise ValueError(f"Cannot update {self} with {other} because they have different lengths.")
        self._data.update(other._data)

    def concatenate(self, other):
        
        for key, value in other.items():
            # if key in self._data:
            #     # If the key already exists, rename it (and rename the other one too)
            #     key1,key2 = f"{key}1", f"{key2}"
            #     self._data[key1] = self._data[key]
            #     del self._data[key]
            #     self._data[key2] = other[key]
            # else:
            self._data[key] = other[key]
        return self


    def __iadd__(self, other):
        if isinstance(other, Params):
            self._data.update(other._data)
        else:
            raise TypeError(f"Cannot add {type(other)} to Params")
        return self

    # def update_from_vector(self, names, x):
    #     # Vectorization note: x can be a 1D array of size nfree, or a 2D array of size (?, nfree).
    #     # names is a 1D array of size nfree
    #     for i, name in enumerate(names):
    #         if name in self.free_params:
    #             del self.free_params[name]
    #         self.all_params[name] = x[i]
        
    #     x_components = np.array([p.split('/')[0] for p in names])
    #     x_names = np.array([p.removeprefix(c+'/') for p,c in zip(names, x_components)])
    #     for component in x_components:
    #         if component in self._components:
    #             self._components[component].update_from_vector(x_names[x_components==component], x[x_components==component])
