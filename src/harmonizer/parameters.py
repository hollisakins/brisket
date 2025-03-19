'''
This module defines the Params class, which is used to store and manage model parameters. 
'''

from __future__ import annotations
import rich, os, sys
import numpy as np
from collections.abc import MutableMapping

from .fitting import priors

from rich.table import Table
from rich.tree import Tree


# from . import config
# from .fitting import priors
from .console import console, setup_logger, PathHighlighter, LimitsHighlighter
# from .models.agn import PowerlawAccrectionDiskModel
# from .models.igm import InoueIGMModel
# from .models.calibration import SpectralCalibrationModel

# from synthesizer.parametric import Stars as ParametricStars
# from synthesizer.parametric import BlackHole as ParametricBlackHole

# # TODO default model choices given source names
# # TODO default parameter choices given source names

# emitter_model_defaults = {
#     'stars': ParametricStars, 
#     'agn': ParametricBlackHole
# }


# sfh_defaults = {
#     'continuity': sfh.Continuity,
# }


reserved_children = ['stars', 'agn', 'nebular', 'dust_attenuation', 'dust_emission', 'igm']
# reserved_children.extend(sfh_defaults.keys())




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

    def __init__(self, template=None, file=None, verbose=False, name=None, parent=None): #*args, **kwargs):

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')
        
        self.name = name
        self.parent = parent
        self.children = {}
        self.sfh_name = None

        # if file is not None:
        #     try:
        #         data = self._parse_from_toml(file)
        #     except FileNotFoundError:
        #         self.logger.error(f"Parameter file {data} not found."); sys.exit()
        # elif template is not None:
        #     try:
        #         data = self._parse_from_toml(os.path.join(utils.param_template_dir, template+'.toml'))
        #     except FileNotFoundError:
        #         self.logger.error(f"Parameter template {data} not found. Place template parameter files in the brisket/defaults/templates/."); sys.exit()
        
        self._data = {}
        self._emitters = []

        self.linked_params = {} # TODO
        self.validated = False

    # def add_emitter(name, emitter_model=None):
    #     '''
    #     Args:
    #         name (str)
    #             Name of the group, used to reference it in later calls to the Params object. 
    #         emitter_model (class, optional)
    #             The model class to use for this group of parameters. If not specified, the 
    #             model will be chosen based on the name of the group based on the model_defaults dict. 
    #     '''
    #     if emitter_model is None:
    #         model_def = None
    #         for key in emitter_model_defaults:
    #             if key in name:
    #                 model_def = emitter_model_defaults[key]
    #                 break
    #         if model_def is None:
    #             raise Exception(f'No default model for emitter {name}, please specify emitter_model')
    #         emitter_model = model_def
        
    #     group = Group(name, emitter_model, parent=self)
    #     self.__setitem__(name, group)

    #     self._emitters.append(name)

    def _add_parameter(self, key, value):
        self._data[key] = value

    def _add_child(self, child_name):
        if child_name in self.children:
            raise Exception(f'{child_name} already exists in Params object')
        child = Params(name=child_name, parent=self)
        self.children[child_name] = child
        return child

    def add_stars(self):
        return self._add_child('stars')
        
    def add_agn(self):
        return self._add_child('agn')
    
    def add_nebular(self, **kwargs):
        child = self._add_child('nebular')
        child._data.update(kwargs)
        return child

    def add_dust_attenuation(self, **kwargs):
        child = self._add_child('dust_attenuation')
        child._data.update(kwargs)
        return child
    
    def add_dust_emission(self, **kwargs):
        child = self._add_child('dust_emission')
        child._data.update(kwargs)
        return child

    def add_igm(self, **kwargs):
        child = self._add_child('igm')
        child._data.update(kwargs)
        return child

    def add_sfh(self, name, **kwargs):
        if self.name != 'stars':
            raise Exception('SFH can only be added to the stars component')
        
        if hasattr(self, name):
            raise Exception(f'SFH with name {name} already exists in stars component')
        
        if self.has_sfh:
            raise Exception('Only one SFH can be added.')

        sfh = self._add_child(name)
        if name in ['continuity', 'bursty_continuity']:
            if not 'n_bins' in kwargs:
                raise Exception('n_bins must be specified for continuity and bursty_continuity SFHs')
            sfh['n_bins'] = kwargs['n_bins']

            if 'bin_edges' in kwargs:
                sfh['bin_edges'] = kwargs['bin_edges']

            if 'z_max' in kwargs:
                sfh['z_max'] = kwargs['z_max']

            df = 2.0
            if name == 'continuity': scale = 0.3
            if name == 'bursty_continuity': scale = 1.0
            for i in range(sfh['n_bins']):
                sfh[f'dsfr{i}'] = priors.StudentsT(low=-10, high=10, loc=0, scale=scale, df=df)

        if name == 'constant':
            sfh['min_age'] = kwargs['min_age']
            sfh['max_age'] = kwargs['max_age'] # TODO assign default units? 



        self.sfh_name = name
        return sfh

    @property
    def has_sfh(self):
        return self.sfh_name is not None

    def __setitem__(self, key, value):
        if isinstance(value, Params):
            value.parent = self
            self.children[key] = value
        else:
            self._data[key] = value

    def __getitem__(self, key):
        if key in self.children: # getting an emitter
            return self.children[key]
        elif key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        elif key in self._data:
            return self._data[key]
        else:
            raise Exception(f"No key {key} found in {self}")
    
    def __contains__(self, key):
        return dict.__contains__(self.all_params, key) or dict.__contains__(self.children, key)
    
    def get(self, key, default=None):
        if key in self:
            return self.__getitem__(key)
        else:
            return default

    # def add_igm(self):
    #     self.include_igm = True
    #     self._data['agn'] = {}

    # def has_emitter(name):
    #     return name in self._emitters

    # def __dict__(self):

    #     return dict(_flattener(self))        

    def __iter__(self):
        return self.to_dict().__iter__()
    
    def __len__(self):
        return len(self.to_dict())

    def __delitem__(self, key):
        if key in self.children:
            del self.children[key]
        elif key in self._data:
            del self._data[key]
        else:
            raise Exception(f"No key {key} found in {self}")


    def to_dict(self) -> dict:
        d = {}
        d.update(self._data)
        for key, value in self.children.items():
            d[key] = value.to_dict()
        return d
    
    def _flattener(self, d, parent=None, sep='/'):
        for key, value in d.items():
            new_key = parent + sep + key if parent else key
            if isinstance(value, dict):
                yield from self._flattener(value, parent=key, sep=sep)
            else:
                yield new_key, value

    @property
    def all_params(self) -> dict:
        return dict(self._flattener(self.to_dict(), sep='/'))

    @property
    def free_params(self) -> dict:
        free_params = {}
        for key, value in self.all_params.items():
            if isinstance(value, priors.Common): 
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
    def ndim(self):
        return len(self.free_param_names)

    # def __delitem__(self, key):
    #     if key in self._components:
    #         del self._components[key]
    #         del self._component_types[key]
    #         del self._component_orders[key]
    #     elif key in self.all_params:
    #         del self.all_params[key]
    #         if key in self.free_params:
    #             del self.free_params[key]
    #     else:
    #         raise Exception(f"No key {key} found in {self}")

    def __repr__(self):
        return f"Params(nparam={self.nparam}, ndim={self.ndim})"
        
    def print_table(self):
        """Prints a summary of the model parameters, in table form."""
        h = PathHighlighter()
        l = LimitsHighlighter()
        if (self.ndim == 0) or (self.nparam != self.ndim):
            if self.ndim == 0:
                table = Table(title="")
            else:
                table = Table(title="Fixed Parameters")

            table.add_column("Parameter name", justify="left", no_wrap=True)
            table.add_column("Value", style='bold #FFE4B5', justify='left', no_wrap=True)

            for i in range(self.nparam): 
                n = self.all_param_names[i]
                if n in self.free_param_names: continue
                table.add_row(h(n), str(self.all_params[n]))

            console.print(table)
                     
        if self.ndim > 0:
            table = Table(title="Free Parameters")
            table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
            table.add_column("Limits", style=None, justify='left', no_wrap=True)
            table.add_column("Prior", style=None, no_wrap=True)
        
            for i in range(self.ndim): 
                n = self.free_param_names[i]
                p = self.free_params[n]
                table.add_row(h(n), str(p))
        
            console.print(table)

    def print_tree(self):
        """Prints a summary of the model parameters, in tree form."""
        tree = Tree(f"[bold italic white]Params[/bold italic white](nparam={self.nparam}, ndim={self.ndim})")
        children = list(self.children)
        names = [n for n in self.all_param_names if '/' not in n]
        for name in names:
            tree.add('[bold #FFE4B5 not italic]' + name + '[white]: [italic not bold #c9b89b]' + self.all_params[name].__repr__())
        for child in children:
            source = tree.add('[bold #6495ED not italic]' + child + '[white]: [italic not bold #6480b3]' + self.children[child].__repr__())#
            params_i = self.children[child]
            names_i = [n for n in params_i.all_param_names if '/' not in n]
            for name_i in names_i:
                source.add('[bold #FFE4B5 not italic]' + name_i + '[white]: [italic not bold #c9b89b]' + params_i.all_params[name_i].__repr__())
            children_i = list(params_i.children)
            for child_i in children_i:
                subsource = source.add('[bold #8fbc8f not italic]' + child_i + '[white]: [italic not bold #869e86]' + params_i.children[child_i].__repr__())
                params_ii = params_i.children[child_i]
                names_ii = [n for n in params_ii.all_param_names if '/' not in n]
                for name_ii in names_ii:
                    subsource.add('[bold #FFE4B5 not italic]' + name_ii + '[white]: [italic not bold #c9b89b]' + params_ii.all_params[name_ii].__repr__())
        console.print(tree)

    def update(self, new_params):
        """Updates the Params object with new_params."""
        assert set(new_params._components.keys()) == set(self._components.keys()), 'Cannot update Params object with different components'

        self.all_params.update(new_params.all_params)
        self.free_params.update(new_params.free_params)

        for component in self._components:
            self._components[component].update(new_params._components[component])

    def update_from_vector(self, names, x):
        # """Updates the free params from a flattened list of parameter values x."""

        # assert len(x) == self.ndim, 'Number of parameters in x must match number of free parameters in Params object'
        for i, name in enumerate(names):
            if name in self.free_params:
                del self.free_params[name]
            self.all_params[name] = x[i]
        
        x_components = np.array([p.split('/')[0] for p in names])
        x_names = np.array([p.removeprefix(c+'/') for p,c in zip(names, x_components)])
        for component in x_components:
            if component in self._components:
                self._components[component].update_from_vector(x_names[x_components==component], x[x_components==component])



    def validate(self, confirm=True):
        """This method checks that all required parameters are defined, 
           warns you if the code is using defaults, and define several 
           internally-used variables. 
           Runs automatically when Params is passed to Model or Fitter.
        """
      
        # Check for required parameters
        for key in self.required_params:
            if key not in self.all_params:
                raise ValueError(f"Required parameter {key} not found in Params object")

        # Check for default values
        for key in self.default_params:
            if key not in self.all_params:
                self.logger.warning(f"Using default value for parameter {key}")

        self.validated = True
        
