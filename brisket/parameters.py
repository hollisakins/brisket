import rich, toml, os, sys
from collections.abc import MutableMapping

from brisket import config
from brisket.fitting import priors
from brisket.console import console, rich_str
from rich.table import Table
from numpy import ndarray

from brisket.models import BaseStellarModel, BaseSFHModel, BaseAGNModel

model_defaults = {'galaxy': BaseStellarModel, 
                  'agn': BaseAGNModel, 
                  'igm': InoueIGMModel}


# TODO default model choices given source names
# TODO default parameter choices given source names

class Params:
    def __init__(self, template=None, file=None): #*args, **kwargs):
        
        if file is not None:
            try:
                data = self._parse_from_toml(file)
            except FileNotFoundError:
                print(f"Parameter file {data} not found."); sys.exit()
        elif template is not None:
            try:
                data = self._parse_from_toml(os.path.join(utils.param_template_dir, template+'.toml'))
            except FileNotFoundError:
                print(f"Parameter template {data} not found. Place template parameter files in the brisket/defaults/templates/."); sys.exit()
        
        self.groups = {}
        self.all_params = {}
        self.free_params = {}
        self.linked_params = {}
        self.validated = False

    def add_group(self, name, model=None, model_type=None):
        if model is None:
            for key in model_defaults:
                if key in name:
                    model = model_defaults[key]
            else:
                raise Exception(f'No default model for source {name}, please specify model')
        group = Group(name, model, parent=self, model_type=model_type)
        self.__setitem__(name, group)

    def add_source(self, name, model=None):
        group = Group(name, model, parent=self, model_type='source')
        self.__setitem__(name, group)

    def add_absorber(self, name, model=None):
        group = Group(name, model, parent=self, model_type='absorber')
        self.__setitem__(name, group)
        
    def add_reprocessor(self, name, model=None):
        group = Group(name, model, parent=self, model_type='reprocessor')
        self.__setitem__(name, group)

    def add_nebular(self, model=None):
        self.add_reprocessor('nebular', model=model)
    
    def add_dust(self, model=None):
        self.add_reprocessor('dust', model=model)
    
    def add_igm(self, model=None):
        self.add_absorber('igm', model=model)

    def __setitem__(self, key, value):

        if isinstance(value, (FreeParam,FixedParam,int,float,str,list,tuple,ndarray)): # setting the value of a parameter, add to all_params
            
            if isinstance(self, Group): # if adding a parameter to a group, prepend the group name to the key
                if isinstance(self.parent, Group):
                    key = self.parent.name + '/' + self.name + '/' + key
                else:
                    key = self.name + '/' + key
            
            if isinstance(value, (int,float,str,list,tuple,ndarray)): # for fixed parameters entered as ints or floats, convert to FixedParam
                value = FixedParam(value)

            self.all_params[key] = value
            if isinstance(value, FreeParam): # if setting a free parameter, add to free_params
                self.free_params[key] = value


        elif isinstance(value, Group): # adding a group 
            if value.model_type == 'source':
                self.sources[key] = value
            elif value.model_type == 'absorber':
                self.absorbers[key] = value
            elif value.model_type == 'reprocessor':
                self.reprocessors[key] = value
            else:
                self.groups[key] = value

    def __getitem__(self, key):
        if key in self.sources: # getting a source
            return self.sources[key]
        elif key in self.absorbers: # getting a absorber
            return self.absorbers[key]
        elif key in self.reprocessors: # getting a reprocessor
            return self.reprocessors[key]
        elif key in self.groups: # getting a group
            return self.groups[key]
        elif key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        else:
            raise Exception(f"No key {key} found in {self}")

    # def __contains__(self, key):
    #     return dict.__contains__(self.data, key)


    def __repr__(self):
        self.validate()
        if config.params_print_summary:
            return self.summary
        elif config.params_print_tree:
            return self.tree
        
    @property
    def summary(self):
        if self.ndim > 0:
            table = Table(title="Fixed Parameters")
        else:
            table = Table(title="")
        table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta", no_wrap=True)

        for i in range(self.nparam): 
            n = self.all_param_names[i]
            if n in self.free_param_names: continue
            table.add_row(n, str(self.all_params[n]))

        tab_str = rich_str(table)
                     
        if self.ndim > 0:
            table = Table(title="Free Parameters")
            table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
            table.add_column("Limits", style="magenta", no_wrap=True)
            table.add_column("Prior", style="magenta", no_wrap=True)
        
            for i in range(self.ndim): 
                n = self.free_param_names[i]
                p = self.free_params[n]
                table.add_row(n, str(p.limits), str(p.prior))
        
            tab_str = tab_str + '\n' + rich_str(table)
        return tab_str

    @property
    def tree(self):
        return ''

    # def update(self, *args, **kwargs):
    #     for k, v in dict(*args, **kwargs).items():
    #         self[k] = v
    

    @property
    def nparam(self):
        return len(self.all_param_names)

    
    @property
    def ndim(self):
        return len(self.free_param_names)

    # def _parse_from_toml(self, filepath):
    #     '''Fixes a bug in TOML where inline dictionaries are stored with some obscure DynamicInlineTableDict class instead of regular old python dict'''
    #     f = toml.load(filepath)
    #     for key in f:
    #         for subkey in f[key]:
    #             if 'DynamicInlineTableDict' in str(type(f[key][subkey])): 
    #                 f[key][subkey] = dict(f[key][subkey])
    #             if isinstance(f[key][subkey], dict):
    #                 for subsubkey in f[key][subkey]:
    #                     if 'DynamicInlineTableDict' in str(type(f[key][subkey][subsubkey])): 
    #                         f[key][subkey][subsubkey] = dict(f[key][subkey][subsubkey])
    #     return f

    def validate(self):
        '''This method checks that all required parameters are defined, 
           warns you if the code is using defaults, and define several 
           internally-used variables. 
           Runs automatically run when printing a Params object or when
           Params is passed to ModelGalaxy or Fitter.
        '''
        for source in self.sources:
            self.sources[source].model = self.sources[source]._model_func(params=self.sources[source])
            
            self.all_params.update(self.sources[source].all_params)
            self.free_params.update(self.sources[source].free_params)
            if len(self.sources[source].sources)>0:
                for subsource in self.sources[source].sources:
                    self.all_params.update(self.sources[source].sources[subsource].all_params)
                    self.free_params.update(self.sources[source].sources[subsource].free_params)
        
        for absorber in self.absorbers:
            self.absorbers[absorber].model = self.absorbers[absorber]._model_func(params=self.absorbers[absorber])
            
            self.all_params.update(self.absorbers[absorber].all_params)
            self.free_params.update(self.absorbers[absorber].free_params)

        for reprocessor in self.reprocessors:
            self.reprocessors[reprocessor].model = self.reprocessors[reprocessor]._model_func(params=self.reprocessors[reprocessor])
            
            self.all_params.update(self.reprocessors[reprocessor].all_params)
            self.free_params.update(self.reprocessors[reprocessor].free_params)

        # initialize self.sources[source].model with params=self.sources[source]
        self.all_param_names = list(self.all_params.keys())
        self.all_param_values = list(self.all_params.values())

        self.free_param_names = list(self.free_params.keys()) # Flattened list of parameter names for free params  
        self.free_param_priors = [param.prior for param in self.free_params.values()] 

        # self.linked_params

        self.validated = True


class Group(Params):
    def __init__(self, name, model, parent=None, model_type=None):
        self.name = name
        self.type = model_type
        self._model_func = model
        self.parent = parent
        self.sources = {}
        self.groups = {}
        self.absorbers = {}
        self.reprocessors = {}
        self.all_params = {}
        self.free_params = {} 
        self.linked_params = {}

    def add_source(self, name, model=None):
        raise Exception('can only add source to base Params object')

    def add_sfh(self, name, model=None):
        if not (self.name=='galaxy' and self.model_type=='source'):
            raise Exception('SFH is special, can only be added to galaxy source')
        sfh = Group(name, model=model, parent=self)
        self.__setitem__(name, sfh)

    def __repr__(self):
        if len(self.sources)>0:
            outstr = f"Source(name='{self.name}', model={self._model_func.__name__})"
            for source in self.sources:
                s = self.sources[source]
                outstr += '\n' + f"\-> Source(name='{s.name}', model={s._model_func.__name__})"
            return outstr
        else:
            return f"Source(name='{self.name}', model={self._model_func.__name__})"


class FreeParam(MutableMapping):
    def __init__(self, low, high, prior='uniform', **hyperparams):
        self.low = low
        self.high = high
        self.limits = (low, high)
        # self.hyperparams = hyperparams
        self.prior = priors.Prior((low, high), prior, **hyperparams)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return f'FreeParam({self.low}, {self.high}, {self.prior})'

class FixedParam:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'FixedParam({self.value})'

    def __str__(self):
        return str(self.value)

    def __float__(self):
        return float(self.value)
    
    def __int__(self):
        return int(self.value)


#         pass

#     def add_galaxy(self, template=None, **kwargs):
#         if template is not None: 
#             pass
#         self._parse_parameters(kwargs)
#         self.galaxy = Galaxy(**kwargs)

#     def _parse_parameters(self, kwargs):
#         param_names = list(kwargs.keys())
#         param_values = [kwargs[k] for k in kwargs.keys()]
#         nparam = len(param_names)
    
#         # Find parameters to be fitted and extract their priors.
#         for i in range(len(nparam)):
#             self.all_param_names.append(param_names[i])
#             self.all_param_values.append(param_values[i])

#             if isfree:
#                 self.free_param_names.append(param_names[i])
#                 self.free_param_limits.append(param_values[i].limits)
#                 self.free_param_pdfs.append(param_values[i].prior)
#                 self.free_param_hypers.append(param_values[i].hypers)

#             if ismirror:
#                 pass

            
#             if istransform: 
#                 pass

#                 # # Prior probability densities between these limits.
#                 # prior_key = all_keys[i] + "_prior"
#                 # if prior_key in list(all_keys):
#                 #     self.pdfs.append(all_vals[all_keys.index(prior_key)])

#                 # else:
#                 #     self.pdfs.append("uniform")

#                 # # Any hyper-parameters of these prior distributions.
#                 # self.hyper_params.append({})
#                 # for i in range(len(all_keys)):
#                 #     if all_keys[i].startswith(prior_key + "_"):
#                 #         hyp_key = all_keys[i][len(prior_key)+1:]
#                 #         self.hyper_params[-1][hyp_key] = all_vals[i]

#             # Find any parameters which mirror the value of a fit param.
#             # if all_vals[i] in all_keys:
#             #     self.mirror_pars[all_keys[i]] = all_vals[i]

#             # if all_vals[i] == "dirichlet":
#             #     n = all_vals[all_keys.index(all_keys[i][:-6])]
#             #     comp = all_keys[i].split(":")[0]
#             #     for j in range(1, n):
#             #         self.params.append(comp + ":dirichletr" + str(j))
#             #         self.pdfs.append("uniform")
#             #         self.limits.append((0., 1.))
#             #         self.hyper_params.append({})

#         # Find the dimensionality of the fit
#         self.ndim = len(self.params)

#     def update(self, kwargs):
#         for k in list(kwargs.keys()):
#             setattr(self, k, kwargs[k])
