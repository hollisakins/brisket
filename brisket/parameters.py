import rich, toml, os, sys
from collections.abc import MutableMapping

from brisket import config
from brisket.fitting import priors
from brisket.console import log, rich_str, PathHighlighter, LimitsHighlighter
from rich.table import Table
from rich.tree import Tree
import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

from brisket.models import PowerlawAccrectionDiskModel, InoueIGMModel
model_defaults = {'agn':PowerlawAccrectionDiskModel, 
                  'igm':InoueIGMModel}


# TODO default model choices given source names
# TODO default parameter choices given source names

class Params:
    def __init__(self, template=None, file=None): #*args, **kwargs):
        
        if file is not None:
            try:
                data = self._parse_from_toml(file)
            except FileNotFoundError:
                log(f"Parameter file {data} not found."); sys.exit()
        elif template is not None:
            try:
                data = self._parse_from_toml(os.path.join(utils.param_template_dir, template+'.toml'))
            except FileNotFoundError:
                log(f"Parameter template {data} not found. Place template parameter files in the brisket/defaults/templates/."); sys.exit()
        
        # self.sources = {}
        # self.absorbers = {}
        # self.reprocessors = {}
        # self.calibrators = {}
        self._components = {}
        self._component_types = {}
        self._component_orders = {}

        self.all_params = {}
        self.free_params = {}
        self.linked_params = {}
        self.validated = False

    def add_group(self, name, model=None):
        if model is None:
            model_def = None
            for key in model_defaults:
                if key in name:
                    model_def = model_defaults[key]
                    break
            if model_def is None:
                raise Exception(f'No default model for source {name}, please specify model')
            model = model_def
        group = Group(name, model, parent=self)
        self.__setitem__(name, group)

    add_source = add_group

    # specific, commongly used models
    def add_nebular(self, model=None):
        self.add_group('nebular', model=model)

    def add_dust(self, model=None):
        self.add_group('dust', model=model)
    
    def add_igm(self, model=None):
        self.add_group('igm', model=model)
        
    def add_calibration(self, model=None):
        self.add_group('calib', model=model)

    ##############################
    def __setitem__(self, key, value):

        ### adding a parameter
        if isinstance(value, (FreeParam,FixedParam,int,float,str,list,tuple,np.ndarray)): # setting the value of a parameter, add to all_params
            if isinstance(value, (int,float,str,list,tuple,np.ndarray)): # for fixed parameters entered as ints or floats, convert to FixedParam
                value = FixedParam(value)
            
            if isinstance(self, Params):
                # just need to add the parameter itself, no need to update anything else
                self.all_params[key] = value
                if isinstance(value, FreeParam): # if setting a free parameter, add to free_params
                    self.free_params[key] = value
            
            if isinstance(self, Group): # add the parameter to the Group, add prefixed parameter to parent
                self.all_params[key] = value
                if isinstance(value, FreeParam): # if setting a free parameter, add to free_params
                    self.free_params[key] = value
                self.parent.__setitem__(self.name + '/' + key, value)

        elif isinstance(value, Group): # adding a group, add to self._components
            # assert isintance(self, Params) or self.model_type=='source', ""
            self._components[key] = value
            self._component_types[key] = value.model.type
            self._component_orders[key] = value.model.order
            # self.all_params.update({key+'/'+k:v for k,v in value.all_params.items()})
            # self.free_params.update({key+'/'+k:v for k,v in value.free_params.items()})
            
            # if isinstance(self, Group):
            #     for subcomp_name, subcomp in value.components.items():
            #         subcomp.model = subcomp.model(params=subcomp)
            #         comp.component_types.append(subcomp.model_type)
            #         comp.component_orders.append(subcomp.model.order)
            #         self.parent.all_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.all_params.items()})
            #         self.parent.free_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.free_params.items()})

            # # initialize self.sources[source].model with params=self.sources[source]


    @property 
    def free_param_names(self):
        return list(self.free_params.keys())
    @property 
    def free_param_priors(self):
        return [param.prior for param in self.free_params.values()] 
    @property
    def all_param_names(self):
        return list(self.all_params.keys())
    @property 
    def all_param_values(self):
        return list(self.all_params.values())

    @property
    def component_names(self):
        return sorted(list(self._components.keys()), key=self._component_orders.__getitem__)
    
    @property
    def component_types(self):
        return {k:self._component_types[k] for k in self.component_names}
    
    @property
    def components(self):
        return {k:self._components[k] for k in self.component_names}

            # self.free_param_priors = [param.prior for param in self.free_params.values()] 


    def __getitem__(self, key):
        if key in self._components: # getting a component/group
            return self._components[key]
        elif key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        else:
            raise Exception(f"No key {key} found in {self}")

    def __contains__(self, key):
        return dict.__contains__(self.all_params, key) or dict.__contains__(self._components, key)

    def __repr__(self):
        # self.validate()
        if config.params_print_summary:
            out = self.summary
        elif config.params_print_tree:
            out = self.tree
        out += f"Number of parameters: {self.nparam}"
        out += '\n' + f"Dimensionality: {self.ndim}"

        out += '\n'
        return out
        
    @property
    def summary(self):
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

            tab_str = rich_str(table)
        else:
            tab_str = ''
                     
        if self.ndim > 0:
            table = Table(title="Free Parameters")
            table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
            table.add_column("Limits", style=None, justify='left', no_wrap=True)
            table.add_column("Prior", style=None, no_wrap=True)
        
            for i in range(self.ndim): 
                n = self.free_param_names[i]
                p = self.free_params[n]
                table.add_row(h(n), l(str(p.limits)), str(p.prior))
        
            tab_str = tab_str + '\n' + rich_str(table)
        return tab_str

    @property
    def tree(self):
        tree = Tree("Params", style='bold italic white')
        comps = list(self.components.keys())
        names = [n for n in self.all_param_names if '/' not in n]
        for name in names:
            tree.add('[bold #FFE4B5 not italic]' + name + '[white]: [italic not bold #c9b89b]' + self.all_params[name].__repr__())
        for comp in comps:
            source = tree.add('[bold #6495ED not italic]' + comp + '[white]: [italic not bold #6480b3]' + self.components[comp].__repr__())#
            params_i = self.components[comp]
            names_i = [n for n in params_i.all_param_names if '/' not in n]
            for name_i in names_i:
                source.add('[bold #FFE4B5 not italic]' + name_i + '[white]: [italic not bold #c9b89b]' + params_i.all_params[name_i].__repr__())
            comps_i = list(params_i.components.keys())
            for comp_i in comps_i:
                subsource = source.add('[bold #8fbc8f not italic]' + comp_i + '[white]: [italic not bold #869e86]' + params_i.components[comp_i].__repr__())
                params_ii = params_i.components[comp_i]
                names_ii = [n for n in params_ii.all_param_names if '/' not in n]
                for name_ii in names_ii:
                    subsource.add('[bold #FFE4B5 not italic]' + name_ii + '[white]: [italic not bold #c9b89b]' + params_ii.all_params[name_ii].__repr__())
        return rich_str(tree)

    

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

    # def validate(self):
    #     '''This method checks that all required parameters are defined, 
    #        warns you if the code is using defaults, and define several 
    #        internally-used variables. 
    #        Runs automatically run when printing a Params object or when
    #        Params is passed to ModelGalaxy or Fitter.
    #     '''

    #     # if not isinstance(self, Group): # first check if this is a Group object -- groups cannot have their own sources (TODO is this necessary? do we run validate on groups)
        
    #     for comp_name, comp in self.components.items():
    #         if comp.model_type == 'source':
    #             comp.model = comp.model(params=comp) # initialize model 
    #             self.component_types.append('source')
    #             self.component_orders.append(comp.model.order)
    #             self.all_params.update({comp_name+'/'+k:v for k,v in comp.all_params.items()})
    #             self.free_params.update({comp_name+'/'+k:v for k,v in comp.free_params.items()})

    #             for subcomp_name, subcomp in comp.components.items():
    #                 subcomp.model = subcomp.model(params=subcomp)
    #                 comp.component_types.append(subcomp.model_type)
    #                 comp.component_orders.append(subcomp.model.order)
    #                 self.all_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.all_params.items()})
    #                 self.free_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.free_params.items()})
    #                 # subcomp.all_params.update({subcomp_name+'/'+k:v for k,v in subcomp.all_params.items()})
    #                 # subcomp.free_params.update({subcomp_name+'/'+k:v for k,v in subcomp.free_params.items()})
    #         else:
    #             comp.model = comp.model(params=comp) # initialize model 
    #             self.component_types.append(comp.model_type)
    #             self.component_orders.append(comp.model.order)
    #             self.all_params.update({comp_name+'/'+k:v for k,v in comp.all_params.items()})
    #             self.free_params.update({comp_name+'/'+k:v for k,v in comp.free_params.items()})

    #     # initialize self.sources[source].model with params=self.sources[source]
    #     self.all_param_names = list(self.all_params.keys())
    #     self.all_param_values = list(self.all_params.values())

    #     self.free_param_names = list(self.free_params.keys()) # Flattened list of parameter names for free params  
    #     self.free_param_priors = [param.prior for param in self.free_params.values()] 

    #     # self.linked_params

    #     self.validated = True

    def update(self, new_params):
        assert set(new_params._components.keys()) == set(self._components.keys()), 'Cannot update Params object with different components'

        self.all_params.update(new_params.all_params)
        self.free_params.update(new_params.free_params)

        for component in self._components:
            self._components[component].update(new_params._components[component])





class Group(Params):
    def __init__(self, name, model, parent=None):
        self.name = name
        self.model = model
        self.parent = parent
        
        self._components = {}
        self._component_types = {}
        self._component_orders = {}

        self.all_params = {}
        self.free_params = {} 
        self.linked_params = {}

    def add_source(self, name, model=None):
        raise Exception('can only add source to base Params object')

    def add_sfh(self, name, model=None):
        if not (self.name=='galaxy' and self.model.type=='source'):
            raise Exception('SFH is special, can only be added to galaxy source')
        sfh = Group(name, model=model, parent=self)
        self.__setitem__(name, sfh)

    def __repr__(self):
        # if len(self.sources)>0:
        #     outstr = f"Source(name='{self.name}', model={self.model.__name__})"
        #     for source in self.sources:
        #         s = self.sources[source]
        #         outstr += '\n' + f"\-> Source(name='{s.name}', model={s.model.__name__})"
        #     return outstr
        # else:
        return f"Group(name='{self.name}', model={self.model.__name__})"

    def __getitem__(self, key):
        if key in self._components: # getting a component/group
            return self._components[key]
        elif key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        elif key == 'redshift':
            return self.parent['redshift']
        else:
            raise Exception(f"No key {key} found in {self}")



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
