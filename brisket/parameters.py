import rich, toml, os, sys
from collections.abc import MutableMapping

from brisket import config
from brisket.fitting import priors
from brisket.console import log, rich_str, PathHighlighter, LimitsHighlighter
from rich.table import Table
from numpy import ndarray


# from brisket.models import BaseStellarModel, BaseSFHModel, BaseAGNModel
# model_defaults = {'galaxy': BaseStellarModel, 
                #   'agn': BaseAGNModel, 
                #   'igm': InoueIGMModel}
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
        self.components = {}
        self.component_types = []
        self.component_orders = []

        self.all_params = {}
        self.free_params = {}
        self.linked_params = {}
        self.validated = False

    def add_group(self, name, model=None, model_type=None):
        if model is None:
            model_def = None
            for key in model_defaults:
                if key in name:
                    model_def = model_defaults[key]
                    break
            if model_def is None:
                raise Exception(f'No default model for source {name}, please specify model')
            model = model_def
        group = Group(name, model, parent=self, model_type=model_type)
        self.__setitem__(name, group)

    def add_source(self, name, model=None):
        self.add_group(name, model=model, model_type='source')

    def add_absorber(self, name, model=None):
        self.add_group(name, model=model, model_type='absorber')
        
    def add_reprocessor(self, name, model=None):
        self.add_group(name, model=model, model_type='reprocessor')
        
    def add_calibrator(self, name, model=None):
        self.add_group(name, model=model, model_type='calibrator')

    # specific, commongly used models
    def add_nebular(self, model=None):
        self.add_reprocessor('nebular', model=model)
    
    def add_dust(self, model=None):
        self.add_reprocessor('dust', model=model)
    
    def add_igm(self, model=None):
        self.add_absorber('igm', model=model)
    
    def add_calibration(self, model=None):
        self.add_calibrator('calib', model=model)

    ##############################
    def __setitem__(self, key, value):

        if isinstance(value, (FreeParam,FixedParam,int,float,str,list,tuple,ndarray)): # setting the value of a parameter, add to all_params

            if isinstance(value, (int,float,str,list,tuple,ndarray)): # for fixed parameters entered as ints or floats, convert to FixedParam
                value = FixedParam(value)

            self.all_params[key] = value
            if isinstance(value, FreeParam): # if setting a free parameter, add to free_params
                self.free_params[key] = value

        elif isinstance(value, Group): # adding a group 
            self.components[key] = value

    def __getitem__(self, key):
        if key in self.components: # getting a component/group
            return self.components[key]
        elif key in self.all_params: # getting a parameter from the base Params object
            return self.all_params[key]
        else:
            raise Exception(f"No key {key} found in {self}")

    def __contains__(self, key):
        return dict.__contains__(self.all_params, key) or dict.__contains__(self.components, key)

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

        table.add_column("Parameter name", justify="left", no_wrap=True)
        table.add_column("Value", style='bold #FFE4B5', justify='center', no_wrap=True)

        h = PathHighlighter()
        l = LimitsHighlighter()
        for i in range(self.nparam): 
            n = self.all_param_names[i]
            if n in self.free_param_names: continue
            table.add_row(h(n), str(self.all_params[n]))

        tab_str = rich_str(table)
                     
        if self.ndim > 0:
            table = Table(title="Free Parameters")
            table.add_column("Parameter name", justify="left", style="cyan", no_wrap=True)
            table.add_column("Limits", style=None, justify='center', no_wrap=True)
            table.add_column("Prior", style=None, no_wrap=True)
        
            for i in range(self.ndim): 
                n = self.free_param_names[i]
                p = self.free_params[n]
                table.add_row(h(n), l(str(p.limits)), str(p.prior))
        
            tab_str = tab_str + '\n' + rich_str(table)
        return tab_str

    @property
    def tree(self):
        return ''

    

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

        # if not isinstance(self, Group): # first check if this is a Group object -- groups cannot have their own sources (TODO is this necessary? do we run validate on groups)
        
        for comp_name, comp in self.components.items():
            if comp.model_type == 'source':
                comp.model = comp._model_func(params=comp) # initialize model 
                self.component_types.append('source')
                self.component_orders.append(comp.model.order)
                self.all_params.update({comp_name+'/'+k:v for k,v in comp.all_params.items()})
                self.free_params.update({comp_name+'/'+k:v for k,v in comp.free_params.items()})

                for subcomp_name, subcomp in comp.components.items():
                    subcomp.model = subcomp._model_func(params=subcomp)
                    comp.component_types.append(subcomp.model_type)
                    comp.component_orders.append(subcomp.model.order)
                    self.all_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.all_params.items()})
                    self.free_params.update({comp_name+'/'+subcomp_name+'/'+k:v for k,v in subcomp.free_params.items()})
                    # subcomp.all_params.update({subcomp_name+'/'+k:v for k,v in subcomp.all_params.items()})
                    # subcomp.free_params.update({subcomp_name+'/'+k:v for k,v in subcomp.free_params.items()})
            else:
                comp.model = comp._model_func(params=comp) # initialize model 
                self.component_types.append(comp.model_type)
                self.component_orders.append(comp.model.order)
                self.all_params.update({comp_name+'/'+k:v for k,v in comp.all_params.items()})
                self.free_params.update({comp_name+'/'+k:v for k,v in comp.free_params.items()})

        # initialize self.sources[source].model with params=self.sources[source]
        self.all_param_names = list(self.all_params.keys())
        self.all_param_values = list(self.all_params.values())

        self.free_param_names = list(self.free_params.keys()) # Flattened list of parameter names for free params  
        self.free_param_priors = [param.prior for param in self.free_params.values()] 

        # self.linked_params

        self.validated = True

    def update(self, new_params):
        assert self.validated, 'Params object must be validated before updating'



class Group(Params):
    def __init__(self, name, model, parent=None, model_type=None):
        self.name = name
        self.model_type = model_type
        self._model_func = model
        self.parent = parent
        
        self.components = {}
        self.component_types = []
        self.component_orders = []

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
        # if len(self.sources)>0:
        #     outstr = f"Source(name='{self.name}', model={self._model_func.__name__})"
        #     for source in self.sources:
        #         s = self.sources[source]
        #         outstr += '\n' + f"\-> Source(name='{s.name}', model={s._model_func.__name__})"
        #     return outstr
        # else:
        return f"Group(name='{self.name}', model={self._model_func.__name__}, model_type={self.model_type})"

    def __getitem__(self, key):
        if key in self.components: # getting a component/group
            return self.components[key]
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
