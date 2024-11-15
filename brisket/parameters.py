import rich, toml, os, sys
from collections.abc import MutableMapping

from brisket import config
from brisket.fitting import priors

base_params = ['redshift']
allowed_components = ['galaxy','agn','nebular','calib','igm']
defaults = {'igm':'Inoue14', 
            'galaxy': {
                'stellar_model':'BC03', 
                't_bc':0.01
                },
            'calib': {
                'f_LSF': 1.0,
                'oversample': 4,
            }}

## ADD TO DEFAULTS
# t_bc = 0.01
# if "t_bc" in list(params):
#     t_bc = params["t_bc"]

### TODO better handling of defaults
### e.g. continuity SFH: need to add defaults for dsfr1, dsfr2, etc 
### add default IGM model, added even if you don't run params.add_igm()


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
        
        
        self.sources = {}
        self.all_params = {}
        self.free_params = {}
        self.linked_params = {}
        
        #self.defaults = []


    def add_source(self, name, model=None):

        # if model is None:
        #     if name=='galaxy':
        #         model = models.BaseStellarModel
        #     if name=='agn':
        #         model = models.BaseAGNModel
            
        source = Source(name, model)
        self.__setitem__(name, source)

        # self.all_param_names += [name+'/'+n for n in source.all_names]

    def __repr__(self):
        self.validate()
        # border_chars = '═║╔╦╗╠╬╣╚╩╝'

        width = config.cols-2
        if width % 2 == 0:
            width -= 1
        if self.ndim > 0:
            outstr = config.border_chars[2] + config.border_chars[0]*width + config.border_chars[4] + '\n'
            outstr += config.border_chars[1] + 'Fixed Parameters'.center(width) + config.border_chars[1] + '\n'
            outstr += config.border_chars[5] + config.border_chars[0]*(width//2) + config.border_chars[3] + config.border_chars[0]*(width//2) + config.border_chars[7] + '\n'
        else: 
            outstr = config.border_chars[2] + config.border_chars[0]*(width//2) + config.border_chars[3] + config.border_chars[0]*(width//2) + config.border_chars[4] + '\n'
        outstr += config.border_chars[1] + 'Parameter name'.center(width//2) + config.border_chars[1] + 'Value'.center(width//2) + config.border_chars[1] + '\n'
        outstr += config.border_chars[5] + config.border_chars[0]*(width//2) + config.border_chars[6] + config.border_chars[0]*(width//2) + config.border_chars[7] + '\n'

        for i in range(self.nparam): 
            if self.all_param_names[i] in self.free_param_names: continue
            #if self.all_param_names[i] in self.defaults: df = ' *'; 
            #else: df = ''
            df = ''
            outstr += config.border_chars[1] + ' ' + (self.all_param_names[i] + df).ljust(width//2-1) + config.border_chars[1] + ' ' +  str(self.all_params[i]).ljust(width//2-1) + config.border_chars[1] + '\n'
        
        outstr += config.border_chars[8] + config.border_chars[0]*(width//2) + config.border_chars[9] + config.border_chars[0]*(width//2) + config.border_chars[10] + '\n'
        outstr += '\n'
        # msg += '(* = adopted default value) \n'
        # for i in range(p.nparams):
            # msg += (p.all_param_names[i], p.all_param_values[i])
         
        if self.ndim > 0:
            outstr += config.border_chars[2] + config.border_chars[0]*width + config.border_chars[4] + '\n'
            outstr += config.border_chars[1] + 'Free Parameters'.center(width) + config.border_chars[1] + '\n'
            outstr += config.border_chars[5] + config.border_chars[0]*(width//3) + config.border_chars[3] + config.border_chars[0]*(width//3) + config.border_chars[3] + config.border_chars[0]*(width//3) + config.border_chars[7] + '\n'
            outstr += config.border_chars[1] + 'Parameter name'.center(width//3) + config.border_chars[1] + 'Limits'.center(width//3) + config.border_chars[1] + 'Prior'.center(width//3) + config.border_chars[1] + '\n'
            outstr += config.border_chars[5] + config.border_chars[0]*(width//3) + config.border_chars[6] + config.border_chars[0]*(width//3) + config.border_chars[6] + config.border_chars[0]*(width//3) + config.border_chars[7] + '\n'
            for i in range(self.ndim): 
                # outstr += config.border_chars[1] + 'Parameter name'.center(width//2) + config.border_chars[1] + 'Prior'.center(width//2) + config.border_chars[1] + '\n'
                n = self.free_param_names[i]
                p = self.free_params[n]
                outstr += config.border_chars[1] + ' ' + (n).ljust(width//3-1) + config.border_chars[1] + ' ' +  str(p.limits).ljust(width//3-1) + config.border_chars[1] + ' ' +  str(p.prior).ljust(width//3-1) + config.border_chars[1] + '\n'

                # outstr += self.free_param_names[i].ljust(25) + '| ' + '\n'
        return outstr
        
    # def update(self, *args, **kwargs):
    #     for k, v in dict(*args, **kwargs).items():
    #         self[k] = v
    

    @property
    def nparam(self):
        return len(self.all_param_names)
    
    @property
    def ndim(self):
        return len(self.free_param_names)

    def __setitem__(self, key, value):
        if isinstance(value, FreeParam) or isinstance(value, FixedParam):    
            # setting the value of a parameter
            self.all_params[key] = value
        if isinstance(value, FreeParam):    
            self.free_params[key] = value

        elif isinstance(value, Source):    
            self.sources[key] = value

    def __getitem__(self, key):
        if key in self.sources:
            return dict.__getitem__(self.sources, key)
        elif key in self.params:
            return dict.__getitem__(self.params, key)

    def __contains__(self, key):
        return dict.__contains__(self.data, key)

    def _parse_from_toml(self, filepath):
        '''Fixes a bug in TOML where inline dictionaries are stored with some obscure DynamicInlineTableDict class instead of regular old python dict'''
        f = toml.load(filepath)
        for key in f:
            for subkey in f[key]:
                if 'DynamicInlineTableDict' in str(type(f[key][subkey])): 
                    f[key][subkey] = dict(f[key][subkey])
                if isinstance(f[key][subkey], dict):
                    for subsubkey in f[key][subkey]:
                        if 'DynamicInlineTableDict' in str(type(f[key][subkey][subsubkey])): 
                            f[key][subkey][subsubkey] = dict(f[key][subkey][subsubkey])
        return f

    def validate(self):
        '''This method checks that all required parameters are defined, 
           warns you if the code is using defaults, and define several 
           internally-used variables. 
           Runs automatically run when printing a Params object or when
           Params is passed to ModelGalaxy or Fitter.
        '''
        for source in self.sources:
            self.all_params.update(self.sources[source].all_params)
            self.free_params.update(self.sources[source].free_params)
        self.all_param_names = list(self.all_params.keys())
        self.all_param_values = list(self.all_params.values())

        self.free_param_names = list(self.free_params.keys()) # Flattened list of parameter names for free params  
        self.free_param_priors = [param.prior for param in self.free_params.values()] 

        # self.linked_params


# define class for for a source (galaxy, agn, etc) which will be a container for parameters
class Source(Params):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.all_params = {}
        self.free_params = {} 
        self.linked_params = {}

    def add_source(name, model=None):
        raise Exception('cant add source to source')

    def add_nebular(model=None):
        pass
    
    def add_dust(model=None):
        pass


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
