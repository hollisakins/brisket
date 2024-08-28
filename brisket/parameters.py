import rich, toml, os, sys
from brisket import utils

# from collections import UserDict
base_params = ['redshift', 'igm']
allowed_components = ['galaxy','agn']
defaults = {'igm':'Inoue14'}


class ParamDict:
    def __init__(self, data): #*args, **kwargs):
        if isinstance(data, str):
            if data.endswith('.toml'):
                try:
                    data = toml.load(data)
                    data.update(data['base'])
                    del data['base']
                except FileNotFoundError:
                    print(f"Parameter file {data} not found."); sys.exit()
            else:
                try:
                    data = toml.load(os.path.join(utils.param_template_dir, data + '.toml'))
                    data.update(data['base'])
                    del data['base']
                except FileNotFoundError:
                    print(f"Parameter template {data} not found. Place template parameter files in the brisket/defaults/templates/."); sys.exit()
            
        # if 'template' in kwargs:
        #     return load_from_toml(os.path.join(utils.param_template_dir, kwargs['template'] + '.toml'))

        self.components = []
        self.all_param_names = []
        self.all_param_values = []
        self.free_param_names = []   # Flattened list of parameter names for free params
        self.free_param_limits = []  # Limits for fitted parameter values
        self.free_param_pdfs = []    # Probability densities within lims
        self.free_param_hypers = []  # Hyperparameters of prior distributions
        self.defaults = []


        self.data = data
        for key in base_params:
            if key in data:
                self.add_param(key, data[key])
            else:
                self.defaults.append(key)
                self.add_param(key, defaults[key])

        for key in allowed_components:
            if not key in data: continue;
            val = data[key]
            assert isinstance(val, dict), f'Key `{key}` must provide component parameters as a dictionary'

            self.components.append(key)
            for subkey in val:
                if isinstance(val[subkey], dict):
                    if not ('low' in val[subkey]) and not ('high' in val[subkey]):
                        for subsubkey in val[subkey]:
                            self.add_param(f'{key}:{subkey}:{subsubkey}', val[subkey][subsubkey])
                    else:
                        self.add_param(f'{key}:{subkey}', val[subkey])
                else:
                    self.add_param(f'{key}:{subkey}', val[subkey])

        for key in data:
            if key not in allowed_components and key not in base_params:            
                msg = f"Key `{key}` is not a recognized base-level parameter or model component"
                msg += f"; supported base-level parameters are: {', '.join(map(repr, base_params))}"
                msg += f", supported model components are: {', '.join(map(repr, allowed_components))}."
                raise KeyError(msg) 

    def __repr__(self):
        # dictrepr = dict.__repr__(self.data)
        # msg = '%s(%s)' % (type(self).__name__, dictrepr)
        if self.ndim > 0:
            msg = '#'*34 + ' Fixed Parameters ' + '#' * 34 + '\n'
        else: 
            msg = ''
        msg += 'Parameter name           | Value \n'
        msg += '-'*25 + '+' + '-'*60 + '\n'
        for i in range(self.nparam): 
            if self.all_param_names[i] in self.free_param_names: continue
            if self.all_param_names[i] in self.defaults: df = ' *'; 
            else: df = ''
            msg += (self.all_param_names[i] + df).ljust(25) + '| ' + str(self.all_param_values[i]).ljust(10) + '\n'
        msg += '(* = adopted default value) \n'
        # for i in range(p.nparams):
            # msg += (p.all_param_names[i], p.all_param_values[i])
         
        if self.ndim > 0:
            msg += '\n'
            msg += '#'*34 + ' Free Parameters ' + '#'*34 + '\n'
            msg += 'Parameter name           | Limits       | Prior        | Hyper parameters  \n'
            msg += '-'*25 + '+' + '-'*14 + '+'+ '-'*14 + '+'+ '-'*29 + '\n'
            for i in range(self.ndim): 
                msg += self.free_param_names[i].ljust(25) + '| ' + str(self.free_param_limits[i]).ljust(13) + '| '+ self.free_param_pdfs[i].ljust(13) + '| ' + str(self.free_param_hypers[i]) + '\n'
        return msg
        
    # def update(self, *args, **kwargs):
    #     for k, v in dict(*args, **kwargs).items():
    #         self[k] = v
    
    def add_param(self, key, val):
        self.all_param_names.append(key)
        self.all_param_values.append(val)
        if isinstance(val, dict):
            assert ('low' in val) and ('high' in val)
            self.free_param_names.append(key)
            self.free_param_limits.append((val['low'], val['high']))
            pdf = 'Uniform'
            if 'prior' in val:
                pdf = val['prior']
            self.free_param_pdfs.append(pdf)
            self.free_param_hypers.append({k:val[k] for k in val if k not in ['low','high','prior']})

    @property
    def nparam(self):
        return len(self.all_param_names)
    
    @property
    def ndim(self):
        return len(self.free_param_names)



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