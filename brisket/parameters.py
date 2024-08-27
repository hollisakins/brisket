


class ModelParams(object):
    
    def __init__(self, 
                 template=None,
                 redshift=None,
                 igm='Inoue14',
                 damping=None):

        if template is not None: # a template for the full parameter set has been specified
            pass
        
        # self.components = []
        self.all_param_names = []
        self.all_param_values = []

        self.free_param_names = []   # Flattened list of parameter names for free params
        self.free_param_limits = []  # Limits for fitted parameter values
        self.free_param_pdfs = []    # Probability densities within lims
        self.free_param_hypers = []  # Hyperparameters of prior distributions



        pass

    def add_galaxy(self, template=None, **kwargs):
        if template is not None: 
            pass
        self._parse_parameters(kwargs)
        self.galaxy = Galaxy(**kwargs)

    def _parse_parameters(self, kwargs):
        param_names = list(kwargs.keys())
        param_values = [kwargs[k] for k in kwargs.keys()]
        nparam = len(param_names)
    
        # Find parameters to be fitted and extract their priors.
        for i in range(len(nparam)):
            self.all_param_names.append(param_names[i])
            self.all_param_values.append(param_values[i])

            if isfree:
                self.free_param_names.append(param_names[i])
                self.free_param_limits.append(param_values[i].limits)
                self.free_param_pdfs.append(param_values[i].prior)
                self.free_param_hypers.append(param_values[i].hypers)

            if ismirror:
                pass

            
            if istransform: 
                pass

                # # Prior probability densities between these limits.
                # prior_key = all_keys[i] + "_prior"
                # if prior_key in list(all_keys):
                #     self.pdfs.append(all_vals[all_keys.index(prior_key)])

                # else:
                #     self.pdfs.append("uniform")

                # # Any hyper-parameters of these prior distributions.
                # self.hyper_params.append({})
                # for i in range(len(all_keys)):
                #     if all_keys[i].startswith(prior_key + "_"):
                #         hyp_key = all_keys[i][len(prior_key)+1:]
                #         self.hyper_params[-1][hyp_key] = all_vals[i]

            # Find any parameters which mirror the value of a fit param.
            # if all_vals[i] in all_keys:
            #     self.mirror_pars[all_keys[i]] = all_vals[i]

            # if all_vals[i] == "dirichlet":
            #     n = all_vals[all_keys.index(all_keys[i][:-6])]
            #     comp = all_keys[i].split(":")[0]
            #     for j in range(1, n):
            #         self.params.append(comp + ":dirichletr" + str(j))
            #         self.pdfs.append("uniform")
            #         self.limits.append((0., 1.))
            #         self.hyper_params.append({})

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def update(self, kwargs):
        for k in list(kwargs.keys()):
            setattr(self, k, kwargs[k])

class Galaxy(ModelParams):
    def __init__(self, stellar_model, logMstar, metallicity, sfh, **sfh_kwargs):

        pass
