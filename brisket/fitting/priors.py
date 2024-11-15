from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.special import erf, erfinv, hyp2f1
from scipy.stats import beta, t, skewnorm


def dirichlet(r, alpha):
    """ This function samples from a Dirichlet distribution based on N-1
    independent random variables (r) in the range (0, 1). The method is
    that of http://www.arxiv.org/abs/1010.3436 by Michael Betancourt."""

    n = r.shape[0]+1
    x = np.zeros(n)
    z = np.zeros(n-1)
    alpha_tilda = np.zeros(n-1)

    if isinstance(alpha, (float, int)):
        alpha = np.repeat(alpha, n)

    for i in range(n-1):
        alpha_tilda[i] = np.sum(alpha[i+1:])

        z[i] = beta.ppf(r[i], alpha_tilda[i], alpha[i])

    for i in range(n-1):
        x[i] = np.prod(z[:i])*(1-z[i])

    x[-1] = np.prod(z)

    return np.cumsum(x)


class PriorVolume(object):
    """ A class which allows for samples to be drawn from a joint prior
    distribution in several parameters and for transformations from the
    unit cube to the prior volume.

    Parameters
    ----------

    limits : list of tuples
        List of tuples containing lower and upper limits for the priors
        on each parameter.

    pdfs : list
        List of prior probability density functions which the parameters
        should be drawn from between the above limits.

    hyper_params : list of dicts
        Dictionaries containing fixed values for any hyper-parameters of
        the above prior distributions.
    """

    def __init__(self, priors):
        self.priors = priors
        self.ndim = len(priors)

    def sample(self):
        """ Sample from the prior distribution. """
        cube = np.random.rand(self.ndim)
        return self.transform(cube)

    def transform(self, cube, ndim=0, nparam=0):
        """ Transform numbers on the unit cube to the prior volume. """
        if type(cube)==np.ndarray: # ultranest fails when the output overwrites the input
            params = cube.copy()
        else:
            params = cube
            
        # Call the relevant prior functions to draw random values.
        for i in range(self.ndim):
            params[i] = self.priors[i].sample(params[i])

        return params


class Prior(object):
    def __init__(self, limits, prior_type, **hyper_params):
        self.limits = limits
        self.prior_type = prior_type
        self.hyper_params = hyper_params
        self.prior_function = self._parse_prior_function(prior_type)

    def evaluate(self, value):
        return self.prior_function(value, self.limits, self.hyper_params)

    def __repr__(self):
        if len(self.hyper_params) > 0:
            return f'{self.prior_function.__name__}({self.limits[0]}, {self.limits[1]}, {", ".join(f"{key}={value}" for key, value in self.hyper_params.items())})'
        else:
            return f'{self.prior_function.__name__}({self.limits[0]}, {self.limits[1]})'
    
    def __str__(self):
        return self.__repr__()

    ########################################################################
    def Uniform(self, value, limits, hyper_params):
        ''''Uniform prior between limits. Returns the x-coord associated with CDF=value.'''
        return limits[0] + (limits[1] - limits[0])*value

    def LogUniform(self, value, limits, hyper_params):
        '''Uniform prior in log10(x).'''
        return np.power(10.,(np.log10(limits[1]/limits[0]))*value + np.log10(limits[0]))

    def LnUniform(self, value, limits, hyper_params):
        '''Uniform prior in ln(x).'''
        return np.exp((np.log(limits[1]/limits[0]))*value + np.log(limits[0]))

    # def pow_10(self, value, limits, hyper_params):
    #     """ Uniform prior in 10**x where x is the parameter. """
    #     value = np.log10((10**limits[1] - 10**limits[0])*value + 10**limits[0])
    #     return value

    # def recip(self, value, limits, hyper_params):
    #     value = 1./((1./limits[1] - 1./limits[0])*value + 1./limits[0])
    #     return value

    # def recipsq(self, value, limits, hyper_params):
    #     """ Uniform prior in 1/x**2 where x is the parameter. """
    #     value = 1./np.sqrt((1./limits[1]**2 - 1./limits[0]**2)*value
    #                         + 1./limits[0]**2)
    #     return value

    def Norm(self, value, limits, hyper_params):
        """ Gaussian prior between limits with specified mu and sigma. """
        mu = hyper_params["mu"]
        sigma = hyper_params["sigma"]

        uniform_max = erf((limits[1] - mu)/np.sqrt(2)/sigma)
        uniform_min = erf((limits[0] - mu)/np.sqrt(2)/sigma)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = sigma*np.sqrt(2)*erfinv(value) + mu

        return value

    def LogNorm(self, value, limits, hyper_params):
        """ Gaussian prior between limits with specified mu and sigma. """
        mu = hyper_params["mu"]
        sigma = hyper_params["sigma"]
        
        uniform_max = erf((np.log10(limits[1]) - mu)/np.sqrt(2)/sigma)
        uniform_min = erf((np.log10(limits[0]) - mu)/np.sqrt(2)/sigma)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = np.power(10., sigma*np.sqrt(2)*erfinv(value) + mu)

        return value


    def GenNorm(self, value, limits, hyper_params):
        """Generalized Normal Distribution with shape parameter beta"""
        mu = hyper_params["mu"]
        sigma = hyper_params['sigma']
        beta = hyper_params["beta"]

        uniform_max = gennorm.cdf(limits[1], loc=mu, scale=sigma, beta=beta)
        uniform_min = gennorm.cdf(limits[0], loc=mu, scale=sigma, beta=beta)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = gennorm.ppf(value, loc=mu, scale=sigma, beta=beta)

        return value


    def SkewNorm(self, value, limits, hyper_params):
        """Generalized Normal Distribution with shape parameter beta"""
        mu = hyper_params["mu"]
        sigma = hyper_params['sigma']
        a = hyper_params["a"]

        uniform_max = skewnorm.cdf(limits[1], loc=mu, scale=sigma, a=a)
        uniform_min = skewnorm.cdf(limits[0], loc=mu, scale=sigma, a=a)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = skewnorm.ppf(value, loc=mu, scale=sigma, a=a)

        return value

    def LogSkewNorm(self, value, limits, hyper_params):
        mu = hyper_params["mu"]
        sigma = hyper_params["sigma"]
        a = hyper_params["a"]

        uniform_max = skewnorm.cdf(np.log10(limits[1]), loc=mu, scale=sigma, a=a)
        uniform_min = skewnorm.cdf(np.log10(limits[0]), loc=mu, scale=sigma, a=a)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = np.power(10., skewnorm.ppf(value, loc=mu, scale=sigma, a=a))

        return value


    def student_t(self, value, limits, hyper_params):

        if "loc" in list(hyper_params):
            loc = hyper_params["loc"]
        else:
            loc = 0
            
        if "df" in list(hyper_params):
            df = hyper_params["df"]
        else:
            df = 2.

        if "scale" in list(hyper_params):
            scale = hyper_params["scale"]
        else:
            scale = 0.3

        uniform_min = t.cdf(limits[0], df=df, scale=scale, loc=loc)
        uniform_max = t.cdf(limits[1], df=df, scale=scale, loc=loc)

        value = (uniform_max-uniform_min)*value + uniform_min

        value = t.ppf(value, df=df, scale=scale, loc=loc)

        return value


    def custom(self, value, limits, hyper_params):
        path = hyper_params['path']
        x,y = np.loadtxt(path).T
        cdf = np.cumsum(y)/np.sum(y)
        uniform_max = np.interp(limits[1], x, cdf)
        uniform_min = np.interp(limits[0], x, cdf)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = np.interp(value, cdf, x)
        return value
        
    def _parse_prior_function(self, pdf):
        if pdf in ['Uniform','Unif','uniform','unif']: return self.Uniform
        elif pdf in ['log_10','log10','log','log_uniform','log_unif','loguniform','LogUniform','LogUnif']: 
            if 0 in self.limits:
                raise ValueError('LogUniform prior cannot have 0 as a limit')
            return self.LogUniform
        elif pdf in ['log_e','loge','ln']: 
            if 0 in self.limits:
                raise ValueError('LogUniform prior cannot have 0 as a limit')
            return self.LnUniform
        # elif pdf in ['pow_10','pow10']: return self.pow_10
        # elif pdf in ['recip','Recip']: return self.recip
        # elif pdf in ['recipsq','Recipsq']: return self.recipsq
        elif pdf in ['Gaussian','Normal','normal','norm','Norm','gaussian','gauss','Gauss']: return self.Norm
        elif pdf in ['LogNorm','LogNormal','LogGaussian','lognorm','lognormal','loggauss','loggaussian']: return self.LogNorm
        elif pdf in ['GenNorm', 'gennorm']: return self.GenNorm
        elif pdf in ['SkewNorm', 'skewnorm']: return self.SkewNorm
        elif pdf in ['LogSkewNorm', 'logskewnorm']: return self.LogSkewNorm
        elif pdf in ['student_t', 't']: return self.student_t
        elif pdf in ['custom']: return self.custom
        else:
            msg = f'prior {pdf} not understood'
            raise KeyError(msg)
        