from synthesizer import exceptions
from scipy.stats import beta, t, skewnorm

__all__ = ["Uniform", "LogUniform"]

class Common:
    """
    The parent class for all SFH parametrisations.

    Attributes:
        name (string)
            The name of this SFH. This is set by the child and encodes
            the type of the SFH. Possible values are defined in
            parametrisations above.
        parameters (dict)
            A dictionary containing the parameters of the model.
    """
    def __init__(self, name, **kwargs):
        """
        Initialise the parent.

        Args:
            name (string)
                The name of this SFH. This is set by the child and encodes
                the type of the SFH. Possible values are defined in
                parametrisations above.
        """

        # Set the name string
        self.name = name

        # Store the model parameters (defined as kwargs)
        self.parameters = kwargs

    def _ppf(self, age):
        """
        Prototype for child defined PPFs (percent point functions).
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            "How did you get here!?"
        )

    def sample(self, n_samples=1):
        return self._ppf(np.random.rand(n_samples))

    def plot(self, show=True, save=False, **kwargs):
        pass

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.parameters.items()])
        return f"{self.name}({params})"

    def __add__(self, other):
        if isinstance(other, PriorVolume):
            return PriorVolume([self] + other.priors)
        else:
            return PriorVolume([self, other])


class Uniform(Common):
    """
    A uniform prior between some limits.

    Attributes:
        low (unyt_quantity)
        high (unyt_quantity)
    """

    def __init__(self, low, high):
        # Initialise the parent
        Common.__init__(self, name="Uniform", low=low, high=high)

        # Set the model parameters
        self.low = low
        self.high = high

    def _ppf(prob):
        '''Returns the value where the CDF = prob.'''



class LogUniform(Common):
    """
    A uniform prior, in log_10 space, between some limits.

    Attributes:
        low (unyt_quantity)
        high (unyt_quantity)
    """

    def __init__(self, low, high):
        # Initialise the parent
        Common.__init__(self, name="LogUniform", low=low, high=high)

        # Set the model parameters
        self.low = low
        self.high = high

    def _ppf(prob):
        '''Returns the value where the CDF = prob.'''
        return np.power(10.,(np.log10(self.high/self.low))*prob + np.log10(self.low))

class StudentsT(Common):
    """
    Student's t-distribution.

    Attributes:
        low (unyt_quantity)
        high (unyt_quantity)
    """

    def __init__(self, low, high, loc=0, scale=0.3, df=2):
        # Initialise the parent
        Common.__init__(self, name="StudentsT", low=low, high=high, loc=loc, scale=scale, df=df)

        # Set the model parameters
        self.low = low
        self.high = high
        self.loc = loc
        self.scale = scale
        self.df = df

    def _ppf(prob):
        '''Returns the value where the CDF = prob.'''
        return np.power(10.,(np.log10(self.high/self.low))*prob + np.log10(self.low))

    def _ppf(prob):
        umin = t.cdf(self.low, df=self.df, scale=self.scale, loc=self.loc)
        umax = t.cdf(self.high, df=self.df, scale=self.scale, loc=self.loc)
        return t.ppf((umax-umin)*prob + umin, df=self.df, scale=self.scale, loc=self.loc)
        




class PriorVolume(object):
    """ A class which allows for samples to be drawn from a joint prior
    distribution in several parameters and for transformations from the
    unit cube to the prior volume.

    Parameters
    ----------
    priors : list of priors (instances of harmonizer.fitting.priors.Common)
    """

    def __init__(self, priors):
        self.priors = priors
        self.ndim = len(priors)
    
    def __add__(self, other):
        if isinstance(other, PriorVolume):
            return PriorVolume(self.priors + other.priors)
        else:
            return PriorVolume(self.priors + [other])

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
            params[i] = self.priors[i](params[i])

        return params

    def __repr__(self):
        return f"PriorVolume({self.priors})"