from synthesizer import exceptions

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
    def __init__(self, name, kwargs):
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

    def __str__(self):
        """
        Print basic summary of the parameterised star formation history.
        """

        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF PARAMETERISED STAR FORMATION HISTORY" + "\n"
        pstr += str(self.__class__) + "\n"
        for parameter_name, parameter_value in self.parameters.items():
            pstr += f"{parameter_name}: {parameter_value}" + "\n"
        pstr += (
            f"median age: {self.calculate_median_age().to('Myr'):.2f}" + "\n"
        )
        pstr += f"mean age: {self.calculate_mean_age().to('Myr'):.2f}" + "\n"
        pstr += "-" * 10 + "\n"

        return pstr



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
