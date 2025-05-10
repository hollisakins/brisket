from ..utils import exceptions
from ..parameters import Params

__all__ = [
    "EmitterModel",
    "AbsorberModel",
    "ReprocessorModel",
]

class EmitterModel:
    """
    Base class for all Emitter models.
    """

    def __init__(self, verbose=False, **kwargs):
        raise exceptions.UnimplementedFunctionality(
            "`EmitterModel` should not be initialized directly. "
            "Please use subclassed models."
        )
                
    def validate(self, kwargs):
        params = Params()
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']
        if 'redshift' in kwargs:
            raise KeyError("Redshift should never be passed to an individual model.")
        return params

    def __repr__(self):
        s = f"{self.__class__.__name__}()"
        return s

    def __add__(self, other):
        """
        Add two models together.
        """
        if isinstance(other, EmitterModel):
            return self._add_emitter(other)
        elif isinstance(other, AbsorberModel):
            return self._add_absorber(other)
        elif isinstance(other, ReprocessorModel):
            return self._add_reprocessor(other)
        else:
            raise TypeError(f"Cannot add {type(other)} with {type(self)}")

    def _add_emitter(self, emitter):
        return Formula(base=self, add=emitter)

    def _add_absorber(self, absorber):
        raise exceptions.UnimplementedFunctionality(
            "In general, adding `absorbers` to `emitters` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the * operator to absorb the SED."
        )

    def _add_reprocessor(self, reprocessor):
        raise exceptions.UnimplementedFunctionality(
            "In general, adding `reprocessors` to `emitters` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the % operator to reprocess the SED."
        )


    def __mul__(self, other):
        """
        Add two models together.
        """
        if isinstance(other, EmitterModel):
            return self._mul_emitter(other)
        elif isinstance(other, AbsorberModel):
            return self._mul_absorber(other)
        elif isinstance(other, ReprocessorModel):
            return self._mul_reprocessor(other)
        else:
            raise TypeError(f"Cannot multiply {type(other)} with {type(self)}")
        
    def _mul_emitter(self, emitter):
        raise exceptions.UnimplementedFunctionality(
            "In general, multiplying `emitters` with other `emitters` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the + operator to combine the SEDs."
        )

    def _mul_absorber(self, absorber):
        return Formula(base=self, mul=absorber)

    def _mul_reprocessor(self, reprocessor):
        raise exceptions.UnimplementedFunctionality(
            "In general, multiplying `reprocessors` with `emitters` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the % operator to reprocess the SED."
        )

    def __mod__(self, other):
        """
        Modulus operator for models, used for reprocessing models.
        """
        if isinstance(other, EmitterModel):
            return self._mod_emitter(other)
        elif isinstance(other, AbsorberModel):
            return self._mod_absorber(other)
        elif isinstance(other, ReprocessorModel):
            return self._mod_reprocessor(other)
        else:
            raise TypeError(f"Cannot mod {type(other)} with {type(self)}")
        
    def _mod_emitter(self, emitter):
        raise exceptions.UnimplementedFunctionality(
            "In general, reprocessing `emitters` with other `emitters` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the + operator to combine the SEDs."
        )

    def _mod_absorber(self, absorber):
        raise exceptions.UnimplementedFunctionality(
            "In general, reprocessing `emitters` with `absorbers` is not implemented. "
            "Certain models may use this functionality, but it is not guaranteed to work. "
            "Please use the * operator to absorb the SED."
        )

    def _mod_reprocessor(self, reprocessor):
        return Formula(base=self, mod=reprocessor)
    
    def __call__(self, other):
        return self.__mod__(other)


class AbsorberModel:
    """
    Base class for all Absorber models.
    """

    def __init__(self, verbose=False, **kwargs):
        raise exceptions.UnimplementedFunctionality(
            "`AbsorberModel` should not be initialized directly. "
            "Please use subclassed models."
        )
        
    def validate(self, kwargs):
        params = Params()
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']
        if 'redshift' in kwargs:
            raise KeyError("Redshift should never be passed to an individual model.")
        return params

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ReprocessorModel:
    """
    Base class for all Reprocessor models.
    """

    def __init__(self, verbose=False, **kwargs):
        raise exceptions.UnimplementedFunctionality(
            "`ReprocessorModel` should not be initialized directly. "
            "Please use the subclassed models."
        )
        
    def validate(self, kwargs):
        params = Params()
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']
        if 'redshift' in kwargs:
            raise KeyError("Redshift should never be passed to an individual model.")
        return params


    def __repr__(self):
        return f"{self.__class__.__name__}()"

