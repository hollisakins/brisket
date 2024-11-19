

class BaseModel:
    def __init__(self, params):
        self._build_defaults(params)
        self.params = params
    
    def _build_defaults(self, params):
        print('Warning: _build_defaults not implemented')
        pass
    
class BaseGriddedModel(BaseModel):
    def _resample(self, wavelengths):
        raise NotImplementedError("Subclasses should implement this method")

class BaseFunctionalModel(BaseModel):
    def _resample(self, wavelengths):
        self.wavelengths = wavelengths

class BaseSourceModel(BaseModel):
    def emit(self, params):
        raise NotImplementedError("Subclasses should implement this method")

class BaseAbsorberModel(BaseModel):
    def absorb(self, params):
        raise NotImplementedError("Subclasses should implement this method")

class BaseReprocessorModel(BaseModel):
    def emit(self, params):
        raise NotImplementedError("Subclasses should implement this method")
    def absorb(self, params):
        raise NotImplementedError("Subclasses should implement this method")

