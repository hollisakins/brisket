from .parameters import Params
from .fitting import priors
from .models import core


from .models.core import Model
from .observation import Observation, Photometry, Spectrum

from rich.traceback import install
install(show_locals=False)
