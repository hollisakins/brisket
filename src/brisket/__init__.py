from .parameters import Parameter
from .observation import Photometry, Spectrum
from .fitting.fitter import Fitter
from .fitting.results import FitResults

from rich.traceback import install
install(show_locals=False)
