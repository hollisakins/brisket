from .parameters import Parameter
from .observation import Photometry, Spectrum
from .fitting.fitter import Fitter

from rich.traceback import install
install(show_locals=False)
