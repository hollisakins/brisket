"""
Custom exceptions for the brisket package.
"""


class BrisketError(Exception):
    """Base exception class for brisket package."""
    pass


class InconsistentParameter(BrisketError):
    """Raised when configuration parameters are inconsistent or invalid."""
    pass


class ModelError(BrisketError):
    """Raised when there are issues with model setup or evaluation."""
    pass


class DataError(BrisketError):
    """Raised when there are issues with observational data."""
    pass


class FilterError(BrisketError):
    """Raised when there are issues with filter handling."""
    pass


class FittingError(BrisketError):
    """Raised when there are issues during model fitting."""
    pass