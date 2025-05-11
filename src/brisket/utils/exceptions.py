"""
Define custom exceptions.
"""

__all__ = [
    'InconsistentParameter',
    'MisspelledParameter',
    'MissingParameter',
]

class InconsistentParameter(Exception):
    """
    Generic exception class for inconsistent parameters.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent parameter choice"


class MisspelledParameter(Exception):
    """
    Generic exception class for inconsistent parameters.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        return "Parameter misspelled"


class MissingParameter(Exception):
    """
    Generic exception class for inconsistent parameters.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        return "Missing required parameter"


class GridParseError(Exception):
    """
    Generic exception class for issues parsing grid files.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        return "Gridf parse error"


class UnimplementedFunctionality(Exception):
    """
    Generic exception class for inconsistent parameters.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        return "Unimplemented functionality"


