
Parameters Module
=================

The ``brisket.parameters`` module is responsible for handling the parameters used in ``brisket``. 
This module provides classes and methods to manage, validate, and manipulate parameters for different models such as galaxies and AGN. 

Classes
-------

- ``Params``: This is the main class for handling parameters. It allows adding sources, validating parameters, and provides a summary of fixed and free parameters.
- ``Group``: A class representing a parameter group, used for further sub-dividing the parameter specification. The ``Group`` class serves as a container for parameters belonging to a given source (e.g. galaxy, AGN) or absorber (e.g. dust) and can have its own sub-Groups.
- ``FreeParam``: A class representing a free parameter with specified limits and prior distributions.
- ``FixedParam``: A class representing a fixed parameter with a constant value. In practice, this is generally not used, as fixed parameters can be provided as integers or floats directly.

Usage
-----

You can initialize a ``Params`` object with a template (see: :doc:`templates`.) or as an emtpy object:

.. highlight:: python

::

    import brisket
    params = brisket.Params(template='default')


From there, you can then add sources and parameters as needed.

The parameter structure of BRISKET is broken up into sources (sources of emission), absorbers (things that absorb emission), and reprocessors (things that 
absorb emission and re-emit as sources of emission). 

Printing the parameters object will provide a nice representation of the parameter structure, including the fixed and free parameters:

::

    print(params)
    
    >>> ┏━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ Parameter name ┃ Value ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ redshift       │   9   │
        │ agn/beta       │ -2.5  │
        │ agn/Muv        │  -21  │
        │ agn/dust/Av    │   1   │
        │ igm/xhi        │  0.9  │
        └────────────────┴───────┘





Defaults and Aliases
--------------------

We include several aliases for adding sources/absorbers/reprocessors to the params object. For example, 

::

    params.add_igm()

is an alias for the slightly longer expression

::

    params.add_absorber('igm', model=briskest.models.InoueIGMModel)

You'll notice that the ``add_igm()`` methods presumes the default ``InoueIGMModel`` model, though this can be changed by passing a different model: ``params.add_igm(model=MadauIGMModel)``.
In any case, the ``add_absorber()`` method is iself an alias for the multi-step process of initializing a "parameter group" to describe the IGM model, noting that it is an "absorber," and adding it to the params object: 

::
    
    igm = brisket.parameters.Group('igm', model=briskest.models.InoueIGMModel, model_type='absorber')
    params['igm'] = igm

This is a bit more verbose, but allows for more flexibility in the parameter structure, and allows you to specify your own custom models. Say, for example, you wanted to include in your model a Damped Lyman-alpha system, you could define a custom DLA absorbption class and add it to the params object like so:

::
    
    class CustomDLAModel(brisket.models.BaseIGMModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def absorb(self, sed_incident):
            # custom absorption code here
            return sed_absorbed

    dla = brisket.parametrs.Group('dla', model=CustomDLAModel, model_type='absorber')
    params['dla'] = dla

More details are provided in the :doc:`custom_models` documentation.




.. Implemented by default: 

.. - Galaxy (Source)
..     - SFH (Group)
.. - AGN (Source)
.. - Nebular (Reprocessor)
.. - Dust (Reprocessor)
.. - IGM (Absorber)
.. - Calibration (Group)