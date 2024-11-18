Parameters Module
=================

The `brisket.parameters` module is responsible for handling the parameters used in the `brisket` package. 
This module provides classes and methods to manage, validate, and manipulate parameters for different models such as galaxies and AGN. 

Classes
-------

- `Params`: This is the main class for handling parameters. It allows adding sources, validating parameters, and provides a summary of fixed and free parameters.
- `Source`: A class representing a source (e.g., galaxy, AGN). It acts as a container for parameters and can have sub-sources.
- `FreeParam`: A class representing a free parameter with specified limits and prior distributions.
- `FixedParam`: A class representing a fixed parameter with a constant value.

Usage
-----

To use the `Params` class, you can initialize it with a template or a file containing parameter definitions. You can then add sources and parameters as needed.

Example:



The parameter structure of BRISKET is broken up into sources (sources of emission), absorbers (things that absorb emission), and reprocessors (things that 
absorb emission and re-emit as sources of emission). 


We include several aliases for adding sources/absorbers/reprocessors to the params object. 
For example, 
```python
params.add_igm()
```
is an alias for 
```python
params.add_absorber('igm', model=briskest.models.InoueIGMModel)
```
which is iself an alias for the multi-step process of initializing an absorber and adding it to the params:
```python
igm = brisket.parameters.Absorber('igm', model=briskest.models.InoueIGMModel)
params['igm'] = igm
```



Implemented by default: 
- Galaxy (Source)
    - SFH (Group)
- AGN (Source)
- Nebular (Reprocessor)
- Dust (Reprocessor)
- IGM (Absorber)
- Calibration (Group)