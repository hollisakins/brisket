# BRISKET: Bagpipes, Repurposed for Ism, Stellar, and blacK holE fitting in Texas

A galaxy/quasar SED fitting code based on `bagpipes` by Adam Carnall.

`brisket` adopts the base code of `bagpipes` but expands it to provide additional modeling options, easier customizability/modularity, and more flexibility. Specifically: 

- new nested parameter structure to allow multi-component models, each component with its own specification of dust/nebular parameters
    - e.g. multi-component galaxy models with different ages *and* different dust attenuation, or composite galaxy+AGN models
- new AGN models 
- new nebular models, built for fitting emission line spectra

- ability to specify custom SFHs or dust attenuation laws as functions passed into the parameter file directly
- a simple least-squares optimization routine for quick tests and prior specification
- ability to specify BC03 or BPASS stellar templates directly, without any need to change the package config
- ability to specify Kroupa or Chabrier IMF directly

some QOL changes
- integration with astropy `Units`/`Quanity` schema 
- CLI/parameter file input and FITS file output for easy operation by non-Python users
- full logging capabilities to easily diagnose issues in your runs


## Command-line interface

`brisket` can be run from within python, like `bagpipes`, or from the command line, by specifying the path to the data file and fitting parameters in a TOML configuration file. Specifically, `brisket` installs the following commands 
```brisket-mod -p param.toml``` 
Used for generating model SEDs

```brisket-fit -p param.toml -o output/```
Used for fitting a model to data

```brisket-plot all -r output/run/...```
Used for generating plots based on existing models or fits. 