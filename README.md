# Brisket

SED fitting package for astronomical observations.

## Overview

Brisket is a Python package for spectral energy distribution (SED) fitting, designed for analyzing astronomical photometry and spectroscopy data using Bayesian inference with NumPyro.

## Features

- Flexible model composition system (sequential and parallel)
- Support for photometric and spectroscopic observations
- Bayesian parameter estimation using MCMC
- Built-in filter database for common surveys
- Unit-aware parameter handling
- Comprehensive results analysis and diagnostics

## Installation

Install the required dependencies:

```bash
pip install astropy numpy matplotlib scipy tqdm spectres rich h5py toml numpyro
```

Then install brisket:

```bash
pip install -e .
```

## Quick Start

```python
import brisket

# Load observational data
photometry = brisket.Photometry(
    name="example",
    filters=["hst/acs/f435w", "hst/acs/f606w"],
    fnu=[1.2, 1.8],  # microJy
    fnu_err=[0.1, 0.2]
)

# Set up model with parameters
model = StellarModel(
    redshift=brisket.Uniform("redshift", 0.1, 3.0),
    mass=brisket.LogUniform("mass", 1e8, 1e12, unit="Msun"),
    age=brisket.Uniform("age", 0.1, 13.0, unit="Gyr")
)

# Run MCMC fitting
fitter = brisket.Fitter(model, photometry)
results = fitter.run()

# Analyze results
results.print_summary()
```

## License

MIT License