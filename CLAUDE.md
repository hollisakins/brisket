# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Test basic package functionality
python -c "import src.brisket; print('Package imports correctly')"
```

### Testing
```bash
# Test specific components (examples)
python -c "from src.brisket.parameters import ParameterManager; print('ParameterManager works')"
python -c "from src.brisket.fitting.fitter import Fitter; print('Fitter works')"
python -c "from src.brisket.results import FitResults; print('FitResults works')"
```

## Code Architecture

### Core Design Philosophy
Brisket is a Bayesian SED fitting package built on JAX/NumPyro with a clear separation between expensive preprocessing (done once at model initialization) and fast evaluation (during MCMC sampling). The architecture emphasizes functional composition, unit safety, and performance.

### Parameter Management System
The most complex component is the parameter management system in `parameters.py`:

- **Parameter Class**: Wraps NumPyro distributions with units and transformations
- **ParameterManager**: Single source of truth for all parameters (both free and fixed)
- **Key Behavior**: Identical Parameter objects are shared across models; different objects are kept separate even with same priors
- **Namespacing**: Multiple instances of same model type get unique namespaces (`StellarModel`, `StellarModel_1`, etc.)
- **Parameter Naming**: NumPyro sampling uses the Parameter's name (e.g., `"mass"`), while internal model mapping uses keyword argument names (e.g., `"log_stellar_mass"`)
- **Unified Storage**: Both free parameters (Parameter objects) and fixed parameters (values) are stored in ParameterManager
- **Model Interface**: `model.parameters` property returns all parameters; `model.parameter_manager` handles all parameter operations

### Model Composition Patterns
Models support two composition types:
- **Sequential** (`dust_model(stellar_model)`): Output of first becomes input to second
- **Parallel** (`stellar_model + emission_model`): Both process same input, outputs summed

All models inherit from `Model` base class with this pattern:
- `__init__()`: Expensive preprocessing, parameter registration
- `_validate()`: Parameter validation  
- `_preprocess()`: Load/interpolate data grids
- `_evaluate()`: Fast evaluation during sampling
- `evaluate()`: Public interface

### Observation System
- **Unified Interface**: `Photometry` and `Spectrum` classes share common `Observation` base
- **Combination**: Use `+` operator to combine different observation types
- **Wavelength Optimization**: Automatically determines optimal model wavelength sampling based on data resolution requirements

### Unit Management
- **Internal Standards**: Wavelengths in Angstroms, SEDs in L_sun/Å
- **Automatic Conversion**: All user inputs converted to internal units
- **Type Safety**: Unit validation prevents common astronomical unit errors

### Import Structure
Key imports for development:
```python
# Core classes
from src.brisket import Photometry, Spectrum, Fitter
from src.brisket.parameters import Parameter, Uniform, Normal, LogUniform
from src.brisket.models.base import Model, CompositeModel
from src.brisket.results import FitResults

# For extending models
from src.brisket.models.stellar import StellarModel  # Example implementation
from src.brisket.utils.units import sanitize_wavelength_array, SEDUnits
```

### Configuration
Package configuration in `config.py` loads from `config.toml` with environment variable overrides:
- Filter and grid directories
- Wavelength ranges and resolution settings  
- Cosmology parameters
- Missing config values should be added to both files

### Critical Dependencies
- **JAX/NumPyro**: All arrays must be JAX-compatible for autodiff and sampling
- **Astropy**: Units, cosmology, and astronomical calculations
- **Rich**: Error formatting and progress bars
- **spectres**: Spectral resampling utilities

### File Organization Patterns
- `__init__.py` files in subdirectories export main classes
- `base.py` files contain abstract base classes and core functionality
- Model implementations in separate files (e.g., `stellar.py`, `spectrum.py`)
- Utils organized by functionality (units, filters, misc, resample)

### Key Architectural Constraints
- All model evaluation must be JAX-compatible (no Python loops in hot paths)
- Parameters must be registered with ParameterManager for MCMC sampling
- Wavelength grids must be consistent across all model components
- Unit conversions should happen at boundaries, not in inner loops
- Expensive operations (file I/O, interpolation setup) must be in `__init__` or `_preprocess()`