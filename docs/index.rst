BRISKET
========

BRISKET (loosely, **Baeysian Retrieval and Inference for Stellar and blacK holeE fitting in Texas**) is a full-featured SED fitting code for fitting galaxy and AGN SEDs. 


.. note::
    
    This project is in very early stages of development.

 
* new nested parameter structure to allow multi-component models, each component with its own specification of dust/nebular parameters

  * e.g. multi-component galaxy models with different ages *and* different dust attenuation, or composite galaxy+AGN models
* new AGN models 
* new nebular models, built for fitting emission line spectra
* ability to specify custom SFHs or dust attenuation laws as functions passed into the parameter file directly
* a simple least-squares optimization routine for quick tests and prior specification
* ability to specify BC03 or BPASS stellar templates directly, without any need to change the package config

As well as some QOL changes

* CLI/parameter file input and FITS file output for easy operation by non-Python users
* full logging capabilities to easily diagnose issues in your runs
* integration with astropy ``Units``/``Quanity`` schema 


Source and installation
-----------------------

For now, the code should be installed by cloning this repository and installing locally with ``pip``:

```bash
git clone https://github.com/hollisakins/brisket.git
cd brisket
pip install -e .
```

Note that the ``-e`` flag installs the package in "editable" mode, so that any changes to the files in your install directory will be reflected in the installed package. 
This is useful while the code is in active development, but may not be necessary for normal use.


Getting started
---------------

TBD

Acknowledgements
----------------

``brisket`` is heavily inspired by other existing SED fitting codes, including ``bagpipes`` by Adam Carnall and ``synthesizer`` by the FLARES simulation team.  
The goal of ``brisket`` is to provide a similarly user-friendly interface as ``bagpipes`` but with additional modeling options and a more modular and flexible codebase. 

* The `Bruzual \& Charlot (2003) <https://arxiv.org/abs/astro-ph/0309134>`_ stellar population models.
* The `Draine \& Li (2007) <https://arxiv.org/abs/astro-ph/0608003>`_ dust emission models.
* The `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ nested sampling algorithm `(Feroz et al. 2013) <https://arxiv.org/abs/1306.2144>`_
* The `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ Python interface for Multinest `(Buchner et al. 2014) <https://arxiv.org/abs/1402.0004>`_.
* The `Cloudy <https://www.nublado.org>`_ photoionization code `(Ferland et al. 2017) <https://arxiv.org/abs/1705.10877>`_.
* The `nautilus <https://nautilus-sampler.readthedocs.io/en/stable/>`_ importance nested sampling algorithm `(Lange 2023) <https://arxiv.org/abs/2306.16923>`_.

* Empirical QSO SED templates from ``qsogen`` `Temple et al. (2021) <https://arxiv.org/abs/2109.04472>`_ 
* The `UltraNest <https://johannesbuchner.github.io/UltraNest/index.html>`_ nested sampling algorithm `(Buchner et al. 2021) <https://arxiv.org/abs/2101.09604>`_
* TBD

.. toctree::
  :maxdepth: 1
  :caption: Contents: 
  
  Home <self>
  install
  features
  models
  cli
  api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   example-model-sed.ipynb
   example-simple-fit.ipynb