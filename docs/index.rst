BRISKET
========

BRISKET (loosely, **Baeysian Retrieval and Inference for Stellar and blacK holeE fitting in Texas**) is a full-featured SED fitting code for fitting galaxy and AGN SEDs. 


.. note::
    
    This project is in very early stages of development.

Key features of BRISKET include: 

* Nested, generalizable parameter structure to allow multi-component models, custom SFHs, and more
* Easily interchangable stellar model grids
* AGN models including both empirical templates and theoretical SEDs passed through photoionization models
* Flexible nebular models, built for fitting emission line spectra
* Modularity, allowing custom model prescriptions 
.. * a simple least-squares optimization routine for quick tests and prior specification

As well as some QOL changes

* Command-line/parameter file interface and FITS file output for easy operation by non-Python users
* Integration with astropy ``units`` package
* Built-in plotting routines for quick visualization of fits


Source and installation
-----------------------

For now, the code should be installed by cloning this repository and installing locally with ``pip``:

::

  git clone https://github.com/hollisakins/brisket.git
  cd brisket
  pip install -e .


Note that the ``-e`` flag installs the package in "editable" mode, so that any changes to the files in your install directory will be reflected in the installed package. 
This is useful while the code is in active development, but may not be necessary for normal use.


Contents
--------

.. toctree::
  :maxdepth: 2
  
  Home <self>
  1_getting_started/getting_started
  2_parameters/parameters
  3_models/models
  api
  




Acknowledgements
----------------

``brisket`` is heavily inspired by other existing SED fitting codes, including ``bagpipes`` by Adam Carnall and ``synthesizer`` by the FLARES simulation team.  
The goal of ``brisket`` is to provide a similarly user-friendly interface as ``bagpipes`` but with additional modeling options and a more modular and flexible codebase. 

Many packages are used under the hood to make ``brisket`` work, including:

* The `Bruzual \& Charlot (2003) <https://arxiv.org/abs/astro-ph/0309134>`_ stellar population models.
* The `Draine \& Li (2007) <https://arxiv.org/abs/astro-ph/0608003>`_ dust emission models.
* The `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ nested sampling algorithm `(Feroz et al. 2013) <https://arxiv.org/abs/1306.2144>`_
* The `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ Python interface for Multinest `(Buchner et al. 2014) <https://arxiv.org/abs/1402.0004>`_.
* The `Cloudy <https://www.nublado.org>`_ photoionization code `(Ferland et al. 2017) <https://arxiv.org/abs/1705.10877>`_.
* The `nautilus <https://nautilus-sampler.readthedocs.io/en/stable/>`_ importance nested sampling algorithm `(Lange 2023) <https://arxiv.org/abs/2306.16923>`_.
* Empirical QSO SED templates from ``qsogen`` `Temple et al. (2021) <https://arxiv.org/abs/2109.04472>`_ 
* The `UltraNest <https://johannesbuchner.github.io/UltraNest/index.html>`_ nested sampling algorithm `(Buchner et al. 2021) <https://arxiv.org/abs/2101.09604>`_
* More TBD
