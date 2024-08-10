BRISKET
========

**Bagpipes, Repurposed for Ism, Stellar, and blacK holE fitting in Texas**

 
A galaxy/quasar SED fitting code based on ``bagpipes`` by Adam Carnall.
``brisket`` adopts the base code of ``bagpipes`` but expands it to provide additional modeling options, easier customizability/modularity, and more flexibility. Specifically: 
 
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

.. Bagpipes is `developed at GitHub <https://github.com/ACCarnall/bagpipes>`_, however the code cannot be installed from there, as the large model grid files aren't included in the repository. The code should instead be installed with pip:

..     pip install bagpipes


.. All of the code's Python dependencies will be automatically installed.


Getting started
---------------

.. The best place to get started is by looking at the `iPython notebook examples <https://github.com/ACCarnall/bagpipes/tree/master/examples>`_. It's a good idea to tackle them in order as the later examples build on the earlier ones. These documentation pages contain a more complete reference guide.


Acknowledgements
----------------

``BRISKET`` is fundamentally an "expansion pack" for ``BAGPIPES``, which is itself a very popular SED fitting code. ``BRISKET`` therefore owes a huge debt to Adam Carnall for his work developing ``BAGPIPES``, as well as many of the project ``BAGPIPES`` relies on:

* The `Bruzual \& Charlot (2003) <https://arxiv.org/abs/astro-ph/0309134>`_ stellar population models.
* The `Draine \& Li (2007) <https://arxiv.org/abs/astro-ph/0608003>`_ dust emission models.
* The `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ nested sampling algorithm `(Feroz et al. 2013) <https://arxiv.org/abs/1306.2144>`_
* The `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ Python interface for Multinest `(Buchner et al. 2014) <https://arxiv.org/abs/1402.0004>`_.
* The `Cloudy <https://www.nublado.org>`_ photoionization code `(Ferland et al. 2017) <https://arxiv.org/abs/1705.10877>`_.
* The `nautilus <https://nautilus-sampler.readthedocs.io/en/stable/>`_ importance nested sampling algorithm `(Lange 2023) <https://arxiv.org/abs/2306.16923>`_.

As well as several projects incorporated into ``BRISKET``:

* Empirical QSO SED templates from ``qsogen`` `Temple et al. (2021) <https://arxiv.org/abs/2109.04472>`_ 
* TBD

 .. toctree::
    :maxdepth: 1
    :hidden:

    index.rst
    cli.rst
