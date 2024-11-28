Model Grids
===========

Many models in BRISKET take advantage of pre-computed model grids. 

Due to the large file sizes, not all model grids are shipped with the code; however, BRISKET includes a utility (``GridManager``) to download grids from the web.
When initializing a model/fit using model grids, the ``GridManager`` will check if the requested grids are available locally at the configured directory (``brisket.config.grid_dir``) (by default, ``brisket/data``). If not, it will ask you if you'd like to download the grids from the web server. Grids are hosted on AWS at `s3://brisket-data<https://brisket-data.s3.amazonaws.com/index.html>`_ (note, you 
can also view and download grids from the browser if you want to download multiple ahead of time).

The following grids are currently available:

- ``bc03``: Bruzual & Charlot (2003) stellar population models

    - ``bc03_miles_chabrier.hdf5``: MILES stellar library, Chabrier IMF, native resolution (221 ages, 7 metallicities)

    - ``bc03_miles_chabrier_a50.hdf5``: downsampled to 50 ages

    - ``bc03_miles_kroupa.hdf5``: MILES stellar library, Kroupa IMF, native resolution (221 ages, 7 metallicities)

- ``bc03+cloudy``: Bruzual & Charlot (2003) stellar population models ran through Cloudy photoionization models

    - ``bc03+cloudy_miles_chabrier_a50_zfixed.hdf5``: MILES stellar library, Chabrier IMF, (50 ages, 7 metallicities)
    
- ``d_igm_grid_inoue14.fits``: IGM model grid from `Inoue et al. (2014)<https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I/abstract>`_ 
