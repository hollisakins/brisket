
Command-line Interface
======================

``brisket`` can be run from within python, like ``bagpipes``, or from the command line, by specifying the path to the data file and fitting parameters in a TOML configuration file. Specifically, ``brisket`` installs the following commands 


brisket-mod: generating model SEDs
----------------------------------

.. code:: sh
    
    brisket-mod -p param.toml

Used for generating model SEDs

brisket-fit: fitting models to data
-----------------------------------

.. code:: sh
    
    brisket-fit -p param.toml -o output/

Used for fitting a model to data

brisket-plot: plotting SEDs and results
---------------------------------------

.. code:: sh

    brisket-plot all -r output/run/...

Used for generating plots based on existing models or fits. 

brisket-filters: managing filter curves
---------------------------------------
