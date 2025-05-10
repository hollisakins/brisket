
Quickstart
==========

.. Generating a model SED

.. .. highlight:: python

.. :: 
    
..     import brisket
    
..     params = brisket.Params()
..     params['redshift'] = 10

..     params.add_source('agn')
..     params['agn']['beta'] = -2.5
..     params['agn']['Muv'] = -22

..     params['agn'].add_dust(model=NullModel)
..     params['agn']['dust']['Av'] = 2

..     params.add_igm()

..     mod = brisket.ModelGalaxy(params)
..     mod.sed.plot(show=True, xlim=(500, 8000), ylim=(-0.1, 0.5))


.. Fitting a model (not yet implemented)

.. :: 

..     import brisket
..     obs = brisket.Observation(filters=filters, phot=flux, phot_err=flux_err)
..     >>> --- loaded data ---

..     fitter = brisket.Fitter(params, obs)
..     >>> --- initializing fit ---

..     result = fitter.fit(n_live=400)
..     >>> --- running fit ---

..     print(result)



