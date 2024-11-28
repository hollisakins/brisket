Defining Custom Models
======================

``brisket`` is designed to be simple and easy-to-use as shipped, but maximally expandable/customizable, for users with more complicating modeling needs. 

This is done by allowing users to define their own custom models, which can be added to the parameter structure in the same way as the built-in models.

For a simple example, say you wanted to include in your model a Damped Lyman-alpha system. 
You could define a custom DLA absorbption class and add it to the params object like so:

::
    
    class CustomDLAModel(brisket.models.BaseIGMModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def absorb(self, sed_incident):
            # custom absorption code here
            return sed_absorbed

    dla = brisket.parametrs.Group('dla', model=CustomDLAModel, model_type='absorber')
    params['dla'] = dla

This will then use your custom absorption code in the fitting process.



Best practices
--------------

When defining custom models, it is best to inherit from the base model class that most 
closely matches the behavior of your custom model.
``brisket`` provides several base classes for different types of models, including:

- ``BaseStellarModel``
- ``BaseSFHModel``
- ``BaseAGNModel``
- ``BaseIGMModel``

Moreover, most models classes are expected to have the following methods: 

- ``_resample(self, wavelengths)``: resample the model to the given wavelengths. For grid-based models, this is 
  necessary in order for the models to be incorporated into any arbitrary SED. This resampling is easily done with 
  the built-in ``brisket.utils.SED`` class, which includes resampling via ``spectres``. For functional models, 
  such as the DLA model above, the ``_resample`` method needs only to define ``self.wavelengths = wavelengths``. 

- ``_build_defaults(self, params)``: checks the provided ``Params`` object and adds any default parameters 
  that need to be specified. A good example of this with certain SFH models, such as the ``ContinuitySFHModel``, 
  which requires repetatively assigning the same t-distribution prior to multiple parameters, which is done 
  in the ``_build_defaults`` method so the user doesn't have to.


