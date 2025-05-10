Defining Custom Models
======================

``brisket`` is designed to be simple and easy-to-use as shipped, but maximally expandable/customizable, for users with more complicated modeling needs. 

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


