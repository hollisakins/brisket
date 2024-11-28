
Overview
========

BRISKET is designed to be a flexible, modular, and user-friendly SED fitting code for fitting galaxy and AGN SEDs. 

What problems is BRISKET designed to solve?
-------------------------------------------

Many galaxy SED fitting codes exist, such as bagpipes, prospector, BEAGLE, and CIGALE. 
However, each code has its own limitations, and can be difficult to modify or extend. 
BRISKET includes several modules not present in existing SED fitting codes, including 
empirial QSO templates, theoretical AGN SEDs passed through photoionization models, and
Moreover, its pure-python implementation and flexible architecture are designed to make it 
easy for users to add their own modules, to suit their specific science goals.


The model/parameter structure
-----------------------------

In the interest of flexibility, BRISKET as..
The core routines (`ModelGalaxy`, `Fitter`) are designed to be generalizable to any model structure. 

BRISKET recognizes four general model "types": ``source``, ``reprocessor``, ``absorber``, ``calibrator``. 
Some more specific models, such as ``sfh`` are implemented separetely. 

