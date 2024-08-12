import brisket
import matplotlib.pyplot as plt

mc = {'redshift':3, 
      'galaxy':{
        'model':'BC03',
        'logMstar':10,
        'metallicity':0.5,
        'sfh':'constant',
        'age_min':0,
        'age_max':0.5,
        'dust_atten':{
            'type':'Calzetti',
            'Av':3,
        },
        }
    }

gal1 = brisket.model_galaxy(mc, logger=brisket.utils.basicLogger)
from copy import copy
mc['galaxy']['dust_emission'] = {'qpah':4.5, 'umin':20, 'gamma':0.5}
gal2 = brisket.model_galaxy(mc, logger=brisket.utils.basicLogger)


fig, ax = plt.subplots()

ax.loglog(gal1.wav_obs, gal1.spectrum_full)
ax.loglog(gal2.wav_obs, gal2.spectrum_full)

mc['galaxy']['dust_emission']['qpah'] = 0.5
gal2.update(mc)
ax.loglog(gal2.wav_obs, gal2.spectrum_full)

ax.set_xlim(0.3, 1e4)
ax.set_ylim(1e-5, 1e4)
plt.show()