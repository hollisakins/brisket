from ..config import cosmo
import numpy as np
from astropy.units import cm

z_array = np.arange(0., 100., 0.01)
age_array = cosmo.age(z_array).value

def age_at_z(z):
    '''Returns the age of the universe, in Gyr, at redshift z'''
    return cosmo.age(z).value

def z_at_age(age):
    '''Returns the redshfit corresponding to a given age of the universe, in Gyr'''
    return np.interp(age, np.flip(age_array), np.flip(z_array))

def fourPiLumDistSq(redshift):
    '''Returns 4*pi*d_L^^2 in sq. cm at the given redshift'''
    dL = cosmo.luminosity_distance(redshift).to(u.cm).value
    return 4*np.pi*dL**2

def lum_to_flux(redshift):
    '''Converts from luminosity density (Lsun/A) to flux density (erg/s/cm2/A)'''
    return 3.825e33/(fourPiLumDistSq(redshift)*(1+redshift))
