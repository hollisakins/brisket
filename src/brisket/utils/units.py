# Not currently used
 
import astropy.units as u
def unit_parser(unit_str):
    d = {
        'angstrom': u.angstrom, 'AA': u.angstrom, 'A': u.angstrom, 
        'micron': u.um, 'um':u.um, 
        'nanometer': u.um, 'nm':u.nm, 
        'ergscma': u.erg/u.s/u.cm**2/u.angstrom,
        'erg/s/cm2/a': u.erg/u.s/u.cm**2/u.angstrom,
        'erg/s/cm2/aa': u.erg/u.s/u.cm**2/u.angstrom,
        'ergscm2a': u.erg/u.s/u.cm**2/u.angstrom,
        'nanojy': u.nJy, 'nanoJy': u.nJy, 'njy': u.nJy, 'nJy': u.nJy, 
        'mujy': u.uJy, 'muJy': u.uJy, 'uJy': u.uJy, 'ujy': u.uJy, 
        'mjy': u.mJy, 'mJy': u.mJy,
        'jy': u.Jy, 'Jy': u.Jy,
    }
    return d[unit_str]
