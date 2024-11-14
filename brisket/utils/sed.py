import sys

class SED(object):
    
    def __init__(self, wav, flux=None, errors=None, wav_units=None, flux_units=None, light=False):
        if flux==None:
            self.logger.warning('No flux information provided, populating flam=fnu=0. If this is intended, you can ignore this message.')
            flux = np.zeros(len(self.wav))

        if not hasattr(wav, "unit"):
            raise Exception
        if not hasattr(flux, "unit"):
            raise Exception
        if isinstance(self.sed_units, str): self.sed_units = utils.unit_parser(self.sed_units)
        if isinstance(self.wav_units, str): self.wav_units = utils.unit_parser(self.wav_units)

        self.wav = wav
        self.flux = wav

        self.incident
        self.transmitted
        self.nebular
        self.reprocessed
        self.escaped
        self.intrinsic
        self.attenuated
        self.dust
        self.total

    def __add__(self, other):
        newobj = SED(wav)
        newobj.incident = self.incident + other.incident
        newobj.transmitted = self.transmitted + other.transmitted
        newobj.nebular = self.nebular + other.nebular
        newobj.reprocessed = self.reprocessed + other.reprocessed
        newobj.escaped = self.escaped + other.escaped
        newobj.intrinsic = self.intrinsic + other.intrinsic
        newobj.attenuated = self.attenuated + other.attenuated
        newobj.dust = self.dust + other.dust
        newobj.total = self.total + other.total
        return newobj

    def resample(self):
        pass

    def __str__(self):
        return f'{self.wav}, {self.flux}'

    def __repr__(self):
        wunit, funit = 'AA', 'ergscma'
        if len(self.wav) > 4:
            wstr = f'[{self.wav[0]:.2f}, {self.wav[1]:.2f}, ..., {self.wav[-2]:.2f}, {self.wav[-1]:.2f}] {wunit}'
            fstr = f'[{self.flux[0]:.1e}, {self.flux[1]:.1e}, ..., {self.flux[-2]:.1e}, {self.flux[-1]:.1e}] {funit}'

        return f'''sed: (wav: {wstr}, \n              fnu: {fstr})'''

    def to(self, unit, inplace=False):
        # if unit is wavelength or frquency, adjust x-units
        # if unit is flux or flux density, adjust y-units
        # if unit is tuple of (wavelength OR frequency, flux OR flux density), adjust both x and y-units
        pass

        
        # if 'spectral flux density' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Converting SED flux units to f_nu ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2 * (1 * self.wav_units)**2 / speed_of_light).to(self.sed_units).value
        # elif 'spectral flux density wav' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Keeping SED flux units in f_lam ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2).to(self.sed_units).value

    # def _flux_to_fnu()

    # things to compute automatically:
    # wavelength, frequency, energy
    # Lnu, Llam, Fnu, Flam
    # bolometric luminosity
    # sed.measure_window_luminosity((1400.0 * Angstrom, 1600.0 * Angstrom))
    # sed.measure_balmer_break()
    # sed.measure_d4000()
    # sed.measure_beta(windows='Calzetti94')







# import numpy as np
# wav = np.linspace(1, 1000, 100)
# flux = np.zeros(len(wav))
# s = sed(wav=wav, flux=flux)
# s


