import sys

class sed(object):
    
    def __init__(self, wav, flam=None, fnu=None, errors=None):
        self.wav = wav
        if flam==None and fnu==None:
            sys.exit()

    # @property
    # def flam(self):
    #     if self.


    def to_flam(self):

        return self


    def to_fnu(self):

        return self

    def __str__(self):
        return f'{self.wav}, {self.flux}'

    def __repr__(self):
        wunit, funit = 'AA', 'ergscma'
        if len(self.wav) > 4:
            wstr = f'[{self.wav[0]:.2f}, {self.wav[1]:.2f}, ..., {self.wav[-2]:.2f}, {self.wav[-1]:.2f}] {wunit}'
            fstr = f'[{self.flux[0]:.1e}, {self.flux[1]:.1e}, ..., {self.flux[-2]:.1e}, {self.wav[-1]:.1e}] {funit}'

        return f'''brisket-sed: (wav: {wstr}, \n              fnu: {fstr})'''


    def to(self, unit):
        # if unit is wavelength or frquency, adjust x-units
        # if unit is flux or flux density, adjust y-units
        pass

import numpy as np
wav = np.linspace(1, 1000, 100)
flux = np.zeros(len(wav))
s = sed(wav=wav, flux=flux)
s


