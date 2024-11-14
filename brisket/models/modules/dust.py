

class DustModel:

    def __init__(self):
        # choose which attenuation curve, etc



        # Pre-compute dust curve shape if fixed for the chosen model.
        if self.type == "Calzetti":
            self.A_cont = self._calzetti(wavelengths)
            self.A_line = self._calzetti(config.line_wavs)

        elif self.type == "Cardelli":
            self.A_cont = self._cardelli(wavelengths)
            self.A_line = self._cardelli(config.line_wavs)

        elif self.type == "SMC":
            self.A_cont = self._smc_gordon(wavelengths)
            self.A_line = self._smc_gordon(config.line_wavs)

        elif self.type == "QSO": # QSO reddening curve from Temple et al. 2021
            wavtmp, flxtmp = np.genfromtxt(config.qsogen_ext_curve, unpack=True)
            R = 3.1
            self.A_cont = (np.interp(wavelengths, wavtmp, flxtmp) + R)/R
            self.A_line = (np.interp(config.line_wavs, wavtmp, flxtmp) + R)/R

        # If Salim dust is selected, pre-compute Calzetti to start from.
        elif self.type == "Salim":
            self.A_cont_calz = self._calzetti(wavelengths)
            self.A_line_calz = self._calzetti(config.line_wavs)



        pass

    def update(self, param):

        # Fixed-shape dust laws are pre-computed in __init__.
        if self.type in ["Calzetti", "Cardelli", "SMC", "QSO"]:
            return

        # Variable shape dust laws have to be computed every time.
        self.A_cont, self.A_line = getattr(self, self.type)(param)
        self.trans_cont, self.trans_line, self.trans_bc = self.transmission()


    def absorb(incident_flux, param):
        Av = self.param['dust_atten']['Av']
        logfscat = self.param['dust_atten']['logfscat']
        fscat = np.power(10., logfscat)
        trans_cont = 10**(-Av*self.A_cont/2.5) + fscat
        return incident_flux * trans, total_energy_absorbed

    def emit(total_energy_absorbed):
        pass




