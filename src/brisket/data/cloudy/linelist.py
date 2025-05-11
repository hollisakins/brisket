'''
Stores the list of emission lines known to brisket.
'''
import numpy as np
from dataclasses import dataclass

def air_to_vac(wav_air):
    # from SDSS: 
    # AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    vac_wavs = np.logspace(2, 9, 10000)
    air_wavs = vac_wavs / (1.0 + 2.735182E-4 + 131.4182 / vac_wavs**2 + 2.76249E8 / vac_wavs**4)
    return np.interp(wav_air, air_wavs, vac_wavs)

@dataclass
class Line:
    name: str
    label: str
    wav: float
    cloudy_label: str

@dataclass
class LineList:
    lines: list[Line]

    @property
    def wavs(self):
        return np.array([line.wav for line in self.lines])

    @property
    def labels(self):
        return [line.label for line in self.lines]

    @property
    def names(self):
        return np.array([line.name for line in self.lines])

    @property
    def cloudy_labels(self):
        return [line.cloudy_label for line in self.lines]
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, name):
        for line in self.lines:
            if line.name == name:
                return line
        return None

# vacuum wavelengths compiled from NIST or SDSS data
lines = [
    # lyman series ###########################################################################################################################
    Line(name='Lya', label=r'${\rm Ly}\alpha$', wav=1215.670, cloudy_label='H  1  1215.67A'),
    Line(name='Lyb', label=r'${\rm Ly}\beta$', wav=1025.720, cloudy_label='H  1  1025.72A'),
    Line(name='Lyg', label=r'${\rm Ly}\gamma$', wav=972.537, cloudy_label='H  1  972.537A'),
    Line(name='Lyd', label=r'${\rm Ly}\delta$', wav=949.743, cloudy_label='H  1  949.743A'),
    Line(name='Lye', label=r'${\rm Ly}\epsilon$', wav=937.804, cloudy_label='H  1  937.804A'),
    Line(name='Ly7', label=r'${\rm Ly}7$', wav=930.748, cloudy_label='H  1  930.748A'),
    Line(name='Ly8', label=r'${\rm Ly}8$', wav=926.226, cloudy_label='H  1  926.226A'),
    Line(name='Ly9', label=r'${\rm Ly}9$', wav=923.150, cloudy_label='H  1  923.150A'),
    # balmer series ##########################################################################################################################
    Line(name='Ha', label=r'${\rm H}\alpha$', wav=air_to_vac(6562.819), cloudy_label='H  1  6562.81A'),
    Line(name='Hb', label=r'${\rm H}\beta$', wav=air_to_vac(4861.333), cloudy_label='H  1  4861.33A'),
    Line(name='Hg', label=r'${\rm H}\gamma$', wav=air_to_vac(4340.471), cloudy_label='H  1  4340.46A'),
    Line(name='Hd', label=r'${\rm H}\delta$', wav=air_to_vac(4101.742), cloudy_label='H  1  4101.73A'),
    Line(name='He', label=r'${\rm H}\epsilon$', wav=air_to_vac(3970.079), cloudy_label='H  1  3970.07A'),
    Line(name='H8', label=r'${\rm H}8$', wav=air_to_vac(3889.050), cloudy_label='H  1  3889.05A'),
    Line(name='H9', label=r'${\rm H}9$', wav=air_to_vac(3835.380), cloudy_label='H  1  3835.38A'),
    Line(name='H10', label=r'${\rm H}10$', wav=air_to_vac(3797.890), cloudy_label='H  1  3797.89A'),
    Line(name='H11', label=r'${\rm H}11$', wav=air_to_vac(3770.637), cloudy_label='H  1  3770.63A'),
    Line(name='H12', label=r'${\rm H}12$', wav=air_to_vac(3750.158), cloudy_label='H  1  3750.15A'),
    Line(name='H13', label=r'${\rm H}13$', wav=air_to_vac(3734.369), cloudy_label='H  1  3734.37A'),
    Line(name='H14', label=r'${\rm H}14$', wav=air_to_vac(3721.945), cloudy_label='H  1  3721.94A'),
    Line(name='H15', label=r'${\rm H}15$', wav=air_to_vac(3711.977), cloudy_label='H  1  3711.97A'),
    Line(name='H16', label=r'${\rm H}16$', wav=air_to_vac(3703.859), cloudy_label='H  1  3703.85A'),
    Line(name='H17', label=r'${\rm H}17$', wav=air_to_vac(3697.157), cloudy_label='H  1  3697.15A'),
    Line(name='H18', label=r'${\rm H}18$', wav=air_to_vac(3691.551), cloudy_label='H  1  3691.55A'),
    Line(name='H19', label=r'${\rm H}19$', wav=air_to_vac(3686.831), cloudy_label='H  1  3686.83A'),
    # paschen series #########################################################################################################################
    Line(name='Paa', label=r'${\rm Pa}\alpha$', wav=18756.1, cloudy_label='H  1  1.87510m'),
    Line(name='Pab', label=r'${\rm Pa}\beta$', wav=12821.6, cloudy_label='H  1  1.28181m'),
    Line(name='Pag', label=r'${\rm Pa}\gamma$', wav=air_to_vac(10938.086), cloudy_label='H  1  1.09381m'),
    Line(name='Pad', label=r'${\rm Pa}\delta$', wav=air_to_vac(10049.368), cloudy_label='H  1  1.00494m'),
    Line(name='Pae', label=r'${\rm Pa}\epsilon$', wav=air_to_vac(9545.969), cloudy_label='H  1  9545.97A'),
    Line(name='Pa9', label=r'${\rm Pa}9$', wav=air_to_vac(9229.014), cloudy_label='H  1  9229.02A'),
    Line(name='Pa10', label=r'${\rm Pa}10$', wav=air_to_vac(9014.909), cloudy_label='H  1  9014.91A'),
    Line(name='Pa11', label=r'${\rm Pa}11$', wav=air_to_vac(8862.782), cloudy_label='H  1  8862.79A'),
    Line(name='Pa12', label=r'${\rm Pa}12$', wav=air_to_vac(8750.472), cloudy_label='H  1  8750.48A'),
    Line(name='Pa13', label=r'${\rm Pa}13$', wav=air_to_vac(8665.019), cloudy_label='H  1  8665.02A'),
    Line(name='Pa14', label=r'${\rm Pa}14$', wav=air_to_vac(8598.392), cloudy_label='H  1  8598.40A'),
    Line(name='Pa15', label=r'${\rm Pa}15$', wav=air_to_vac(8545.383), cloudy_label='H  1  8545.39A'),
    Line(name='Pa16', label=r'${\rm Pa}16$', wav=air_to_vac(8502.483), cloudy_label='H  1  8502.49A'),
    Line(name='Pa17', label=r'${\rm Pa}17$', wav=air_to_vac(8467.254), cloudy_label='H  1  8467.26A'),
    Line(name='Pa18', label=r'${\rm Pa}18$', wav=air_to_vac(8437.956), cloudy_label='H  1  8437.96A'),
    Line(name='Pa19', label=r'${\rm Pa}19$', wav=air_to_vac(8413.318), cloudy_label='H  1  8413.32A'),
    Line(name='Pa20', label=r'${\rm Pa}20$', wav=air_to_vac(8392.397), cloudy_label='H  1  8392.40A'),
    # brackett series ########################################################################################################################
    Line(name='Bra', label=r'${\rm Br}\alpha$', wav=air_to_vac(40511.30), cloudy_label='H  1  4.05115m'),
    Line(name='Brb', label=r'${\rm Br}\alpha$', wav=air_to_vac(26251.29), cloudy_label='H  1  2.62515m'),
    Line(name='Brg', label=r'${\rm Br}\alpha$', wav=air_to_vac(21655.09), cloudy_label='H  1  2.16553m'),
    Line(name='Brd', label=r'${\rm Br}\alpha$', wav=air_to_vac(19445.40), cloudy_label='H  1  1.94456m'),
    Line(name='Bre', label=r'${\rm Br}\alpha$', wav=air_to_vac(18174.00), cloudy_label='H  1  1.81741m'),
    Line(name='Br10', label=r'${\rm Br}\alpha$', wav=air_to_vac(17362.00), cloudy_label='H  1  1.73621m'),
    # pfund series ###########################################################################################################################
    Line(name='Pf6', label=r'${\rm Pf}6$', wav=air_to_vac(74577.699), cloudy_label='H  1  7.45777m'),
    Line(name='Pf7', label=r'${\rm Pf}7$', wav=air_to_vac(46524.699), cloudy_label='H  1  4.65247m'),
    Line(name='Pf8', label=r'${\rm Pf}8$', wav=air_to_vac(37395.099), cloudy_label='H  1  3.73951m'),
    Line(name='Pf9', label=r'${\rm Pf}9$', wav=air_to_vac(32960.699), cloudy_label='H  1  3.29607m'),
    Line(name='Pf10', label=r'${\rm Pf}10$', wav=air_to_vac(30383.500), cloudy_label='H  1  3.03835m'),
    # humphreys series #######################################################################################################################
    Line(name='Hu7', label=r'${\rm Hu}7$', wav=air_to_vac(123684.000), cloudy_label='H  1  12.3684m'),
    Line(name='Hu8', label=r'${\rm Hu}8$', wav=air_to_vac(75003.800), cloudy_label='H  1  7.50038m'),
    Line(name='Hu9', label=r'${\rm Hu}9$', wav=air_to_vac(59065.500), cloudy_label='H  1  5.90655m'),
    Line(name='Hu10', label=r'${\rm Hu}10$', wav=air_to_vac(51272.199), cloudy_label='H  1  5.12722m'),
    # helium lines ###########################################################################################################################
    Line(name='HeI10830', label=r'${\rm He}\,I$', wav=air_to_vac(10830.340), cloudy_label='Blnd  1.08302m'),
    Line(name='HeI7065', label=r'${\rm He}\,I$', wav=air_to_vac(7065.196), cloudy_label='Blnd  7065.25A'),
    Line(name='HeI6678', label=r'${\rm He}\,I$', wav=air_to_vac(6678.15), cloudy_label='He 1  6678.15A'),
    Line(name='HeI5876', label=r'${\rm He}\,I$', wav=air_to_vac(5875.624), cloudy_label='Blnd  5875.66A'),
    Line(name='HeI4471', label=r'${\rm He}\,I$', wav=air_to_vac(4471.479), cloudy_label='Blnd  4471.50A'),
    Line(name='HeI3889', label=r'${\rm He}\,I$', wav=air_to_vac(3888.647), cloudy_label='He 1  3888.64A'),
    Line(name='HeI3188', label=r'${\rm He}\,I$', wav=air_to_vac(3187.745), cloudy_label='He 1  3187.74A'),
    Line(name='HeII4685', label=r'He\,II\,$\lambda 4685$', wav=air_to_vac(4685.710), cloudy_label='He 2  4685.68A'),
    Line(name='HeII3203', label=r'He\,II\,$\lambda 3203$', wav=air_to_vac(3203.100), cloudy_label='He 2  3203.08A'),
    Line(name='HeII2733', label=r'He\,II\,$\lambda 2733$', wav=air_to_vac(2733.289), cloudy_label='He 2  2733.28A'),
    Line(name='HeII1640', label=r'He\,II\,$\lambda 1640$', wav=1640.400, cloudy_label='He 2  1640.41A'),
    # carbon lines ###########################################################################################################################
    Line(name='CI9850', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 9850$', wav=air_to_vac(9850.260), cloudy_label='Blnd  9850.00A'),
    Line(name='CI8727', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 8727$', wav=air_to_vac(8727.130), cloudy_label='C  1  8727.13A'),
    Line(name='CI4621', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 4621$', wav=air_to_vac(4621.570), cloudy_label='C  1  4621.57A'),
    Line(name='CI10', label=r'$[{\rm C}\,\textsc{i}](1-0)$', wav=6095900., cloudy_label='C  1  609.590m'),
    Line(name='CI21', label=r'$[{\rm C}\,\textsc{i}](2-1)$', wav=3702690., cloudy_label='C  1  370.269m'),
    Line(name='CII158', label=r'$[{\rm C}\,\textsc{ii}]\,158\,\mu$m', wav=1576360., cloudy_label='C  2  157.636m'),
    Line(name='CII2326', label=r'${\rm C}\,\textsc{ii}]\,\lambda 2326', wav=2326., cloudy_label='Blnd  2326.00A'),
    Line(name='CII1335', label=r'${\rm C}\,\textsc{ii}\,\lambda 1335', wav=1335.708, cloudy_label='Blnd  1335.00A'),
    Line(name='CIII1907', label=r'${\rm C}\,\textsc{iii}]\,\lambda 1907$', wav=1906.624, cloudy_label='C  3  1908.73A'),
    Line(name='CIII1909', label=r'${\rm C}\,\textsc{iii}]\,\lambda 1909$', wav=1908.791, cloudy_label='C  3  1906.68A'),
    Line(name='CIV1548', label=r'${\rm C}\,\textsc{iv}]\,\lambda 1548$', wav=1548.187, cloudy_label='C  4  1548.19A'),
    Line(name='CIV1551', label=r'${\rm C}\,\textsc{iv}]\,\lambda 1551$', wav=1550.772, cloudy_label='C  4  1550.77A'),
    # nitrogen lines #########################################################################################################################
    Line(name='NI5200', label=r'$[{\rm N}\,\textsc{i}]\,\lambda 5200$', wav=air_to_vac(5200.257), cloudy_label='N  1  5200.26A'),
    Line(name='NII6583', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 6583$', wav=air_to_vac(6583.460), cloudy_label='N  2  6583.45A'),
    Line(name='NII6548', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 6548$', wav=air_to_vac(6548.050), cloudy_label='N  2  6548.05A'),
    Line(name='NII5754', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 5754$', wav=air_to_vac(5754.590), cloudy_label='N  2  5754.61A'),
    Line(name='NII2139', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 2139$', wav=air_to_vac(2139.010), cloudy_label='N  2  2139.01A'),
    Line(name='NII122', label=r'$[{\rm N}\,\textsc{ii}]\,122\,\mu$m', wav=1217670., cloudy_label='N  2  121.767m'),
    Line(name='NII205', label=r'$[{\rm N}\,\textsc{ii}]\,205\,\mu$m', wav=2052440., cloudy_label='N  2  205.244m'),
    Line(name='NIII57', label=r'$[{\rm N}\,\textsc{iii}]\,57\,\mu$m', wav=573238., cloudy_label='N  3  57.3238m'),
    # oxygen lines ###########################################################################################################################
    Line(name='OI8446', label=r'${\rm O}\,\textsc{i}\,\lambda 8446$', wav=air_to_vac(8446.359), cloudy_label='Blnd  8446.00A'),
    Line(name='OI7254', label=r'${\rm O}\,\textsc{i}\,\lambda 7254$', wav=air_to_vac(7254.448), cloudy_label='Blnd  8446.00A'),
    Line(name='OI6364', label=r'$[{\rm O}\,\textsc{i}]\,\lambda 6364$', wav=air_to_vac(6363.776), cloudy_label='O  1  6363.78A'),
    Line(name='OI6300', label=r'$[{\rm O}\,\textsc{i}]\,\lambda 6300$', wav=air_to_vac(6300.304), cloudy_label='O  1  6300.30A'),
    Line(name='OI63', label=r'$[{\rm O}\,\textsc{i}]\,63\,\mu$m', wav=631679., cloudy_label='O  1  63.1679m'),
    Line(name='OI145', label=r'$[{\rm O}\,\textsc{i}]\,145\,\mu$m', wav=1454950., cloudy_label='O  1  145.495m'),
    Line(name='OII7331', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 7331$', wav=air_to_vac(7330.730), cloudy_label='Blnd  7332.00A'),
    Line(name='OII7320', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 7320', wav=air_to_vac(7319.990), cloudy_label='Blnd  7323.00A'),
    Line(name='OII3729', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 3729', wav=air_to_vac(3728.815), cloudy_label='Blnd  3729.00A'),
    Line(name='OII3726', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 3726', wav=air_to_vac(3726.032), cloudy_label='Blnd  3726.00A'),
    Line(name='OII2471', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 2471', wav=2471.00, cloudy_label='Blnd  2471.00A'),
    Line(name='OIII5007', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 5007', wav=air_to_vac(5006.843), cloudy_label='O  3  5006.84A'),
    Line(name='OIII4959', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 4959', wav=air_to_vac(4958.911), cloudy_label='O  3  4958.91A'),
    Line(name='OIII4363', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 4363', wav=air_to_vac(4363.210), cloudy_label='Blnd  4363.00A'),
    Line(name='OIII2321', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 2321', wav=air_to_vac(2320.951), cloudy_label='O  3  2320.95A'),
    Line(name='OIII1666', label=r'${\rm O}\,\textsc{iii}]\,\lambda 1666', wav=air_to_vac(1666.150), cloudy_label='O  3  1666.15A'),
    Line(name='OIII1661', label=r'${\rm O}\,\textsc{iii}]\,\lambda 1661', wav=air_to_vac(1660.809), cloudy_label='O  3  1660.81A'),
    Line(name='OIII88', label=r'$[{\rm O}\,\textsc{iii}]\,88\,\mu$m', wav=883323., cloudy_label='O  3  88.3323m'),
    Line(name='OIII52', label=r'$[{\rm O}\,\textsc{iii}]\,52\,\mu$m', wav=518004., cloudy_label='O  3  51.8004m'),
    # neon lines #############################################################################################################################
    Line(name='NeII12um', label='', wav=128101.0, cloudy_label='Ne 2  12.8101m'),
    Line(name='NeIII15um', label='', wav=155509.0, cloudy_label='Ne 3  15.5509m'),
    Line(name='NeIII36um', label='', wav=360036.0, cloudy_label='Ne 3  36.0036m'),
    Line(name='NeIII3967', label=r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3967$', wav=air_to_vac(3967.470), cloudy_label='Ne 3  3967.47A'),
    Line(name='NeIII3869', label=r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3869$', wav=air_to_vac(3868.760), cloudy_label='Ne 3  3868.76A'),
    Line(name='NeIII3342', label='', wav=air_to_vac(3342.180), cloudy_label='Ne 3  3342.18A'),
    Line(name='NeIII1814', label='', wav=1814.560, cloudy_label='Ne 3  1814.56A'),
    Line(name='NeIV2423', label=r'$[{\rm Ne}\,\textsc{iv}]\,\lambda 2423$', wav=air_to_vac(2421.66), cloudy_label='Ne 4  2421.66A'),
    Line(name='NeV3427', label=r'$[{\rm Ne}\,\textsc{v}]\,\lambda 3427$', wav=air_to_vac(3425.881), cloudy_label='Ne 5  3425.88A'),
    Line(name='MgII2796', label=r'$[{\rm Mg}\,\textsc{ii}]\,\lambda\lambda 2796$', wav=air_to_vac(2795.528), cloudy_label='Mg 2  2795.53A'),
    Line(name='MgII2803', label=r'$[{\rm Mg}\,\textsc{ii}]\,\lambda\lambda 2803$', wav=air_to_vac(2802.705), cloudy_label='Mg 2  2802.71A'),
    Line(name='Blnd  4720.00A', label='', wav=4.720000000000000000e+03, cloudy_label='Blnd  4720.00A'),
    Line(name='Si 2  34.8046m', label='', wav=3.480460000000000000e+05, cloudy_label='Si 2  34.8046m'),
    Line(name='S  2  1.03364m', label='', wav=1.033639999999999964e+04, cloudy_label='S  2  1.03364m'),
    Line(name='S  2  6730.82A', label='', wav=6.730819999999999709e+03, cloudy_label='S  2  6730.82A'),
    Line(name='S  2  6716.44A', label='', wav=6.716439999999999600e+03, cloudy_label='S  2  6716.44A'),
    Line(name='S  2  4068.60A', label='', wav=4.068599999999999909e+03, cloudy_label='S  2  4068.60A'),
    Line(name='S  2  4076.35A', label='', wav=4.076349999999999909e+03, cloudy_label='S  2  4076.35A'),
    Line(name='S  3  18.7078m', label='', wav=1.870780000000000000e+05, cloudy_label='S  3  18.7078m'),
    Line(name='S  3  33.4704m', label='', wav=3.347040000000000000e+05, cloudy_label='S  3  33.4704m'),
    Line(name='S  3  9530.62A', label='', wav=9.530620000000000800e+03, cloudy_label='S  3  9530.62A'),
    Line(name='S  3  9068.62A', label='', wav=9.068620000000000800e+03, cloudy_label='S  3  9068.62A'),
    Line(name='S  3  6312.06A', label='', wav=6.312060000000000400e+03, cloudy_label='S  3  6312.06A'),
    Line(name='S  3  3721.63A', label='', wav=3.721630000000000109e+03, cloudy_label='S  3  3721.63A'),
    Line(name='S  4  10.5076m', label='', wav=1.050760000000000000e+05, cloudy_label='S  4  10.5076m'),
    Line(name='Ar 2  6.98337m', label='', wav=6.983369999999999709e+04, cloudy_label='Ar 2  6.98337m'),
    Line(name='Ar 3  7135.79A', label='', wav=7.135789999999999964e+03, cloudy_label='Ar 3  7135.79A'),
    Line(name='Ar 3  7751.11A', label='', wav=7.751109999999999673e+03, cloudy_label='Ar 3  7751.11A'),
    Line(name='Ar 3  5191.82A', label='', wav=5.191819999999999709e+03, cloudy_label='Ar 3  5191.82A'),
    Line(name='Ar 3  3109.18A', label='', wav=3.109179999999999836e+03, cloudy_label='Ar 3  3109.18A'),
    Line(name='Ar 3  21.8253m', label='', wav=2.182530000000000000e+05, cloudy_label='Ar 3  21.8253m'),
    Line(name='Ar 3  8.98898m', label='', wav=8.988980000000000291e+04, cloudy_label='Ar 3  8.98898m'),
    Line(name='Ar 4  7332.15A', label='', wav=7.332149999999999636e+03, cloudy_label='Ar 4  7332.15A'),
    Line(name='Al 2  2669.15A', label='', wav=2.669150000000000091e+03, cloudy_label='Al 2  2669.15A'),
    Line(name='Al 2  2660.35A', label='', wav=2.660349999999999909e+03, cloudy_label='Al 2  2660.35A'),
    Line(name='Al 2  1855.93A', label='', wav=1.855930000000000064e+03, cloudy_label='Al 2  1855.93A'),
    Line(name='Al 2  1862.31A', label='', wav=1.862309999999999945e+03, cloudy_label='Al 2  1862.31A'),
    Line(name='Cl 2  14.3639m', label='', wav=1.436390000000000000e+05, cloudy_label='Cl 2  14.3639m'),
    Line(name='Cl 2  8578.70A', label='', wav=8.578700000000000728e+03, cloudy_label='Cl 2  8578.70A'),
    Line(name='Cl 2  9123.60A', label='', wav=9.123600000000000364e+03, cloudy_label='Cl 2  9123.60A'),
    Line(name='Cl 3  5537.87A', label='', wav=5.537869999999999891e+03, cloudy_label='Cl 3  5537.87A'),
    Line(name='Cl 3  5517.71A', label='', wav=5.517710000000000036e+03, cloudy_label='Cl 3  5517.71A'),
    Line(name='P  2  60.6263m', label='', wav=6.062630000000000000e+05, cloudy_label='P  2  60.6263m'),
    Line(name='P  2  32.8620m', label='', wav=3.286200000000000000e+05, cloudy_label='P  2  32.8620m'),
    Line(name='Fe 2  1.25668m', label='', wav=1.256679999999999927e+04, cloudy_label='Fe 2  1.25668m'),
]
linelist = LineList(lines)