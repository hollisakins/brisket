import numpy as np
import h5py 
from brisket.utils import utils

imf = 'chabrier'
spectra = 'miles'
# age_sampling = 'native'
age_sampling = np.linspace(6., np.log10(13.78)+9., 50)

keys = ['m22','m32','m42','m52','m62','m72','m82']
zmets = [0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.]

if imf == 'chabrier':
    imf2 = 'chab'
if spectra == 'miles':
    spectra2 = 'xmiless'
if imf == 'kroupa':
    imf2 = 'kroup'


Nzmets = len(zmets)
Nages, Nwavs = 0, 0
for key in keys:
    with open(f'brisket/data/raw_data/bc03/bc03_{spectra}_{imf}/bc2003_hr_{spectra2}_{key}_{imf2}_ssp.ised_ASCII', 'r') as f:
        lines = f.readlines()
        ages = np.array(lines[0].split()[1:],dtype=float)
        wavs = np.array(lines[6].split()[1:],dtype=float)
        if Nages==0:
            Nages = len(ages)
        assert Nages==len(ages)
        if Nwavs==0:
            Nwavs = len(wavs)
        assert Nwavs==len(wavs)

grid = np.zeros((Nzmets, Nages, Nwavs))
live_frac = np.zeros((Nzmets,Nages))
for i,key,zmet in zip(np.arange(len(zmets)),keys,zmets):
    with open(f'brisket/data/raw_data/bc03/bc03_{spectra}_{imf}/bc2003_hr_{spectra2}_{key}_{imf2}_ssp.ised_ASCII', 'r') as f:
        lines = f.readlines()
    
        ages = np.array(lines[0].split()[1:],dtype=float)
        wavelengths = np.array(lines[6].split()[1:],dtype=float)

        seds = np.zeros((len(ages),len(wavelengths)))
        for j,line in enumerate(lines[7:-12]):
            seds[j] = np.array(line.split()[1:-53],dtype=float)

        livfrac = np.array(lines[-11].split()[1:],dtype=float)

        grid[i] = seds
        live_frac[i] = livfrac



if type(age_sampling)==str:
    assert age_sampling == 'native'
    outfilepath = f'brisket/data/bc03_{spectra}_{imf}_native.hdf5'
else:
    outfilepath = f'brisket/data/bc03_{spectra}_{imf}_a{len(age_sampling)}.hdf5'
    # Set up edge positions for age bins for stellar + nebular models.
    age_bins = np.power(10., utils.make_bins(age_sampling, make_rhs=True)[0])
    age_bins[0] = 0.
    age_bins[-1] = 13.78e9
    age_widths = age_bins[1:] - age_bins[:-1]

    #### set up the grids to the configured age sampling
    grid_resampled = np.zeros((Nzmets, len(age_sampling), Nwavs))

    raw_age_lhs, raw_age_widths = utils.make_bins(ages, make_rhs=True)
    # Force raw ages to span full range from 0 to age of Universe.
    raw_age_widths[0] += raw_age_lhs[0]
    raw_age_lhs[0] = 0.

    if raw_age_lhs[-1] < age_bins[-1]:
        raw_age_widths[-1] += age_bins[-1] - raw_age_lhs[-1]
        raw_age_lhs[-1] = age_bins[-1]

    start = 0
    stop = 0

    # Loop over the new age bins
    for j in range(len(age_bins) - 1):

        # Find the first raw bin partially covered by the new bin
        while raw_age_lhs[start + 1] <= age_bins[j]:
            start += 1

        # Find the last raw bin partially covered by the new bin
        while raw_age_lhs[stop+1] < age_bins[j + 1]:
            stop += 1

        # If new bin falls completely within one raw bin
        if stop == start:
            grid_resampled[:, j, :] = grid[:, start, :]

        # If new bin has contributions from more than one raw bin
        else:
            start_fact = ((raw_age_lhs[start + 1] - age_bins[j])
                            / (raw_age_lhs[start + 1] - raw_age_lhs[start]))

            end_fact = ((age_bins[j + 1] - raw_age_lhs[stop])
                        / (raw_age_lhs[stop + 1] - raw_age_lhs[stop]))

            raw_age_widths[start] *= start_fact
            raw_age_widths[stop] *= end_fact

            width_slice = raw_age_widths[start:stop + 1]

            summed = np.sum(np.expand_dims(width_slice, axis=1)
                            * grid[:, start:stop + 1, :], axis=1)

            grid_resampled[:, j, :] = summed/np.sum(width_slice)

            raw_age_widths[start] /= start_fact
            raw_age_widths[stop] /= end_fact

    live_frac_resampled = np.zeros((Nzmets, len(age_sampling)))
    for i in range(Nzmets):
        live_frac_resampled[i, :] = np.interp(age_sampling, ages, live_frac[i, :])

    ages = age_sampling
    grid = grid_resampled
    live_frac = live_frac_resampled


with h5py.File(outfilepath, 'w') as hf:
    hf.create_dataset('axes', data=['metallicities','ages','wavs'])
    hf.create_dataset('metallicities', data=zmets)
    hf.create_dataset('ages', data=ages)
    hf.create_dataset('wavs', data=wavs)
    hf.create_dataset('live_frac', data=live_frac)
    hf.create_dataset('grid', data=grid)