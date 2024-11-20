import numpy as np
import h5py 

imf = 'kroupa'
spectra = 'miles'

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



with h5py.File(f'brisket/data/bc03_{spectra}_{imf}.hdf5', 'w') as hf:
    hf.create_dataset('axes', data=['metallicities','ages','wavs'])
    hf.create_dataset('metallicities', data=zmets)
    hf.create_dataset('ages', data=ages)
    hf.create_dataset('wavs', data=wavs)
    hf.create_dataset('live_frac', data=live_frac)
    hf.create_dataset('grid', data=grid)