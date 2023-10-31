#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:32:32 2023

@author: jdcooper
"""
#%% Imports
from ase.io import read
import numpy as np
#%%
from atomReader import atomReader

atoms = read("ali_cyclohexane.xyz", index= ':')

molecules = atomReader(atoms)

molecules.projection()

#%%
from atomReader import spectraReader

spectra = spectraReader('asap_traj_fine.pkl')
labels = spectra.projection(25, dim= 2)


energies = spectra.energies[:,:25]
intensities = spectra.intensities[:,:25]

#%%
from Trident import Trident
#%%
target_peak = 1
mask = labels == target_peak

target_energies = (energies * mask).sum(axis = 1)/mask.sum(axis=1)
target_intensities = (intensities * mask).sum(axis = 1)/mask.sum(axis=1)

## Prune nan elements
mask = ~np.isnan(target_energies)

target_energies = target_energies[mask]
target_intensities = target_intensities[mask]
#%%
trident = Trident()

trident.fit(
    molecules.distances[mask,:], 
    energies_ = target_energies, 
    intensities_ = target_intensities)




