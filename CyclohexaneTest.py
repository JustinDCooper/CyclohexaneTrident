#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:58:22 2023

@author: jdcooper
"""



#%% Imports
from ase.io import read
import numpy as np
import dill

from sklearn.kernel_ridge import KernelRidge

#%%
reference_structs = read("C6H12_references.xyz", index= ':')
atoms = read("cyclohexane_lammps.xyz", index= ':')

with open('lammps_traj_THISISTHEONE.pkl', "rb") as fin:
            enerlist, intenlist, ovlps = dill.load(fin)
            
last_peak = 25
enerlist = np.array(enerlist)[:,:last_peak]
intenlist = (np.array(intenlist)**2).mean(axis=1)[:,:last_peak]
ovlps = np.array(ovlps)[:,:last_peak,:]

#%%
from TridentTest import Trident

trident = Trident()

models = {
    'energy' : KernelRidge,
    'intensity' : KernelRidge
    }

e_cv_params = {
    'alpha' : np.logspace(-5,1,10),
    'gamma' : np.logspace(-5,1,10),
    'kernel' : ['rbf']
    }

i_cv_params = {
    'alpha' : np.logspace(-5,1,10),
    'gamma' : np.logspace(-5,1,10),
    'kernel' : ['rbf']
    }

trident.set_cv_params(models,e_cv=e_cv_params,i_cv=i_cv_params)

trident.fit(atoms,enerlist,intenlist,ovlps,atom_ref=reference_structs)

#%%

y = trident.predict(atoms[:50])