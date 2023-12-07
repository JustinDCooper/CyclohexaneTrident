#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:23:36 2023

@author: jdcooper
"""
import pandas as pd
from ase.io import read
import numpy as np
import dill

from sklearn.model_selection import train_test_split

import sklearn.metrics.pairwise

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

atoms_train, atoms_test, e_train, e_test, i_train, i_test, ovlps_train,_ = train_test_split(atoms,enerlist,intenlist,ovlps,test_size=0.2)

e_test_df = pd.DataFrame(e_test)
i_test_df = pd.DataFrame(i_test)

e_test_df.to_pickle('KernelAnalysis/true_test_energy.pkl')
i_test_df.to_pickle('KernelAnalysis/true_test_intensity.pkl')

#%%
from TridentTest import Trident
from sklearn.kernel_ridge import KernelRidge

#%%
prec = 50

for kernel in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS:
    if kernel in ['additive_chi2', 'chi2']:
        continue
    print(kernel)
    trident = Trident()

    models = {
        'energy' : KernelRidge,
        'intensity' : KernelRidge
        }

    e_cv_params = {
            'alpha' : np.logspace(-5,3,prec),
            'kernel' : [kernel]
        }
    i_cv_params = {
            'alpha' : np.logspace(-5,3,prec),
            'kernel' : [kernel]
        }
    
    if kernel in ['rbf', 'laplacian', 'polynomial', 'exponential', 'chi2', 'sigmoid','additive_chi2']:
        e_cv_params['gamma'] = np.logspace(-7,2,prec)
        i_cv_params['gamma'] = np.logspace(-7,2,prec)
    if kernel in ['polynomial','sigmoid']:
        e_cv_params['coef0'] = np.logspace(-1,5,10)
        i_cv_params['coef0'] = np.logspace(-1,5,20)
    if kernel == 'polynomial':
        e_cv_params['degree'] = np.arange(1,5)
        i_cv_params['degree'] = np.arange(1,5)

    trident.set_cv_params(models,e_cv=e_cv_params,i_cv=i_cv_params)

    trident.fit(atoms_train,e_train,i_train,ovlps_train,atom_ref=reference_structs)

    ## Make Predictions
    e_pred, i_pred = trident.predict(atoms_test)
    
    ## Export to pkl
    e_pred.to_pickle(f'KernelAnalysis/{kernel}_energy_prediction')
    i_pred.to_pickle(f'KernelAnalysis/{kernel}_intensity_prediction')
    
    
    