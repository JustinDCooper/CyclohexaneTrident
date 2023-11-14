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

reference_structs = read("C6H12_references.xyz", index= ':')
atoms = read("cyclohexane_lammps.xyz", index= ':')

molecules = atomReader(atoms, reference_structs)

molecules.projection()

#%%
from atomReader import spectraReader

target_struct = 1

spectra = spectraReader('lammps_traj_THISISTHEONE.pkl', labels= molecules.labels==target_struct)


#%%
spectra_cluster_params = {
                'min_cluster_size' : 10,
                'min_samples' : 7,
                'allow_single_cluster' : True,
                'cluster_selection_epsilon' : 0.2
    }

labels = spectra.projection(lastpk= 25, cluster_params_= spectra_cluster_params)

#%% Label validation

cluster_count_value = np.zeros((len(spectra.label_names),10))

for struct in labels:
    for lab in spectra.label_names:
        cluster_count_value[lab,struct.tolist().count(lab)] += 1
    
standard_count = np.argmax(cluster_count_value,axis = 1)

struct_label_count = np.zeros((labels.shape[0],len(spectra.label_names)))
for i, struct in enumerate(labels):
    for lab in spectra.label_names:
        struct_label_count[i,lab] = struct.tolist().count(lab)

#%% Feature Sectioning

## Index for target XAS cluster ######## IMPORTANT : First peak is not always indexed at 0. Make sure to check the XAS figure
cl = 1 ### this is the only transformation or search parameter that may need to be changed in current analysis ### Will update following proof of concept

mask = struct_label_count[:,cl] == standard_count[cl]

X_cl_train = molecules.get_cluster_distances(target_struct)[mask]
energy_train = np.array([inner for inner in spectra.get_cl_energy(cl) if len(inner) == standard_count[cl]])
inten_train = np.array([inner for inner in spectra.get_cl_intensity(cl) if len(inner) == standard_count[cl]])

#%%
from Trident import Trident

estimators = {
        'energies': 'KernelRidge',
        'intensities':'KernelRidge'
        }

e_cv_params = {
        'alpha' : np.logspace(-5,-3,10),
        'gamma' : np.logspace(-5,-3,10),
        'kernel' : ['rbf']
    }
i_cv_params = {
        'alpha' : np.logspace(-3,1,20),
        'gamma' : np.logspace(-7,-2,20),
        'kernel' : ['rbf']
    }

trident = Trident(
            params= estimators, 
            e_cv_params_= e_cv_params, 
            i_cv_params_= i_cv_params
            )

#%%
trident.fit(
                PAS_= X_cl_train, 
                energies_= energy_train,
                intensities_= inten_train
                )




