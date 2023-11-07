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

valid_count = np.count_nonzero(np.all(struct_label_count == standard_count,axis=1))

#%% 
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

cl = 0

mask = struct_label_count[:,cl] == standard_count[cl]

X_cl_train = molecules.distances[mask]
energy_train = np.array([inner for inner in spectra.get_cl_energy(cl) if len(inner) == standard_count[cl]])
inten_train = np.array([inner for inner in spectra.get_cl_intensity(cl) if len(inner) == standard_count[cl]])
#%%

##Search
alpha_range= np.logspace(-4,1, num=10)
gamma_range= np.logspace(-8, -4, num= 10)

## Cross-Validation
CV_energy = GridSearchCV(
    KernelRidge(),
    param_grid={"alpha": alpha_range, "gamma": gamma_range, 'kernel' : ['rbf']},
)
CV_intensity = GridSearchCV(
    KernelRidge(),
    param_grid={"alpha": alpha_range, "gamma": gamma_range, 'kernel' : ['rbf']},
)

## Model Fitting
CV_energy.fit(X_cl_train, energy_train)
CV_intensity.fit(X_cl_train, inten_train)

print(f"Energy Estimator R2 score: {CV_energy.best_score_:.3f}")
print(f"Intensity Estimator R2 score: {CV_intensity.best_score_:.3f}")
print(f"Energy Estimator params: {CV_energy.best_params_}")
print(f"Intensity Estimator params: {CV_intensity.best_params_}")

#%%
from Trident import Trident

trident = Trident().fit(molecules.distances, energies[:,0], intensities[:,0])




