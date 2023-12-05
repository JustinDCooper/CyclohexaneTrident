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

spectra_cluster_params = {
                'min_cluster_size' : 10,
                'min_samples' : 7,
                'allow_single_cluster' : True,
                'cluster_selection_epsilon' : 0.2
    }
#%%
spectra_labels = spectra.projection(lastpk= 25, cluster_params_= spectra_cluster_params)

#%% Label validation


## How many occurances of a label in each structure
struct_label_count = np.array(
    [[np.count_nonzero(struct_label == lab) for lab in spectra.label_names] for struct_label in spectra_labels]
)

## How many sturctures have a certain number of occurances of a cluster
cluster_count_value = np.array([ np.bincount(collumn, minlength = len(spectra.label_names)) for collumn in struct_label_count.T])


## The standard number of occurances of each cluster
standard_count = np.argmax(cluster_count_value,axis = 1)

#%%
from sklearn.model_selection import train_test_split
from Trident import Trident
import sklearn.metrics.pairwise
import dill

#%%
for kernel in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS:
# for kernel in ['rbf']:

    result_energies = []
    result_intensities = []
    
    
    # Train-Test-Split
    
    for cl in spectra.label_names:
        X = molecules.get_cluster_distances(target_struct)
        energies = spectra.energies[:,:25]
        intensities = spectra.intensities[:,:25]
        label_idx = (spectra.labels == cl).reshape(energies.shape)
        
        X_train,X_test,en_train,en_test,inten_train,inten_test,label_train,label_test = train_test_split(X,energies,intensities,label_idx,random_state= 0)
            
        label_en = []
        label_inten = []
        for row_idx, en_r, inten_r in zip(label_train,en_train,inten_train):
            label_en.append(en_r[row_idx].tolist())
            label_inten.append(inten_r[row_idx].tolist())
        
        ### Feature Sectioning
        mask = np.count_nonzero(label_train, axis=1) == standard_count[cl]
        
        X_train = X_train[mask]
        energy_train = np.array([inner for inner in label_en if len(inner) == standard_count[cl]])
        inten_train = np.array([inner for inner in label_inten if len(inner) == standard_count[cl]])
        
        ###
       
        estimators = {
                'energies': 'KernelRidge',
                'intensities':'KernelRidge'
                }
        
        e_cv_params = {
                'alpha' : np.logspace(-5,-3,10),
                'kernel' : [kernel]
            }
        i_cv_params = {
                'alpha' : np.logspace(-3,1,20),
                'kernel' : [kernel]
            }
        if kernel in ['rbf', 'laplacian', 'polynomial', 'exponential', 'chi2', 'sigmoid','additive_chi2']:
            e_cv_params['gamma'] = np.logspace(-5,-3,10)
            i_cv_params['gamma'] = np.logspace(-7,-2,20)
        if kernel in ['polynomial','sigmoid']:
            e_cv_params['coef0'] = np.logspace(-1,5,10)
            i_cv_params['coef0'] = np.logspace(-1,5,20)
        if kernel == 'polynomial':
            e_cv_params['degree'] = np.arange(1,5)
            i_cv_params['degree'] = np.logspace(1,5)

        
        trident = Trident(
                    params= estimators, 
                    e_cv_params_= e_cv_params, 
                    i_cv_params_= i_cv_params
                    )
        
        ###
        trident.fit(
                        PAS_= X_train, 
                        energies_= energy_train,
                        intensities_= inten_train
                        )
        
        ###
        
        prediction_component = trident.predict(X_test)
        result_energies.append(prediction_component['energies'])
        result_intensities.append(prediction_component['intensities'])
    ###
    
    result_energies = np.concatenate(result_energies, axis = 1)
    result_intensities = np.concatenate(result_intensities,axis = 1)
    #%%
    with open(f"KernelAnalysis/{kernel}.pkl", "wb") as fout:
        dill.dump([result_energies, result_intensities, trident.estimator_params,trident.estimator_R2], fout)















