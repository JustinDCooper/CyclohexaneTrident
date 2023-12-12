#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:01:21 2023

@author: jdcooper
"""

import pandas as pd
from ase.io import read
import numpy as np
import dill

from sklearn.model_selection import train_test_split

from Trident import Trident
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge

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

#%%

models = {
    'energy' : KernelRidge,
    'intensity' : KernelRidge
    }

e_cv_params = {
        'alpha' : np.logspace(-5,3,30),
        'gamma' : np.logspace(-7,2,30),
        'kernel' : ['rbf']
    }
i_cv_params = {
        'alpha' : np.logspace(-5,3,30),
        'gamma' : np.logspace(-7,2,30),
        'kernel' : ['rbf']
    }

trident = Trident(
            models= models,
            e_cv= e_cv_params,
            i_cv=i_cv_params
            )

trident.fit(
        atoms_= atoms_train,
        XAS_en= e_train,
        XAS_inten= i_train,
        XAS_ovlps= ovlps_train,
        ref=reference_structs)

#%%

e_pred_df, i_pred_df = trident.predict(atoms_test)
#%%

from sklearn.metrics import mean_squared_error

e_true = e_test
i_true = i_test

e_pred = np.array(e_pred_df)
i_pred = np.array(i_pred_df)

e_mse = mean_squared_error(e_true.T, e_pred.T, multioutput= 'raw_values')
i_mse = mean_squared_error(i_true.T, i_pred.T, multioutput= 'raw_values')

#%%

def plot_spectra(ax, energy, intensity, labeling= True, **kwargs):
    e_min = np.min(energy)
    e_max = np.max(energy)
    dE = (e_max - e_min)*0.05
    erange= np.linspace(e_min - dE,e_max + dE, num=600, endpoint=True)
    spec = stick_to_spectra(energy, intensity, 0.005, erange)
    
    if labeling:
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity')
    
    ax.plot(erange,spec, **kwargs)
    ax.plot([e_min-2*dE,e_max+2*dE],[0,0], c='k', linewidth = 1)

def stick_to_spectra(E,osc,sigma,x):
    matrix = E.ndim > 1
    if matrix:
        spectra= []
        for r in range(osc.shape[0]):
            spectra.append(stick_to_spectra(E[r,:],osc[r,:],sigma,x, matrix= False))
        spectra= np.array(spectra)
        return spectra
    else:
        gE=[]
        for Ei in x:
            tot=0
            for Ej,os in zip(E,osc):
                tot+=os*np.exp(-(((Ej-Ei)/sigma)**2))
            gE.append(tot)
        return gE

#%%

fig = plt.figure(figsize= (10,7), layout='constrained')
axs = fig.subplot_mosaic([["Best E MSE", "Best I MSE"],["Worst E MSE", "Worst I MSE"]])

for (label, ax), error in zip(axs.items(),[e_mse.min(),i_mse.min(),e_mse.max(),i_mse.max()]):
    ax.set_title(f"{label}, MSE: {error:.4f}")
    
plot_spectra(axs["Best E MSE"], e_true[e_mse.argmin()],i_true[e_mse.argmin()], c= 'orange', label= "True")
plot_spectra(axs["Best E MSE"], e_pred[e_mse.argmin()],i_pred[e_mse.argmin()], c= 'purple', label= "Predicted")
plot_spectra(axs["Best E MSE"], e_true.mean(axis=0),i_true.mean(axis=0), c= 'grey', label= "Mean", alpha=0.3, linestyle='dashed')

plot_spectra(axs["Best I MSE"], e_true[i_mse.argmin()],i_true[i_mse.argmin()], c= 'orange', label= "True")
plot_spectra(axs["Best I MSE"], e_pred[i_mse.argmin()],i_pred[i_mse.argmin()], c= 'purple', label= "Predicted")
plot_spectra(axs["Best I MSE"], e_true.mean(axis=0),i_true.mean(axis=0), c= 'grey', label= "Mean", alpha=0.3, linestyle='dashed')


plot_spectra(axs["Worst E MSE"], e_true[e_mse.argmax()],i_true[e_mse.argmax()], c= 'orange', label= "True")
plot_spectra(axs["Worst E MSE"], e_pred[e_mse.argmax()],i_pred[e_mse.argmax()], c= 'purple', label= "Predicted")
plot_spectra(axs["Worst E MSE"], e_true.mean(axis=0),i_true.mean(axis=0), c= 'grey', label= "Mean", alpha=0.3, linestyle='dashed')

plot_spectra(axs["Worst I MSE"], e_true[i_mse.argmax()],i_true[i_mse.argmax()], c= 'orange', label= "True")
plot_spectra(axs["Worst I MSE"], e_pred[i_mse.argmax()],i_pred[i_mse.argmax()], c= 'purple', label= "Predicted")
plot_spectra(axs["Worst I MSE"], e_true.mean(axis=0),i_true.mean(axis=0), c= 'grey', label= "Mean", alpha=0.3, linestyle='dashed')
plt.legend()

fig.suptitle("Cyclohexane")

#%%
fig1, ax1 = plt.subplots(2,2)
negatives = [42,23,100,54]
for row in range(2):
    for col in range(2):
        neg_ind = negatives[row*2 + col]
        plot_spectra(ax1[row,col], e_true[neg_ind],i_true[neg_ind], c= 'orange', label= "True")
        plot_spectra(ax1[row,col], e_pred[neg_ind],i_pred[neg_ind], c= 'purple', label= "Predicted")
        plot_spectra(ax1[row,col], e_true.mean(axis=0),i_true.mean(axis=0), c= 'grey', label= "Mean", alpha=0.3, linestyle='dashed')

plt.legend()

fig1.suptitle("Cyclohexane")


