#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:33:37 2023

@author: jdcooper
"""
#%% General Imports
import numpy as np
import copy

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

import sea_urchin.alignement.align as ali
import sea_urchin.clustering.metrics as met

#%% Custom Imports

from RootNode import RootNode

#%% Trident
class Trident():
    '''
    Estimator/Generator fit on structures to target XAS and ground state energy
    '''
    def __init__(self, models, e_cv= {}, i_cv= {}):
        '''
        Parameters
        ----------
        e_cv : dict, optional
            energy cross validation parameters to be fed to GridSearchCV. The default is {}.
        i_cv : dict, optional
            intensity cross validation parameters to be fed to GridSearchCV. The default is {}.
        '''
        self.set_cv_params(models, e_cv, i_cv)
    
    def set_cv_params(self,models={},e_cv={},i_cv={}):
        '''
        Manually Set Cross-Validation ranges for energy and intensity models

        Parameters
        ----------
        models : dict, optional
            Target estimator models. The default is {'energy' : Ridge, 'intensity': KernelRidge}.
        e_cv : dict, optional
            energy cross validation parameters to be fed to GridSearchCV. The default is {}.
        i_cv : dict, optional
            intensity cross validation parameters to be fed to GridSearchCV. The default is {}.
        '''
        self.model_params = {
                'energy' : Ridge,
                'intensity' : KernelRidge
            }
        self.model_params.update(models)
        
        # All models need alpha range
        alpha_range = {'alpha' : np.logspace(-1,5)}
        
        self.CV_params = {
                'energy' : alpha_range | e_cv,
                'intensity' : alpha_range | i_cv
            }
        
    def fit(self, atoms_, XAS_en, XAS_inten, XAS_ovlps, ref):
        self.init_tree(atoms_, ref)
        self.model_tree.set_XAS(
                            energies= XAS_en, 
                            intensities= XAS_inten, 
                            ovlps= XAS_ovlps
                            )
        self.model_tree.fit()
    
    def init_tree(self, atoms_, ref=None, align= {}):
        '''
        Initialize Internal Model Tree with unalligned atoms and reference structures.

        Parameters
        ----------
        atoms_ : list(Atoms)
            Unalligned atoms to be fit to spectral features.
        ref : list(Atoms)
            Reference Structures to allign atom_ to.
        align : dict
            Allignment parameters
        '''
        ## Set Internal Reference to atoms ##
        mutable_atoms = copy.deepcopy(atoms_)
        self.ref = ref
        
        ## Align ##
        mutable_atoms = Trident.align_atoms(mutable_atoms, ref, align)
        seperations = met.get_distances(mutable_atoms)
        
        self.model_tree = RootNode(seperations)
        self.model_tree.atoms= mutable_atoms
        self.model_tree.setModelParams(estimator_params= self.model_params, cv_params= self.CV_params)
        
    def predict(self, atoms_):
        mutable_atoms = copy.deepcopy(atoms_)
        
        mutable_atoms = Trident.align_atoms(mutable_atoms, self.ref)            ### CHECK IF THERE IS AN ISSUE WITH ALIGNING TEST ATOMS TO
        seperations = met.get_distances(mutable_atoms)                          ### TO REFERENCES DIRECTLY OR NEED TO BE ALIGNED WITH
                                                                                ### TRAINING ATOMS
        prediction_energies, prediction_intensities = self.model_tree.predict(seperations)
        
        return prediction_energies, prediction_intensities
        
    def align_atoms(atoms, ref, align= {}):
        ## Align ##
        alignment = {
                "type"      : "fastoverlap",
                "permute" : "elements",
                "inversion" : False
                } | align
        
        # Align
        if ref == None:
            mutable_atoms, __ = ali.align_to_mean_structure(
                                                atoms, 
                                                alignment, 
                                                nmax= 10, 
                                                start_structure= None
                                                )
        else:
            mutable_atoms, __ = ali.align_clusters_to_references_parallel(
                                                atoms, 
                                                ref,
                                                alignment
                                                )
        return mutable_atoms