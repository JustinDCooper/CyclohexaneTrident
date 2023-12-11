#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:39:22 2023

@author: jdcooper
"""
#%% imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from Helpers import standardCount

#%% LeafEstimator
class LeafEstimator:
    def __init__(self, ID, root):
        self.ID = ID
        self.root = root
        
        self.energies_scaler = None
        self.intensities_scaler = None
        
        self.energy_estimator = None
        self.intensity_estimator = None
        
    def fit(self):
        estimator_params = self.root.root.estimator_params
        cv_params = self.root.root.cv_params
        
        X = self.seperations
        X = np.concatenate( (X, np.ones((X.shape[0],1))), axis= 1)
        
        y_energy = self.XAS_energies
        self.energies_scaler = StandardScaler().fit(y_energy)
        y_energy = self.energies_scaler.transform(y_energy)
        
        # Cross Validation
        e_Cross_validator = GridSearchCV(
            estimator_params['energy'](),
            param_grid= cv_params['energy'],
        )
        e_Cross_validator.fit(X,y_energy)
        
        ## Verbage ##
        print(f"Cluster {self.ID} Energy Estimator R2 score: {e_Cross_validator.best_score_:.3f}")
        print(f"Cluster {self.ID} Energy Estimator params: {e_Cross_validator.best_params_}", end= '\n\n')
        self.energy_estimator = e_Cross_validator.best_estimator_
        
        
        y_intensity = self.XAS_intensities
        self.intensities_scaler = StandardScaler().fit(y_intensity)
        y_intensity = self.intensities_scaler.transform(y_intensity)
        
        # Cross Validation
        i_Cross_validator = GridSearchCV(
            estimator_params['intensity'](),
            param_grid= cv_params['intensity'],
        )
        i_Cross_validator.fit(X,y_intensity)
        
        ## Verbage ##
        print(f"Cluster {self.ID} Intensity Estimator R2 score: {i_Cross_validator.best_score_:.3f}")
        print(f"Cluster {self.ID} Intensity Estimator params: {i_Cross_validator.best_params_}", end= '\n\n')
        self.intensity_estimator = i_Cross_validator.best_estimator_
        
    def predict(self, seperations):
        local_seperations = seperations.copy()
        local_seperations[len(seperations.columns)] = pd.Series(1,index= seperations.index)
        energies_scaled_pred = self.energy_estimator.predict(local_seperations)
        energies_pred = pd.DataFrame(self.energies_scaler.inverse_transform(energies_scaled_pred), index= local_seperations.index)
        
        intensities_scaled_pred = self.intensity_estimator.predict(local_seperations)
        intensities_pred = pd.DataFrame(self.intensities_scaler.inverse_transform(intensities_scaled_pred), index= local_seperations.index)
        
        return energies_pred, intensities_pred
    
    #%% Property Methods (Private Attributes)
    @property
    def standard_samples(self):
        standard_transition_frequency = standardCount(self.root.labels,[self.ID[1]])
        valid_samples = np.sum(self.root.labels == self.ID[1], axis= 1) == standard_transition_frequency
        return valid_samples
    @property
    def seperations(self):
        return self.root.seperations[self.standard_samples]
    @property
    def XAS_energies(self):
        std_samples = self.standard_samples
        std_XAS_energies = self.root.XAS_energies[std_samples]
        std_cluster_energies = std_XAS_energies[self.root.labels[std_samples] == self.ID[1]].reshape((std_XAS_energies.shape[0],-1))
        return std_cluster_energies
    @property
    def XAS_intensities(self):
        std_samples = self.standard_samples
        std_XAS_intensities = self.root.XAS_intensities[std_samples]
        std_cluster_intensities = std_XAS_intensities[self.root.labels[std_samples] == self.ID[1]].reshape((std_XAS_intensities.shape[0],-1))
        return std_cluster_intensities