#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:37:09 2023

@author: jdcooper
"""
#%% Imports
import pandas as pd
import hdbscan


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

#%% Custom Imports

from TreeNode import TreeNode
from BranchNode import BranchNode


#%% RootNode
class RootNode(TreeNode):
    def __init__(self, seperations, ID='Root'):
        self.seperations_scaler = StandardScaler().fit(seperations)
        self.__seperations = self.seperations_scaler.transform(seperations)
        self.__XAS_energies = None
        self.__XAS_intensities = None
        self.__XAS_ovlps = None
        TreeNode.__init__(self, ID)
    
    def setModelParams(self, estimator_params= {}, cv_params= {}):
        self.estimator_params = {
                'energy' : Ridge,
                'intensity' : KernelRidge
            } | estimator_params
        self.cv_params = cv_params
    
    def set_XAS(self, energies, intensities, ovlps):
        self.__XAS_energies = energies
        self.__XAS_intensities = intensities
        self.__XAS_ovlps = ovlps
    
    def fit(self):
        self.fit_projection()
        self.fit_cluster()
        self.partition()
        for branch in self.children:
            branch.fit()
        
    def fit_projection(self, params= {}):        
        TreeNode.fit_projection(self, 
                                   data= self.__seperations,
                                   params= params)
    
    def fit_cluster(self, params= {}):
        TreeNode.fit_cluster(self, 
                                data= self.__seperations,
                                params= params)
        
    def partition(self):
        self.children.clear()
        labels = set(self.labels)
        labels.discard(-1)
        
        for label in labels:
            subNode = BranchNode( 
                            ID= label, 
                            root= self
                            )
            
            self.add_child(subNode)
            
    def predict(self, seperations):
        test_seperations = pd.DataFrame(self.seperations_scaler.transform(seperations))
        
        u = self.projector.transform(test_seperations)
        test_labels, _ = hdbscan.approximate_predict(self.clusterer,u)
        
        XAS_enlist, XAS_intenlist = [], []
        for branch in self.children:
            sample_energies, sample_intensities = branch.predict(
                                                            seperations= test_seperations[test_labels == branch.ID]
                                                            )
            XAS_enlist.append(sample_energies)
            XAS_intenlist.append(sample_intensities)
            
        XAS_energies = pd.concat(XAS_enlist)
        XAS_intensities = pd.concat(XAS_intenlist)
        
        XAS_energies.sort_index(inplace= True)
        XAS_intensities.sort_index(inplace= True)
        
        return XAS_energies, XAS_intensities
    
    def plot_projections(self):
        TreeNode.plot_projections(self, self.seperations)
            
    #%% Property Methods (Private Attributes)
    @property
    def seperations(self):
        return self.__seperations
    @property
    def labels(self):
        return self.clusterer.labels_
    @property
    def XAS_energies(self):
        return self.__XAS_energies
    @property
    def XAS_intensities(self):
        return self.__XAS_intensities
    @property
    def XAS_ovlps(self):
        return self.__XAS_ovlps