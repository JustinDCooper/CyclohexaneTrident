#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:38:30 2023

@author: jdcooper
"""
#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Custom Imports

from TreeNode import TreeNode
from LeafEstimator import LeafEstimator


#%% BranchNode
class BranchNode(TreeNode):
    def fit(self):
        self.fit_projection()
        self.fit_cluster()
        self.partition()
        for estimator in self.children:
            estimator.fit()            
    
    def fit_projection(self, params= {}):
        sample_count, transition_count, LIVO_count = self.XAS_ovlps.shape           #For Reformatting ovlps
        individual_ovlps = np.reshape(self.XAS_ovlps, ( sample_count * transition_count, LIVO_count )) # Reformated OVLPS
        
        TreeNode.fit_projection(self, 
                                   data= individual_ovlps, 
                                   params= params)
    
    def fit_cluster(self, params= {}):        
        sample_count, transition_count, LIVO_count = self.XAS_ovlps.shape           #For Reformatting ovlps
        individual_ovlps = np.reshape(self.XAS_ovlps, ( sample_count * transition_count, LIVO_count ))
        
        TreeNode.fit_cluster(self, 
                                data= individual_ovlps, 
                                params= params)
        
    def partition(self):
        self.children.clear()
        labels = set(self.labels.reshape(-1))
        labels.discard(-1)
        
        for label in labels:
            subNode = LeafEstimator(
                            ID= (self.ID,label),
                            root= self)
            
            self.add_child(subNode)
            
    def predict(self, seperations):
        enerlist, intenlist = [], []
        for estimator in self.children:
            cl_energies, cl_intensities = estimator.predict(
                                                    seperations= seperations
                                                    )
            enerlist.append(cl_energies)
            intenlist.append(cl_intensities)
            
        XAS_energies_pred = pd.concat(enerlist,axis=1)
        XAS_intensities_pred = pd.concat(intenlist,axis=1)
        
        XAS_energies_pred.columns = range(len(XAS_energies_pred.columns))
        XAS_intensities_pred.columns = range(len(XAS_intensities_pred.columns))
        
        return XAS_energies_pred, XAS_intensities_pred
    
    def plot_projections(self):
        sample_count, transition_count, LIVO_count = self.XAS_ovlps.shape
        individual_ovlps = np.reshape(self.XAS_ovlps, ( sample_count * transition_count, LIVO_count )) # Reformated OVLPS
        TreeNode.plot_projections(self, individual_ovlps)

#%% Visuals
    def plot_clusters(self, samples = None):        
        if samples == None:
            samples = np.full(self.labels.shape[0], True)
        elif type(samples) == int:
            samples = [samples]
        if type(samples) == list:
            temp = np.full(self.labels.shape[0], False)
            temp[samples] = True
            samples= temp
            
        labels = self.labels[samples].reshape(-1)
        energies = self.XAS_energies[samples].reshape(labels.shape)
        intensities = self.XAS_intensities[samples].reshape(labels.shape)
        
        ### Cluster Colors ###
        true_label_count = len(set(labels[labels != -1]))
        palette = sns.color_palette('Paired', true_label_count)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in labels]
        
        cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                          zip(cluster_colors, self.clusterer.probabilities_)])
        
        plt.figure()
        for atom_label in set(labels):
            mask = (labels == atom_label)
            plt.scatter(energies[mask], intensities[mask], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)
        
        
    
    #%% Property Methods (Private Attributes)
    # Could be made virtual but I think it would make it harder to read
    @property
    def labels(self):
        # return np.tile(np.arange(self.XAS_energies.shape[1]),(self.XAS_energies.shape[0],1))
        return np.reshape(self.clusterer.labels_,self.XAS_energies.shape)
    @property
    def seperations(self):
        return self.root.seperations[self.root.labels == self.ID]
    @property
    def XAS_energies(self):
        return self.root.XAS_energies[self.root.labels == self.ID]
    @property
    def XAS_intensities(self):
        return self.root.XAS_intensities[self.root.labels == self.ID]
    @property
    def XAS_ovlps(self):
        return self.root.XAS_ovlps[self.root.labels == self.ID]
    @property
    def atoms(self):
        return [self.root.atoms[i] for i in range(len(self.root.atoms)) if (self.root.labels == self.ID)[i]]
    
    #%% Dunder
    
    def __getitem__(self, index) -> LeafEstimator:
        return TreeNode.__getitem__(self, index)