#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:35:38 2023

@author: jdcooper
"""
#%% imports
import umap
import hdbscan
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

#%% TreeNode
class TreeNode:
    projection_params = {
            'n_neighbors' : 5,
            'min_dist' : 0.0,
            'n_components' : 2,
            'random_state' : 3
        }
    cluster_params = {
            'min_cluster_size' : 100,
            # 'min_samples' : 7,
            'allow_single_cluster' : True,
            # 'cluster_selection_epsilon' : 0.5,
            'prediction_data' : True
        }
    def __init__(self, ID, root= None):
        self.ID = ID
        self.root = root
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)
        
    def fit_projection(self, data, params):
        self.projection_params = TreeNode.projection_params | params
        self.projector = umap.UMAP(**self.projection_params).fit(data)
    
    def fit_cluster(self, data, params):
        self.cluster_params = TreeNode.cluster_params | params
        
        u = self.projector.transform(data)
        
        self.clusterer = hdbscan.HDBSCAN(**self.cluster_params).fit(u)
    
    def __getitem__(self,index):
        assert index < len(self.children)
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
    def plot_projections(self, feature):
        labels = self.labels.reshape(-1)
        ### Cluster Colors ###
        true_label_count = len(set(labels[labels != -1]))
        palette = sns.color_palette('Paired', true_label_count)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in labels]
        
        cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                          zip(cluster_colors, self.clusterer.probabilities_)])
        
        ### Plot Projections ###
        
        u = self.projector.transform(feature)
        
        plt.figure()
        for atom_label in set(labels):
            mask = labels == atom_label
            plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)

        plt.title(f'{self.ID} UMAP Projection')
        plt.xlabel('U1')
        plt.ylabel('U2')
        plt.legend()
        pass