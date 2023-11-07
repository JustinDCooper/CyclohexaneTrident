#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:42:51 2023

@author: jdcooper
"""
#%%Imports
import sea_urchin.clustering.metrics as met
import umap
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
import dill
import colorcet as cc
import numpy as np
from sklearn.preprocessing import StandardScaler

#%% xyzreader
class atomReader():
    
    def __init__(self, atoms):
        self.fit(atoms)
    
    def fit(self, atoms):
        self.atoms = atoms
        self.distances = met.get_distances(atoms)
        
    def projection(self):
        
        ### Project ###
        projector = umap.UMAP(
                    n_neighbors= 15,
                    n_components= 2
                    )
        
        u = projector.fit_transform(self.distances)
        
        ### Cluster ###
        clusterer = hdbscan.HDBSCAN(min_cluster_size= 10, min_samples = 1, allow_single_cluster= True).fit(u)
        palette = sns.color_palette('Paired', 12)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
        
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
        
        
        ### Plot Projections ###
        plt.figure()
        plt.scatter(u[:,0], u[:,1], c = cluster_member_colors, alpha= 0.6, s= 12)
        plt.title('UMAP Projection')
        plt.xlabel('U1')
        plt.ylabel('U2')

#%% spectraReader

class spectraReader():
    def __init__(self,pkl):
        with open(pkl, "rb") as fin:
                    enerlist, intenlist, ovlps = dill.load(fin)
        
        self.fit(enerlist,intenlist,ovlps)
    
    def fit(self, energy, intensity, ovlps):
        self.energies = np.array(energy)
        self.intensities = np.array(intensity).mean(axis=1)
        self.ovlps = np.array(ovlps)
        
    def projection(self, lastpk = None, dim = 2):
        if lastpk == None:
            lastpk = self.enerlist.shape[1]
        
        IDshape = self.ovlps.shape[2]
            
        ovrlps = self.ovlps[:,:lastpk,:].reshape(-1,IDshape)
        
        # Scaling
        scaler = StandardScaler()
        ovrlps_scaled = scaler.fit_transform(ovrlps)
        
        ### UMAP ###
        proj = umap.UMAP(n_components= dim, n_neighbors= 10, min_dist= 0.0)
        
        self.ovlp_umap = proj.fit_transform(ovrlps_scaled)
        
        ### Cluster ###
        clusterer = hdbscan.HDBSCAN(min_cluster_size= 25, cluster_selection_epsilon= 0.2).fit(self.ovlp_umap)
        
        self.labels = clusterer.labels_
        self.label_names= np.unique(self.labels)[1:]


        ### Extraction ###
        
        self.__clusters = []
        
        en = self.energies[:,:lastpk]
        inten = self.intensities[:,:lastpk]
        
        # Outliers
        outlier_idx = self.labels == -1
        out_ei_idx = outlier_idx.reshape(en.shape)
        self.outliers = (ovrlps[outlier_idx], en[out_ei_idx], inten[out_ei_idx])
        
        # Clusters
        for label in self.label_names:
            label_idx = (self.labels == label).reshape(en.shape)
            
            label_ovlp = []
            label_en = []
            label_inten = []
            for row_idx, ovlp_r, en_r, inten_r in zip(label_idx,self.ovlps[:,:lastpk,:],en,inten):
                label_ovlp.append(ovlp_r[row_idx].tolist())
                label_en.append(en_r[row_idx].tolist())
                label_inten.append(inten_r[row_idx].tolist())
            
            self.__clusters.append((label_ovlp, label_en, label_inten))
                
        
        ### Plot ###
        self.show_ovrlp_projections()
        self.show_spectral_clusters()
            
        return clusterer.labels_.reshape(-1,lastpk)
                
                  
    def show_ovrlp_projections(self):
        palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(self.labels)))
        
        plt.figure()
        
        for label in self.label_names:
            labeled_idx = self.labels==label
            
            
            plt.scatter(
                        self.ovlp_umap[labeled_idx,0],
                        self.ovlp_umap[labeled_idx,1],
                        c= palette[label],
                        label= label,
                        s= 5)
            
        label_count = len(np.unique(self.labels))
        plt.title(f'Number of labels: {label_count}')
        plt.legend()
        plt.show()
        
    def show_spectral_clusters(self):
        palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(self.labels)))
        
        plt.figure()
        
        for label in self.label_names:
            energy = np.concatenate(self.get_cl_energy(label))
            inten = np.concatenate(self.get_cl_intensity(label))
            plt.scatter(
                    energy,
                    inten**2,
                    c = palette[label],
                    label = label,
                    s= 5
                )
        plt.xlabel('Energy')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        
        
    ### Accessors ###
    
    def get_cl_ovrlp(self,cl):
        return self.__clusters[cl][0]
        
    def get_cl_energy(self, cl):
        return self.__clusters[cl][1]
    
    def get_cl_intensity(self,cl):
        return self.__clusters[cl][2]
    
        
        
        
        
        
        
        
        
        
        
        
        