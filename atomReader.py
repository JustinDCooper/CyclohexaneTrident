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
import sea_urchin.alignement.align as ali

#%% xyzreader
class atomReader():
    
    def __init__(self, atoms, references=None):
        self.fit(atoms, references)
    
    def fit(self, atoms, references= None):
        
        alignment = {
                "type"      : "fastoverlap",
                "permute" : "elements",
                "inversion" : False
                }

        # Align
        if references == None:
            self.atoms, __ = ali.align_to_mean_structure(
                                                atoms, 
                                                alignment, 
                                                nmax= 10, 
                                                start_structure= None
                                                )
        else:
            self.atoms, __ = ali.align_clusters_to_references_parallel(
                                                atoms, 
                                                references,
                                                alignment
                                                )
        
        self.distances = met.get_distances(self.atoms)
        
    def projection(self, umap_params_= {}, cluster_params_= {}):
        
        ### Projection ###
        self.umap_params = {
                'n_neighbors' : 5,
                'min_dist' : 0.0,
                'n_components' : 2,
                'random_state' : 0
            }
        self.umap_params.update(**umap_params_)
        
        projector = umap.UMAP(
                    **self.umap_params
                    )
        
        u = projector.fit_transform(self.distances)
        
        ### Cluster ###
        self.cluster_params = {
                'min_cluster_size' : 10,
                'min_samples' : 7,
                'allow_single_cluster' : True,
                'cluster_selection_epsilon' : 0.5
            }
        self.cluster_params.update(**cluster_params_)
        
        clusterer = hdbscan.HDBSCAN(**self.cluster_params).fit(u)
        self.labels= clusterer.labels_
        
        atomReader.show_projections(u, clusterer)

        
    def show_projections(u, clusterer):
        palette = sns.color_palette('Paired', 12)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
        
        cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)])
        
        ### Plot Projections ###
        plt.figure()
        for label in np.unique(clusterer.labels_):
            mask = clusterer.labels_ == label
            plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = label)

        plt.title('UMAP Projection')
        plt.xlabel('U1')
        plt.ylabel('U2')
        plt.legend()
        
    def get_cluster_distances(self, cl):
        return self.distances[self.labels==cl]

#%% spectraReader

class spectraReader():
    def __init__(self,pkl, labels= None):
        with open(pkl, "rb") as fin:
                    enerlist, intenlist, ovlps = dill.load(fin)
        
        
        self.fit(
            np.array(enerlist)[labels],
            (np.array(intenlist)**2).mean(axis=1)[labels],
            np.array(ovlps)[labels]
            )
        
    
    def fit(self, energy, intensity, ovlps):
        self.energies = energy
        self.intensities = intensity
        self.ovlps = ovlps
        
    def projection(self, umap_params_= {}, cluster_params_= {}, lastpk = None):
        # Trim Spectra up to lastpk
        if lastpk == None:
            lastpk = self.energies.shape[1]
        
        
        # IDshape is the number of overlap basis orbitals
        IDshape = self.ovlps.shape[2]
        ovlps = self.ovlps[:,:lastpk,:].reshape(-1,IDshape)
        
        
        ## Scaling
        scaler = StandardScaler()
        ovlps_scaled = scaler.fit_transform(ovlps)
        
        
        ### UMAP ###
        self.umap_params = {
                'n_neighbors' : 20,
                'min_dist' : 0.0,
                'n_components' : 2,
                'random_state' : 15
            }
        
        self.umap_params.update(**umap_params_)
        proj = umap.UMAP(**self.umap_params)
        self.ovlp_umap = proj.fit_transform(ovlps_scaled)
        
        
        ### Cluster ###
        self.cluster_params= {
                'min_cluster_size' : 10,
                'min_samples' : 7,
                'allow_single_cluster' : True,
                'cluster_selection_epsilon' : 0.3
            }
        self.cluster_params.update(**cluster_params_)
        
        clusterer = hdbscan.HDBSCAN(**self.cluster_params).fit(self.ovlp_umap)
        
        self.labels = clusterer.labels_
        self.label_names= np.unique(self.labels)
        
        ## remove outliers
        self.label_names = self.label_names[self.label_names != -1]


        ### Extraction ###
        self.__clusters = []
        
        en = self.energies[:,:lastpk]
        inten = self.intensities[:,:lastpk]
        
        # Outliers
        outlier_idx = self.labels == -1
        out_ei_idx = outlier_idx.reshape(en.shape)
        self.outliers = (ovlps[outlier_idx], en[out_ei_idx], inten[out_ei_idx])
        
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
        self.show_ovlp_projections()
        self.show_spectral_clusters()
            
        return clusterer.labels_.reshape(-1,lastpk)
                
                  
    def show_ovlp_projections(self):
        palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(self.labels)))
        
        plt.figure()
        
        outlier_idx = self.labels == -1
        plt.scatter(
            self.ovlp_umap[outlier_idx,0],
            self.ovlp_umap[outlier_idx,1],
            c= (0.5,0.5,0.5),
            label= "Outliers",
            s=5
            )
        for label in self.label_names:
            labeled_idx = self.labels==label
            
            plt.scatter(
                        self.ovlp_umap[labeled_idx,0],
                        self.ovlp_umap[labeled_idx,1],
                        c= palette[label],
                        label= label,
                        s= 5)
            
        label_count = len(self.label_names)
        plt.title(f'Number of clusters: {label_count}')
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
                    inten,
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
    
        
        
        
        
        
        
        
        
        
        
        
        