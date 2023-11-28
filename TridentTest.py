#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:51:16 2023

@author: jdcooper
"""
#%% IMPORTS

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

import copy
import sea_urchin.alignement.align as ali
import sea_urchin.clustering.metrics as met

import umap
import hdbscan

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#%% MAIN

class Trident():
    def __init__(self, e_cv= {}, i_cv= {}):
        self.atoms = info_struct()
        self.XAS = info_struct()
        self.set_cv_params(e_cv,i_cv)
        self.model_tree = TreeNode()
    
    def set_cv_params(self,models={},e_cv={},i_cv={}):
        self.model_params = {
                'energy' : Ridge,
                'intensity' : KernelRidge
            }
        self.model_params.update(models)
        e_cv_params = {
                'alpha' : np.logspace(0,1,5),
                # 'gamma' : np.logspace(-1,4)
            }
        e_cv_params.update(e_cv)
        i_cv_params = {
                'alpha' : np.logspace(4,9,5),
                # 'gamma' : np.logspace(-6,-1)
            }
        i_cv_params.update(i_cv)
        
        self.CV_params = {
                'energy' : e_cv_params,
                'intensity' : i_cv_params
            }
    
    def fit(self,atoms_,energies_,intensities_,ovlps_, atom_ref=None):
        self.set_atoms(atoms_,ref= atom_ref)
        self.set_XAS(energies_, intensities_, ovlps_)
        
        self.fit_atoms_projection()
        self.fit_XAS_projection()
        
        self.fit_estimators()
        
    def fit_estimators(self):
        X = self.atoms.sep_scaler.transform(self.atoms.seperations)
        
        # Intercept
        X = np.concatenate( (X, np.ones((X.shape[0],1))), axis= 1)
        
        # Loop Through Atomic CLusters
        for atomic_cluster_label, transition_label_set in enumerate(self.XAS.labels):
        # for atomic_cluster_label, transition_label_set in [(1,self.XAS.labels[1])]:
            
            valid_labels = set(transition_label_set.reshape(-1))
            valid_labels.discard(-1)
            
            ## How many occurances of a label in each structure
            label_count = np.array([np.sum(transition_label_set == label, axis= 1) for label in valid_labels]).T
            
            ## The standard number of occurances of each cluster
            unique, counts = np.unique(label_count, axis= 0, return_counts= True)
            standard_count = unique[counts.argmax()]            
            
            
            atomic_cluster = self.atomic_cluster(atomic_cluster_label)
            X_atomic_cl = X[atomic_cluster]
            y_atomic_cl= {
                'energy' : self.XAS.energies[atomic_cluster],
                'intensity' : self.XAS.intensities[atomic_cluster]
            }
            
            branch_node = TreeNode(atomic_cluster_label)
            
            # Loop Through Transition State Clusters
            for transition_label in valid_labels:
                
                leaf_node = LeafEstimator((atomic_cluster_label,transition_label))
                
                target_peaks = transition_label_set == transition_label
                ## Final extraction of features and targets
                valid_samples = np.sum(target_peaks, axis= 1) == standard_count[transition_label]
                X_ts_cl = X_atomic_cl[valid_samples]
                
                # Shuffle
                permutation = np.random.permutation(X_ts_cl.shape[0])
                X_feature = X_ts_cl[permutation]
                
                # Mask of relevant peaks
                t_mask = np.array([valid_samples]).T*target_peaks

                for target in y_atomic_cl:
                    y = y_atomic_cl[target][t_mask].reshape((-1,standard_count[transition_label]))
                    
                    # Target Scalers
                    leaf_node.scaler[target] = StandardScaler().fit(y)
                
                    y = leaf_node.scaler[target].transform(y)
                
                    # Shuffle
                    y = y[permutation]
                    
                    # Cross Validation
                    Cross_validator = GridSearchCV(
                        self.model_params[target](),
                        param_grid=self.CV_params[target],
                    )
                    Cross_validator.fit(X_feature,y)
                    ## Verbage ##
                    print(f"Cluster {atomic_cluster_label},{transition_label} {target} Estimator R2 score: {Cross_validator.best_score_:.3f}")
                    print(f"Cluster {atomic_cluster_label},{transition_label} {target} Estimator params: {Cross_validator.best_params_}", end= '\n\n')
                    estimator = self.model_params[target](**Cross_validator.best_params_)
                    
                    pipe = Pipeline([
                                ('Scaler',StandardScaler()),
                                ('Estimator', estimator)
                                ])
                    pipe.fit(X_feature,y)
                    leaf_node.pipeline[target] = pipe
                            
                
                branch_node.add_child(leaf_node)
                
            self.model_tree.add_child(branch_node)
                
    def predict(self, atoms_):
        
        mutable_atoms = copy.deepcopy(atoms_)
        
        mutable_atoms = Trident.align_atoms(mutable_atoms, self.atoms.ref)
        
        X = met.get_distances(mutable_atoms)
        
        u = self.atoms.umap.transform(X)
        molecular_labels, strengths = hdbscan.approximate_predict(self.atoms.hdbscan, u)
        molecular_labels_set = set(molecular_labels.reshape(-1))
        molecular_labels_set.discard(-1)
        
        # Intercept
        X = np.concatenate( (X, np.ones((X.shape[0],1))), axis= 1)
        X_df = pd.DataFrame(X)
        X_df['Atomic Label'] = molecular_labels
        

        energy_df_list = []
        intensity_df_list = []
        
        peak_count = 0
        for mole_label in molecular_labels_set:
            X_molecular_df = X_df[X_df['Atomic Label'] == mole_label].iloc[:,:-1]
            
            energy_df = pd.DataFrame(index=X_molecular_df.index)
            intensity_df = pd.DataFrame(index=X_molecular_df.index)
            
            for estimator in self.model_tree[mole_label]:
                Y_cl = estimator.predict(X_molecular_df)
                
                peaks_per_cluster = Y_cl['energy'].shape[1]
                
                for peak_idx in range(peaks_per_cluster):
                    energy_df[peak_count + peak_idx] = Y_cl['energy'][:,peak_idx]
                    intensity_df[peak_count + peak_idx] = Y_cl['intensity'][:,peak_idx]
                    
                peak_count += peaks_per_cluster    
            
            energy_df_list.append(energy_df)
            intensity_df_list.append(intensity_df)
            
        energy_result = pd.concat(energy_df_list)
        intensity_result = pd.concat(intensity_df_list)
        
        return energy_result, intensity_result
        
    def align_atoms(atoms, ref):
        
        ## Align ##
        alignment = {
                "type"      : "fastoverlap",
                "permute" : "elements",
                "inversion" : False
                }
        
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
        
    def set_atoms(self, atoms_, ref=None):
        ## Set Internal Reference to atoms ##
        mutable_atoms = copy.deepcopy(atoms_)
        
        ## Align ##
        mutable_atoms = Trident.align_atoms(mutable_atoms, ref)
         
        ## Set Internal Reference To Alligned Atoms and Their Atomic Seperations ##
        self.atoms.atoms = mutable_atoms
        self.atoms.ref = ref
        self.atoms.seperations = met.get_distances(mutable_atoms)
        self.atoms.sep_scaler = StandardScaler().fit(self.atoms.seperations)
        
    def fit_atoms_projection(self, umap_params_= {}, cluster_params_= {}):
        assert hasattr(self.atoms, 'atoms'), "Atoms must be set prior to calling fit_atoms_projection"
        ### Projection ###
        umap_params = {
                'n_neighbors' : 5,
                'min_dist' : 0.0,
                'n_components' : 2,
                'random_state' : 3
            }
        umap_params.update(**umap_params_)
        
        atom_umap_model = umap.UMAP(**umap_params).fit(self.atoms.seperations)
        self.atoms.umap = atom_umap_model
        
        ### Cluster ###
        
        u = atom_umap_model.transform(self.atoms.seperations)
        
        cluster_params = {
                'min_cluster_size' : 100,
                # 'min_samples' : 7,
                'allow_single_cluster' : True,
                # 'cluster_selection_epsilon' : 0.5,
                'prediction_data' : True
            }
        cluster_params.update(**cluster_params_)
        
        atom_cluster_model = hdbscan.HDBSCAN(**cluster_params).fit(u)
        self.atoms.hdbscan = atom_cluster_model
        
        self.atoms.labels = atom_cluster_model.labels_
        
    def show_atom_projection(self):
        assert hasattr(self.atoms, 'labels'), "Model must first be fit to the atoms"
        ### Cluster Colors ###
        true_label_count = len(np.unique(self.atoms.labels[self.atoms.labels != -1]))
        palette = sns.color_palette('Paired', true_label_count)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in self.atoms.labels]
        
        cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, self.atoms.hdbscan.probabilities_)])
        
        ### Plot Projections ###
        
        u = self.atoms.umap.transform(self.atoms.seperations)
        
        plt.figure()
        for atom_label in np.unique(self.atoms.labels):
            mask = self.atoms.labels == atom_label
            plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)

        plt.title('Atom UMAP Projection')
        plt.xlabel('U1')
        plt.ylabel('U2')
        plt.legend()
        
    def set_XAS(self,energies_,intensities_,ovlps_):
        ## Set Internal Reference to XAS ##
        self.XAS.energies = copy.deepcopy(energies_)
        self.XAS.intensities = copy.deepcopy(intensities_)
        self.XAS.ovlps = copy.deepcopy(ovlps_)
        
    def fit_XAS_projection(self, umap_params_ = {}, cluster_params_ = {}):
        
        ## Arrange transition ovlps ##
        LIVO_count = self.XAS.ovlps.shape[2]
        ordered_ovlps = self.XAS.ovlps.reshape((-1,LIVO_count))
       
        ### Projection and Cluster ###
        umap_params = {
                'n_neighbors' : 20,
                'min_dist' : 0.0,
                'n_components' : 2,
                'random_state' : 15
            }
        umap_params.update(**umap_params_)
        
        cluster_params= {
                'min_cluster_size' : 10,
                'min_samples' : 7,
                'allow_single_cluster' : True,
                'cluster_selection_epsilon' : 0.3
            }
        cluster_params.update(**cluster_params_)
        
        # Intuitively standardize data
        self.XAS.ovlp_scaler = StandardScaler().fit(ordered_ovlps)
        
        self.XAS.umap = []
        self.XAS.hdbscan = []
        self.XAS.labels = []
        
        transition_count = self.XAS.ovlps.shape[1]
        atom_cluster_labels = np.unique(self.atoms.labels[self.atoms.labels != -1])
        for atom_label in atom_cluster_labels:
            labeled_ovlps = self.XAS.ovlps[self.atomic_cluster(atom_label)].reshape((-1,LIVO_count))
            scaled_labeled_ovlps = self.XAS.ovlp_scaler.transform(labeled_ovlps)
            
            umap_model = umap.UMAP(**umap_params).fit(scaled_labeled_ovlps)
            u = umap_model.transform(scaled_labeled_ovlps)
            hdbscan_model = hdbscan.HDBSCAN(**cluster_params).fit(u)
            
            
            self.XAS.umap.append(umap_model)
            self.XAS.hdbscan.append(hdbscan_model)
            self.XAS.labels.append(hdbscan_model.labels_.reshape((-1,transition_count)))
        
    def show_XAS_projection(self, struct = None):
        
        # Plot all 
        if struct == None:
            for atom_label in np.unique(self.atoms.labels[self.atoms.labels != -1]):
                self.show_XAS_projection(atom_label)
            return
        
        cluster_XAS_labels = self.XAS.labels[struct].reshape(-1,)
        
        # Palette
        true_label_count = len(np.unique(cluster_XAS_labels[cluster_XAS_labels != -1]))
        palette = sns.color_palette('Paired', true_label_count)
        
        cluster_colors = [palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in cluster_XAS_labels]
        
        
        cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, self.XAS.hdbscan[struct].probabilities_)])
        
        ### Plot Projections ###
        
        # Project
        LIVO_count = self.XAS.ovlps.shape[2]
        labeled_ovlps = self.XAS.ovlps[self.atomic_cluster(struct)].reshape((-1,LIVO_count))
        scaled_labeled_ovlps = self.XAS.ovlp_scaler.transform(labeled_ovlps)
        
        u = self.XAS.umap[struct].transform(scaled_labeled_ovlps)
        
        # Plot
        plt.figure()
        for atom_label in np.unique(cluster_XAS_labels):
            mask = cluster_XAS_labels == atom_label
            plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)

        plt.title(f'Atomic Structure {struct} LIVO Overlap UMAP Projection')
        plt.xlabel('U1')
        plt.ylabel('U2')
        plt.legend()
        
    #%% Accessors
    
    def atomic_cluster(self,label):
        return self.atoms.labels == label
        
    #%% MISC
        
        
class TreeNode:
    def __init__(self, data= None):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
            
    def __iter__(self):
        for child in self.children:
            if type(child) == TreeNode:
                for leaf in child.children:
                    yield leaf
            else:
                yield child
                
    def __getitem__(self,index):
        return self.children[index]
        
        
class LeafEstimator():
    
    def __init__(self,ID):
        self.ID = ID
        self.pipeline = {}
        self.scaler = {}
        
    def predict(self, X):
        y = {}
        for (target, pipe), scaler in zip(self.pipeline.items(),self.scaler.values()):
            y_scaled = pipe.predict(X)
            y[target] = scaler.inverse_transform(y_scaled)
            
        return y
        
        
        
        
        
        
        
class info_struct():
    pass        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        