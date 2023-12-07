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
from sklearn.utils import shuffle

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
    '''
    Estimator/Generator fit on structures to target XAS and ground state energy
    '''
    def __init__(self,models, e_cv= {}, i_cv= {}):
        '''
        Parameters
        ----------
        e_cv : dict, optional
            energy cross validation parameters to be fed to GridSearchCV. The default is {}.
        i_cv : dict, optional
            intensity cross validation parameters to be fed to GridSearchCV. The default is {}.
        '''
        self.atoms = info_struct()
        self.XAS = info_struct()
        self.set_cv_params(models, e_cv, i_cv)
        self.model_tree = TreeNode('Root')
    
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
    
    def fit(self,atoms_,energies_,intensities_,ovlps_, atom_ref=None):
        '''
        Assign reference data to model and fit estimators to structural and spectral data.

        Parameters
        ----------
        atoms_ : list (n_samples)
            list of atoms objects.
        energies_ : array (n_samples,n_transitions)
            array of transition energies from XAS.
        intensities_ : array (n_samples,n_transitions)
            array of transition intensities from XAS.
        ovlps_ : array (n_samples,n_transitions,n_VVO)
            VVO projection of each transition of each spectral sample.
        atom_ref : list
            reference structures to align atoms to. The default is None.
        '''
        self.set_atoms(atoms_,ref= atom_ref)
        self.set_XAS(energies_, intensities_, ovlps_)
        
        self.fit_atoms_projection()
        self.fit_XAS_projection()
        
        self.fit_estimators()
        
    def fit_estimators(self):
        '''
        Fit model tree to set atomic/spectral data and their projections.

        Returns
        -------
        None.

        '''
        # Scale inputs
        X_all = self.atoms.sep_scaler.transform(self.atoms.seperations)
        # Loop Through Atomic CLusters
        for atomic_cluster_label, transition_label_set in enumerate(self.XAS.labels):
            
            atomic_cluster_mask = self.atomic_cluster(atomic_cluster_label)
            
            # Unique non-outlier labels
            valid_labels = set(transition_label_set.reshape(-1))
            valid_labels.discard(-1)
            
            ## The most common frequency of each label in each structure
            standard_count = standardCount(transition_label_set,valid_labels)
            
            
            atomic_cluster_node = TreeNode(atomic_cluster_label)
            
            # Loop Through Transition State Clusters
            for transition_label in valid_labels:
                target_peaks = transition_label_set == transition_label
                
                ## Final extraction of features and targets
                valid_samples = np.sum(target_peaks, axis= 1) == standard_count[transition_label]
                X = X_all[atomic_cluster_mask][valid_samples]
                
                # Mask of relevant peaks
                t_mask = np.array([valid_samples]).T*target_peaks
                
                y = {
                    'energy' : self.XAS.energies[atomic_cluster_mask][t_mask].reshape((-1,standard_count[transition_label])),
                    'intensity' : self.XAS.intensities[atomic_cluster_mask][t_mask].reshape((-1,standard_count[transition_label]))
                }
                
                X, y['energy'], y['intensity'] = shuffle(X,y['energy'],y['intensity'])
                
                leaf_node = LeafEstimator((atomic_cluster_label,transition_label))
                leaf_node.fit(
                        X= X,
                        y= y, 
                        model_params= self.model_params, 
                        CV_params= self.CV_params
                        )
                
                atomic_cluster_node.add_child(leaf_node)
                
            self.model_tree.add_child(atomic_cluster_node)
                
    def predict(self, atoms_):
        '''
        Predict XAS of input atoms

        Parameters
        ----------
        atoms_ : list[Atoms]
            list of atoms to predict XAS of.

        Returns
        -------
        energy_result : Dataframe (n_samples,n_transitions)
            Dataframe of transition energies.
        intensity_result : Dataframe (n_samples,n_transitions)
            Dataframe of transition intensities.

        '''
        
        atoms = copy.deepcopy(atoms_)
        
        atoms = Trident.align_atoms(atoms, self.atoms.ref)
        
        X = met.get_distances(atoms)
        
        u = self.atoms.umap.transform(X)
        molecular_labels, _ = hdbscan.approximate_predict(self.atoms.hdbscan, u)
        
        # Intercept
        X = np.concatenate( (X, np.ones((X.shape[0],1))), axis= 1)
        X_df = pd.DataFrame(X)
        
        energies_pred, intensities_pred = self.model_tree.predict(X_df, molecular_labels)
        
        return energies_pred, intensities_pred
        
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
        '''
        Establish model references to atom training set

        Parameters
        ----------
        atoms_ : list[Atoms]
            List of atom set to train model on.
        ref : list[Atoms], optional
            If not None atoms are aligned to the structures in ref. The default is None.

        Returns
        -------
        None.

        '''
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
        '''
        Establish projection and clustering of atomic structure training set

        Parameters
        ----------
        umap_params_ : dict, optional
            Desired parameters of the umap model. The default is {}.
        cluster_params_ : dict, optional
            Desired parameters of the HDBSCAN model. The default is {}.

        Returns
        -------
        None.

        '''
        assert hasattr(self.atoms, 'atoms'), "Atoms must be set prior to calling fit_atoms_projection"
        
        ### Projection ###
        umap_params = {
                'n_neighbors' : 5,
                'min_dist' : 0.0,
                'n_components' : 2,
                'random_state' : 3
            } | umap_params_
        # umap_params.update(**umap_params_)
        
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
            } | cluster_params_
        # cluster_params.update(**cluster_params_)
        
        atom_cluster_model = hdbscan.HDBSCAN(**cluster_params).fit(u)
        self.atoms.hdbscan = atom_cluster_model
        
        self.atoms.labels = atom_cluster_model.labels_
        
    def show_atom_projection(self):
        '''
        Visualize atomic structure projections

        Returns
        -------
        None.

        '''
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
        '''
        Establish model references to XAS training set

        Parameters
        ----------
        energies_ : array (n_samples,n_transitions)
            Transitions energy training data.
        intensities_ : array (n_samples,n_transitions)
            Transitions intensity training data.
        ovlps_ : array (n_samples,n_transitions,n_VVO)
            Projections of each transition onto VVO basis.

        Returns
        -------
        None.

        '''
        ## Set Internal Reference to XAS ##
        self.XAS.energies = copy.deepcopy(energies_)
        self.XAS.intensities = copy.deepcopy(intensities_)
        self.XAS.ovlps = copy.deepcopy(ovlps_)
        
    def fit_XAS_projection(self, umap_params_ = {}, cluster_params_ = {}):
        '''
        Establish projection and clustering of XAS training set

        Parameters
        ----------
        umap_params_ : dict, optional
            Desired parameters of the umap model. The default is {}.
        cluster_params_ : dict, optional
            Desired parameters of the HDBSCAN model. The default is {}.

        Returns
        -------
        None.

        '''
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
        '''
        Visualize XAS projection and clustering

        Parameters
        ----------
        struct : int, optional
            If None displays XAS projections for all atomic clusters, otherwise displays that for only struct atomic cluster. The default is None.

        Returns
        -------
        None.

        '''
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
    def __init__(self, ID, data= None):
        self.ID = ID
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        
    def predict(self, X, atomic_labels = None):
        
        # If we are the root we need to seperate the atomic clusters
        if self.ID == 'Root':
            XAS_enerlist, XAS_intenlist = [], []
            for branch in self.children:
                cluster_mask = atomic_labels == branch.ID
                
                # If one cluster is not present in the prediction set continue
                if np.all(cluster_mask == False):
                    continue
                
                XAS_en, XAS_inten = branch.predict(X[cluster_mask])
                XAS_enerlist.append(XAS_en)
                XAS_intenlist.append(XAS_inten)
            
            # We recieve entire spectra so we just need to put them into the same df
            XAS_energies = pd.concat(XAS_enerlist,axis= 0, keys= range(len(XAS_enerlist)), names= ["Molecular Cluster", "Sample"]).sort_values(by="Sample")
            XAS_intensities = pd.concat(XAS_intenlist, axis= 0, keys= range(len(XAS_enerlist)), names= ["Molecular Cluster", "Sample"]).sort_values(by="Sample")
            
            return XAS_energies, XAS_intensities
        else: # If we are an atomic branch we simply need to predict each cluster and group them to form a spectra
        
            enerlist, intenlist = [], []
            for estimator in self.children:
                assert estimator.ID[0] == self.ID
                energy, intensity = estimator.predict(X) # Predict transitions for each cluster
                
                cluster_energies = pd.DataFrame(energy, index= X.index, columns= None)
                cluster_intensities = pd.DataFrame(intensity, index= X.index, columns= None)
                
                enerlist.append( cluster_energies)
                intenlist.append( cluster_intensities)
            
            ## Concatenate clusters togeter and reset headers
            energies = pd.concat(enerlist,axis= 1)
            energies.columns = range(len(energies.columns))
            intensities = pd.concat(intenlist,axis= 1)
            intensities.columns = range(len(intensities.columns))
            return energies, intensities            ## Return XAS
                
            
    def __iter__(self):
        self.current = -1
        return self
    
    def __next__(self):
        self.current += 1
        
        if self.current >= len(self.children):
            raise StopIteration
            
        return self.children[self.current]
                
    def __getitem__(self,index):
        return self.children[index]
        
class LeafEstimator():
    
    def __init__(self,ID):
        self.ID = ID
        self.pipeline = {}
        self.target_scaler = {}
        
    def fit(self, X, y, model_params, CV_params):
        '''
        Fit cluster specific regression model

        Parameters
        ----------
        X : Structural data (n_samples,n_features)
            Input to be regressed.
        y : dict {'energy' 'intensity'}
            dictionary of target (n_samples, n_transitions) training values.
        model_params : dict {'energy' 'intensity'}
            Dictionary of models to be used.
        CV_params : dict
            CV range dictionary.

        Returns
        -------
        None.

        '''
        
        X = np.concatenate( (X, np.ones((X.shape[0],1))), axis= 1)
        structure_scaler = StandardScaler().fit(X)
        X_scaled = structure_scaler.transform(X)
        
        for target, values in y.items():
            
            self.target_scaler[target] = StandardScaler().fit(values)
            values = self.target_scaler[target].transform(values)
            
            # Cross Validation
            Cross_validator = GridSearchCV(
                model_params[target](),
                param_grid= CV_params[target],
            )
            Cross_validator.fit(X_scaled,values)
            
            ## Verbage ##
            print(f"Cluster {self.ID} {target} Estimator R2 score: {Cross_validator.best_score_:.3f}")
            print(f"Cluster {self.ID} {target} Estimator params: {Cross_validator.best_params_}", end= '\n\n')
            
        
            pipe = Pipeline([
                        ('Scaler',structure_scaler),
                        ('Estimator', Cross_validator.best_estimator_)
                        ])

            
            self.pipeline[target] = pipe
        
    def predict(self, X):
        y = {}
        for (target, pipe), scaler in zip(self.pipeline.items(),self.target_scaler.values()):
            y_scaled = pipe.predict(X)
            y[target] = scaler.inverse_transform(y_scaled)
        return y.values()
        
        
class info_struct():
    pass        
        

#%% Helper Methods

def standardCount(label_arr, labels = None):
    '''
    Most Common Number of occurances of each label in each row of label_arr

    Parameters
    ----------
    labels : array (n_labels)
        List of target labels. If None assumes all unique elements of label_arr
    label_arr : array (n_rows, n_col)
        reference array to find most common elements.

    Returns
    -------
    List of most common frequency of each label (n_labels).

    '''
    
    ## How many occurances of a label in each structure
    label_count = np.array([np.sum(label_arr == label, axis= 1) for label in labels]).T
    
    ## The standard number of occurances of each cluster
    unique, counts = np.unique(label_count, axis= 0, return_counts= True)
    standard_count = unique[counts.argmax()]
    
    return standard_count
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        