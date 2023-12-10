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
# from sklearn.utils import shuffle

import copy
import sea_urchin.alignement.align as ali
import sea_urchin.clustering.metrics as met

import umap
import hdbscan

# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

#%% MAIN

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
    
    def init_tree(self, atoms_, ref):
        ## Set Internal Reference to atoms ##
        mutable_atoms = copy.deepcopy(atoms_)
        self.ref = ref
        
        ## Align ##
        mutable_atoms = Trident.align_atoms(mutable_atoms, ref)
        seperations = met.get_distances(mutable_atoms)
        
        self.model_tree = RootNode(seperations)
        self.model_tree.setModelParams(estimator_params= self.model_params, cv_params= self.CV_params)
        
    def predict(self, atoms_):
        mutable_atoms = copy.deepcopy(atoms_)
        
        mutable_atoms = Trident.align_atoms(mutable_atoms, self.ref)            ### CHECK IF THERE IS AN ISSUE WITH ALIGNING TEST ATOMS TO
        seperations = met.get_distances(mutable_atoms)                          ### TO REFERENCES DIRECTLY OR NEED TO BE ALIGNED WITH
                                                                                ### TRAINING ATOMS
        prediction_energies, prediction_intensities = self.model_tree.predict(seperations)
        
        return prediction_energies, prediction_intensities
        
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
    

    #%% Show
    # def show_atom_projection(self):
    #     '''
    #     Visualize atomic structure projections

    #     Returns
    #     -------
    #     None.

    #     '''
    #     assert hasattr(self.atoms, 'labels'), "Model must first be fit to the atoms"
    #     ### Cluster Colors ###
    #     true_label_count = len(np.unique(self.atoms.labels[self.atoms.labels != -1]))
    #     palette = sns.color_palette('Paired', true_label_count)
        
    #     cluster_colors = [palette[x] if x >= 0
    #                       else (0.5, 0.5, 0.5)
    #                       for x in self.atoms.labels]
        
    #     cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
    #                      zip(cluster_colors, self.atoms.hdbscan.probabilities_)])
        
    #     ### Plot Projections ###
        
    #     u = self.atoms.umap.transform(self.atoms.seperations)
        
    #     plt.figure()
    #     for atom_label in np.unique(self.atoms.labels):
    #         mask = self.atoms.labels == atom_label
    #         plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)

    #     plt.title('Atom UMAP Projection')
    #     plt.xlabel('U1')
    #     plt.ylabel('U2')
    #     plt.legend()
    # def show_XAS_projection(self, struct = None):
    #     '''
    #     Visualize XAS projection and clustering

    #     Parameters
    #     ----------
    #     struct : int, optional
    #         If None displays XAS projections for all atomic clusters, otherwise displays that for only struct atomic cluster. The default is None.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     # Plot all 
    #     if struct == None:
    #         for atom_label in np.unique(self.atoms.labels[self.atoms.labels != -1]):
    #             self.show_XAS_projection(atom_label)
    #         return
        
    #     cluster_XAS_labels = self.XAS.labels[struct].reshape(-1,)
        
    #     # Palette
    #     true_label_count = len(np.unique(cluster_XAS_labels[cluster_XAS_labels != -1]))
    #     palette = sns.color_palette('Paired', true_label_count)
        
    #     cluster_colors = [palette[x] if x >= 0
    #                       else (0.5, 0.5, 0.5)
    #                       for x in cluster_XAS_labels]
        
        
    #     cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
    #                      zip(cluster_colors, self.XAS.hdbscan[struct].probabilities_)])
        
    #     ### Plot Projections ###
        
    #     # Project
    #     LIVO_count = self.XAS.ovlps.shape[2]
    #     labeled_ovlps = self.XAS.ovlps[self.atomic_cluster(struct)].reshape((-1,LIVO_count))
    #     scaled_labeled_ovlps = self.XAS.ovlp_scaler.transform(labeled_ovlps)
        
    #     u = self.XAS.umap[struct].transform(scaled_labeled_ovlps)
        
    #     # Plot
    #     plt.figure()
    #     for atom_label in np.unique(cluster_XAS_labels):
    #         mask = cluster_XAS_labels == atom_label
    #         plt.scatter(u[mask,0], u[mask,1], c = cluster_member_colors[mask], alpha= 0.6, s= 12, label = atom_label)

    #     plt.title(f'Atomic Structure {struct} LIVO Overlap UMAP Projection')
    #     plt.xlabel('U1')
    #     plt.ylabel('U2')
    #     plt.legend()
        
#%% TreeNodeNew
class TreeNodeNew:
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
        self.projection_params = TreeNodeNew.projection_params | params
        self.projector = umap.UMAP(**self.projection_params).fit(data)
    
    def fit_cluster(self, data, params):
        self.cluster_params = TreeNodeNew.cluster_params | params
        
        u = self.projector.transform(data)
        
        self.clusterer = hdbscan.HDBSCAN(**self.cluster_params).fit(u)
    
    def __getitem__(self,index):
        assert index < len(self.children)
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
        
#%% RootNode
class RootNode(TreeNodeNew):
    def __init__(self, seperations, ID='Root'):
        self.seperations_scaler = StandardScaler().fit(seperations)
        self.__seperations = self.seperations_scaler.transform(seperations)
        self.__XAS_energies = None
        self.__XAS_intensities = None
        self.__XAS_ovlps = None
        TreeNodeNew.__init__(self, ID)
    
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
        TreeNodeNew.fit_projection(self, 
                                   data= self.__seperations,
                                   params= params)
    
    def fit_cluster(self, params= {}):
        TreeNodeNew.fit_cluster(self, 
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

#%% BranchNode

class BranchNode(TreeNodeNew):
    def fit(self):
        self.fit_projection()
        self.fit_cluster()
        self.partition()
        for estimator in self.children:
            estimator.fit()
    
    def fit_projection(self, params= {}):
        sample_count, transition_count, LIVO_count = self.XAS_ovlps.shape           #For Reformatting ovlps
        individual_ovlps = np.reshape(self.XAS_ovlps, ( sample_count * transition_count, LIVO_count )) # Reformated OVLPS
        
        TreeNodeNew.fit_projection(self, 
                                   data= individual_ovlps, 
                                   params= params)
    
    def fit_cluster(self, params= {}):        
        sample_count, transition_count, LIVO_count = self.XAS_ovlps.shape           #For Reformatting ovlps
        individual_ovlps = np.reshape(self.XAS_ovlps, ( sample_count * transition_count, LIVO_count ))
        
        TreeNodeNew.fit_cluster(self, 
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
        
    
    #%% Property Methods (Private Attributes)
    # Could be made virtual but I think it would make it harder to read
    @property
    def labels(self):
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
    if labels == None:
        labels = set(label_arr.reshape(-1))
        labels.discard(-1)
    ## How many occurances of a label in each structure
    label_count = np.array([np.sum(label_arr == label, axis= 1) for label in labels]).T
    
    ## The standard number of occurances of each cluster
    unique, counts = np.unique(label_count, axis= 0, return_counts= True)
    standard_count = unique[counts.argmax()]
    
    return standard_count
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        