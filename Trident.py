#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:54:58 2023

@author: jdcooper
"""
#%% Imports

# from benzene import PAS, energies, intensities, GS_energies

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error

import copy

import sea_urchin.clustering.metrics as met
from scipy.constants import physical_constants

#%% Main

class Trident():
    '''
    Estimator/Generator fit on structures to target XAS and ground state energy
    '''
    def __init__(self,params={},e_cv_params_={},i_cv_params_={}):
        '''
        Estimator/Generator fit on structures to target XAS and ground state energy
        
        Parameters\t
        ‾‾‾‾‾‾‾‾\t
        params: dict of model to be used, default= {'energies':'linear','intensities':'kernel','GS_energy':'kernel'}\t
        e_cv_params: dict of cross-validation parameters ranges for energies estimator\t
        i_cv_params: dict of cross-validation parameters ranges for intensities estimator\t
        '''
        self.params = {'energies': 'linear','intensities':'kernel','GS_energy':'kernel'}
        self.params.update(params)
        
        
        
        e_cv_params = {
                'alpha' : np.logspace(0,1),
                # 'gamma' : np.logspace(-1,4)
            }
        e_cv_params.update(e_cv_params_)
        i_cv_params = {
                'alpha' : np.logspace(4,9),
                'gamma' : np.logspace(-6,-1)
            }
        i_cv_params.update(i_cv_params_)
        
        self.CV_params = {
                'energies' : e_cv_params,
                'intensities' : i_cv_params
            }
        
        self.pipe= {}
        
    def prepare_data(x,y):
        
        y = copy.deepcopy(y)
        X = np.concatenate( (copy.deepcopy(x), np.ones((x.shape[0],1))), axis= 1)
        
        ##Shuffle
        permutation = np.random.permutation(x.shape[0])
        
        X = X[permutation]
        y = y[permutation]
        
        return X, y
    
    def fit_single(self, X_, target_, label):
        
        X, target = Trident.prepare_data(X_, target_)
        
        ## Scaler ##
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scaler_target = StandardScaler()
        target = scaler_target.fit_transform(target)
        
        ## Cross-Validation ##
        
        match self.params[label]:
            case 'Ridge':
                model = Ridge
            case 'KernelRidge':
                model = KernelRidge

        Cross_validator = GridSearchCV(
            model(),
            param_grid=self.CV_params[label],
        )
        
        Cross_validator.fit(X_scaled,target)
        
        ## Verbage ##
        print(f"{label} Estimator R2 score: {Cross_validator.best_score_:.3f}")
        print(f"{label} Estimator params: {Cross_validator.best_params_}", end= '\n\n')
        
        
        ## Assignment ##
        estimator = model(**Cross_validator.best_params_)
        
        self.pipe[label] = Pipeline([
                        ('Scaler',StandardScaler()),
                        ('Estimator', estimator)
                        ])
        
        self.pipe[label].fit(X,target)
        
        pass
    
    def fit(self, PAS_, energies_= None, intensities_= None, GS_energy_ = None):
        '''
        Fit Pariwise-atomic-seperations (PAS) to estimators of XAS spectra and grounds state energies
        
        Parameters\t
        ‾‾‾‾‾‾‾‾\t
        PAS_: Array of PAS (n_samples,n_features)\t
        energies_: Array of XAS stick energy positions (n_samples, n_sticks)\t
        intensities_: Array of XAS stick Intensities (n_samples, n_sticks)\t
        GS_energy_: total ground state energies in units of eV (n_samples,)\t
        '''
        
        if np.all(energies_ != None):
            self.fit_single(PAS_, energies_, 'energies')
        if np.all(intensities_ != None):
            self.fit_single(PAS_, intensities_, 'intensities')
        if np.all(GS_energy_ != None):
            self.fit_single(PAS_, GS_energy_, 'GS_energy')
            
            
    def score(self, PAS_test, data_test, metric = 'R2', verbose= False):
        '''
        Return dict of estimator scores.
        
        Parameters\t
        ‾‾‾‾‾‾‾‾\t
        PAS_test: Array of PAS (n_samples,n_features)\t
        data_test: dict of target values {'energies':_, 'intensities':_,'GS_energy':_}\t
        metric: metric of score {'R2','MSE'}, default= 'R2'\t
        verbose: Print scores, default= False\t
        '''
        
        pred = self.predict(PAS_test)
        
        match metric:
            case 'MSE':
                score = {
                        'energies' : mean_squared_error(y_true= data_test['energies'], y_pred= pred['energies'], multioutput= 'raw_values'),
                        'intensities' : mean_squared_error(y_true= data_test['intensities'], y_pred= pred['intensities'], multioutput= 'raw_values'),
                        'GS_energy' : mean_squared_error(y_true= data_test['GS_energy'], y_pred= pred['GS_energy'], multioutput= 'raw_values') if  self.hs_GS else None
                    }
            case 'R2':
                score = {
                        'energies' : r2_score(y_true= data_test['energies'], y_pred= pred['energies'], multioutput= 'raw_values'),
                        'intensities' : r2_score(y_true= data_test['intensities'], y_pred= pred['intensities'], multioutput= 'raw_values'),
                        'GS_energy' : r2_score(y_true= data_test['GS_energy'], y_pred= pred['GS_energy'], multioutput= 'raw_values') if  self.hs_GS else None
                    }
        
        if verbose:
            print('Energies {} score : {:.3f}'.format(metric, np.mean(score['energies'])))
            print('Intensities {} score : {:.3f}'.format(metric, np.mean(score['intensities'])))
            print('GS Energy {} score : {:.3f}'.format(metric, score['GS_energy']))
        
        return score
        


    def predict(self,PAS_):
        '''
        Makes Predictions of XAS and Ground State Energy
        
        Parameters\t
        ‾‾‾‾‾‾‾‾\t
        PALS_ : Pairwise atomic separations of molecular structure (n_samples,n_features)
        
        Return\t
        ‾‾‾‾‾\t
        return['energies'] : XAS stick energies predictions (n_samples, n_sticks)
        return['intensities'] : XAS stick intensities predicitions (n_samples, n_sticks)
        return['GS_energy'] : Ground state energy predictions (n_samples,)
        '''
        X = np.concatenate( (copy.deepcopy(PAS_), np.ones((PAS_.shape[0],1))), axis= 1)
        
        prediction = {
            'energies' : self.pipe['energies'].predict(X),
            'intensities' : self.pipe['intensities'].predict(X),
            'GS_energy' : self.pipe['GS_energy'].predict(X) if self.hs_GS else None
            }
        
        return prediction



    def weigh_spectra(data_,T= 293.15):
        ## Weighting
        k = physical_constants['Boltzmann constant in eV/K'][0]
        
        delta_GS_energy = data_['GS_energy'] - np.min(data_['GS_energy'])
        weight = np.exp(-delta_GS_energy/(k*T))
        weighted_intensities= data_['intensities'] * weight / np.sum(weight)
        
        return weighted_intensities        
        

    def shake(self, atom_, T = 293.15, stdev= 0.03, size= 10000):
        '''
        Rattle around an atom and attribute weights by their predicted GS energy within a 
        boltzmann distribution to the XAS of resulting structures to be added to form a single convolutional spectrum
        
        Parameters\t
        ‾‾‾‾‾‾‾‾\t
        atom_ : Central atom for structures to be sampled around\t
        T : Temperature, used for calculation of boltzmann weight\t
        stdev : stdev of rattle\t
        size : Number of structures to sample\t
        
        Return\t
        ‾‾‾‾‾\t
        energies : Filtered XAS stick energy\t
        weighted_intensities : Filtered XAS stick intensity\t
        atom_batch : list of sampled atoms objects\t
        prediction: Unfiltered predictions of atom_batch\t
        '''
        atom = copy.deepcopy(atom_)
        
        ## Generate List of Rattled atoms
        atom_batch = []
        for i in range(size):
            temp_atom = copy.deepcopy(atom)
            temp_atom.rattle(stdev= stdev, seed= i)
            atom_batch.append(temp_atom)
            
        ## Get PAS from batch
        PAS = met.get_distances(atom_batch)
        
        prediction = self.predict(PAS)
        
        ## Weighing
        weighted_intensities = Trident.weigh_spectra(prediction,T)
        
        ## Reshaping
        weighted_intensities = weighted_intensities.reshape(1,-1)[0]
        energies = prediction['energies'].reshape(1,-1)[0]
        
        ## Filtering
        valid_mask = weighted_intensities >= 0
        weighted_intensities = weighted_intensities[valid_mask]
        energies= energies[valid_mask]
        
        print('Confidence: {:.2f}'.format(np.count_nonzero(valid_mask)/len(valid_mask)))
        
        return energies, weighted_intensities, atom_batch, prediction
            