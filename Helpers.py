#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:40:59 2023

@author: jdcooper
"""
#%% Imports
import numpy as np

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
        