#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:54:09 2019

@author: Yun

preprocess data
"""

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# use this when reading from csv
def process_csv(file_path):
    # read in weights
    weights = np.genfromtxt(file_path,delimiter=',', dtype=np.float64)
    size = weights.shape[0]

    # mean-center and normalize
    mean = np.mean(weights, axis=0)
    weights = weights-mean
    weights = preprocessing.normalize(weights)

    # perform density filtration with k=200, p=0.3
    k=200
    p=0.3
    top = int(size*p)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(weights)
    distances, indices = nbrs.kneighbors(weights)
    furthest = distances[:, k-1]
    ind_of_closest_dist = np.argpartition(furthest, top)[:top]
    #indices = np.unique(indices[ind_of_closest_dist][:,[0,k-1]].flatten('C'))
    #filtered_weights = weights[indices,:]
    filtered_weights = weights[ind_of_closest_dist,:]
    
    return filtered_weights

# use this when weights are given as a numpy array
def process_nparray(weights):
    size = weights.shape[0]

    # mean-center and normalize
    mean = np.mean(weights, axis=0)
    weights = weights-mean
    weights = preprocessing.normalize(weights)

    # perform density filtration with k=200, p=0.3
    k=200
    p=0.3
    top = int(size*p)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(weights)
    distances, indices = nbrs.kneighbors(weights)
    furthest = distances[:, k-1]
    ind_of_closest_dist = np.argpartition(furthest, top)[:top]
    #indices = np.unique(indices[ind_of_closest_dist][:,[0,k-1]].flatten('C'))
    #filtered_weights = weights[indices,:]
    filtered_weights = weights[ind_of_closest_dist,:]
    
    return filtered_weights

