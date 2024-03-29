#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:30:21 2019

@author: ynxysunny
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
from sklearn import neighbors
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams

import kmapper as km


num_classes = 10
num_models = 49
weight_dim = 9

# loading weights from weight files
def load_weights(model, num_models, weight_dim):
    all_weights = [] 
    for i in range(num_models): 
        model.load_weights('model%s_weights.h5' %i)
        weights = model.get_weights()
        conv1_weights = np.einsum('klij->ijkl', weights[0])
        conv1_weights = conv1_weights.reshape(-1, weight_dim)
        all_weights.append(conv1_weights)
    
    all_weights = np.array(all_weights).reshape(-1, weight_dim)
    
    return all_weights

# mean center and normalize the weight vectors
def process_weights(weights):
    mean_weights = weights.mean(axis=0)
    centered_weights = weights - mean_weights
    normalized_weights = sklearn.preprocessing.normalize(centered_weights, norm='l2')
    
    return normalized_weights

# apply kNN density filtration with k=200 and p=0.3
def findKNN(X,k):
    knn = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto', p=2)
    knn.fit(X)
    dist, nbrs= knn.kneighbors(X, n_neighbors=k, return_distance=True)
    filter_dist=dist[:, k-1].reshape(-1, 1)
    new_X = np.hstack((X, filter_dist))
    
    return new_X

# sort by distance to the 200th neighbor and extract the top fraction p
def NNfiltering (weights, neighbor, p_filter, weight_dim):
    num_weights = weights.shape[0]
    processed_weights = findKNN(weights, neighbor)
    sorted_weights = processed_weights[processed_weights[:,weight_dim].argsort()]
    filtered_weights = sorted_weights[:math.floor(num_weights*p_filter), :]
    new_weights = filtered_weights[:, :-1]
    #print (new_weights.shape)
    
    return new_weights

def main(): 
    # initialize model 
    architecture = [64, 8, 512]
    model = Sequential()
    model.add(Conv2D(architecture[0], kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(architecture[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(architecture[2], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
    # load weights 
    weights = load_weights(model, num_models, weight_dim)
    
    # process weights
    normalized_weights = process_weights(weights)
    
    # do nearest neighbor filtering
    neighbor_idx = 200
    p_filter = 0.3
    new_weights = NNfiltering(normalized_weights, neighbor_idx,p_filter, weight_dim)
    
    # initialize mapper
    mapper = km.KeplerMapper(verbose =1)
    
    # define the pca lens
    lens= mapper.fit_transform(new_weights, projection=PCA(n_components=2))
    
    # create graph
    graph = mapper.map(lens,
                      new_weights,
                      nr_cubes=30,
                      overlap_perc=0.67,
                      clusterer=sklearn.cluster.AgglomerativeClustering(n_clusters=2, 
                                                                        linkage = 'single'))
    mapper.visualize(graph, path_html="MNIST-648512.html",
                 title="MNIST-648512")
    
    diagrams1 = ripser(new_weights)['dgms']
    plot_diagrams(diagrams1, show = True)
    
    diagrams2 = ripser(weights)['dgms']
    plot_diagrams(diagrams2, show = True)


if __name__ == '__main__':
   main()