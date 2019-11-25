#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:48:48 2019

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
num_models = 37


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# loading weights from files
all_weights = [] 
for i in range(num_models): 
    model.load_weights('model%s_weights.h5' %i)
    weights = model.get_weights()
    fl_weights = np.einsum('klij->ijkl', weights[0])
    fl_weights = fl_weights.reshape(-1,9)
    all_weights.append(fl_weights)

all_weights = np.array(all_weights).reshape(-1, 9)
num_weights = all_weights.shape[0]
print (all_weights.shape)

# mean center and normalize the weight vectors
mean_weights = all_weights.mean(axis=0)
centered_weights = all_weights - mean_weights
normalized_weights = sklearn.preprocessing.normalize(centered_weights, norm='l2')

# apply kNN density filtration with k=200 and p=0.3
def findKNN(X,k):
    knn = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto', p=2)
    knn.fit(X)
    dist, nbrs= knn.kneighbors(X, n_neighbors=k, return_distance=True)
    filter_dist=dist[:, k-1].reshape(-1, 1)
    new_X = np.hstack((X, filter_dist))
    
    return new_X

# sort by distance to the 200th neighbor and extract the top fraction p
p_filter = 0.3
processed_weights = findKNN(normalized_weights, 200)
sorted_weights = processed_weights[processed_weights[:,9].argsort()]
filtered_weights = sorted_weights[:math.floor(num_weights*p_filter), :]
new_weights = filtered_weights[:, :-1]
print (new_weights.shape)


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
mapper.visualize(graph, path_html="MNIST-643264.html",
                 title="MNIST-643264")

diagrams1 = ripser(new_weights)['dgms']
plot_diagrams(diagrams1, show = True)

diagrams2 = ripser(all_weights)['dgms']
plot_diagrams(diagrams2, show = True)

