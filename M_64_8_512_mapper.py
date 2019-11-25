#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:05:22 2019

@author: Yun

Read in weight vectors, perform mean-centering and normalization, feed in Mapper
"""

import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

# read in weights
weights = np.genfromtxt('weights_64_8_512.csv',delimiter=',', dtype=np.float64)
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
indices = np.unique(indices[ind_of_closest_dist][:,[0,k-1]].flatten('C'))
filtered_weights = weights[indices,:]

# feed into mapper with resolution=30, gain=3
# provide affinity to clustering algorithm: Variance Normalized Euclidean
resolution = 30
gain = 3
perc_overlap = float(1-1/gain)
mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.fit_transform(filtered_weights, projection=PCA(n_components=2))
graph = mapper.map(projected_data,
                   filtered_weights,
                   clusterer=AgglomerativeClustering(linkage="single", n_clusters=120),
                   cover=km.Cover(resolution, perc_overlap))
mapper.visualize(graph, path_html="MNIST M(64,8,512).html",
                 title="MNIST M(64,8,512)")
