#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:05:22 2019

@author: Yun

Read in weight vectors, perform mean-centering and normalization, feed in Mapper
"""

import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from preprocess import process_csv

filtered_weights = process_csv('weights_64_8_512.csv')

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
