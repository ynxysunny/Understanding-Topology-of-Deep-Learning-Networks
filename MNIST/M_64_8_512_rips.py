#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:24:38 2019

@author: Yun

Calculate persistent homology of the weights collected from M(64,8,512)
"""

import numpy as np
from ripser import ripser
from persim import plot_diagrams
from sklearn import datasets
from preprocess import process_csv


# read in weights
filtered_weights = process_csv('weights_64_8_512.csv')
#data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]
diagrams = ripser(filtered_weights)['dgms']
plot_diagrams(diagrams, show=True)

