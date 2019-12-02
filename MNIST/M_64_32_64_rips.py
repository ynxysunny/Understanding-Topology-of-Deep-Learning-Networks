#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:22:36 2019

@author: Yun
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import process_nparray
from ripser import ripser
from persim import plot_diagrams

weights = []
for num_model in range(20):
    model = load_model(f"MNIST_trained_models_64_32_64/model_{num_model}.h5")
    conv_layer = model.get_layer('conv')
    w = conv_layer.get_weights()[0]
    for i in range(w.shape[-1]):
        filt = w[:,:,:,i].flatten('C')
        weights.append(filt)
weights = np.array(weights, dtype=np.float64)

filtered = process_nparray(weights)
diagrams = ripser(filtered)['dgms']
plot_diagrams(diagrams, show=True)
