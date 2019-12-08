#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:04:54 2019

@author: ynxysunny
"""

import wav2mfcc
from sklearn.model_selection import train_test_split

def get_and_split():
    mfccs, labels = wav2mfcc.get_data()

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    return X_train, X_test, y_train, y_test, input_shape