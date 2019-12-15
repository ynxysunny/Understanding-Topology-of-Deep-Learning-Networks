#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:49:59 2019

@author: ynxysunny
"""

import get_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model

batch_size = 64
epochs = 40
num_classes=10
num_models = 100

X_train, X_test, y_train, y_test, input_shape = get_data.get_and_split()

architecture = [64, 32, 64]

model = Sequential()
model.add(Conv2D(filters=architecture[0], kernel_size=3, padding='same', activation='relu', input_shape=input_shape)) 
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=architecture[1], kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(architecture[2], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.save('weights/initial_model.h5')

print(model.summary())

for i in range(num_models):
    model = load_model('weights/initial_model.h5')    
    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights('weights/model%s_weights.h5' % i)
    del model