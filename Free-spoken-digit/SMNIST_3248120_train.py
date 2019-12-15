#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:06:25 2019

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

architecture = [32, 48, 120]

model = Sequential()
model.add(Conv2D(architecture[0], kernel_size=3, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(architecture[1], kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(architecture[2], kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.save('weights/initial_model.h5')

print(model.summary())

for i in range(96,num_models):
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
