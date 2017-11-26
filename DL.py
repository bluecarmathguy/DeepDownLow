# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:58:18 2017

@author: bob
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import csv


fid  = open('/home/bob/code/python/Deep_Learn/hw_data.csv','r')
r = csv.reader(fid)
A = []
y = []
for ii in r:
    A.append([float(jj) for jj in ii[:2]])
    y.append(int(ii[2]))
fid.close()
B = np.array(A)


fid  = open('/home/bob/code/python/Deep_Learn/hw_red.csv','r')
r = csv.reader(fid)
C = []
yt = []
for ii in r:
    C.append([float(jj) for jj in ii[:2]])
    yt.append(int(ii[2]))
fid.close()
D = np.array(C)



model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
#model.add(Dropout(0.5))
#odel.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(B, y,epochs=20,batch_size=128)
score = model.evaluate(D, yt, batch_size=128)