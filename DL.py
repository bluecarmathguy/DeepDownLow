# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:58:18 2017

@author: bob
"""

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn import preprocessing
import csv 

fid  = open('./hw_data.csv','r')
r = csv.reader(fid);
A = []
y = []
for ii in r:
    A.append([float(jj) for jj in ii[:2]])
    y.append(int(ii[2]))
fid.close()
B = np.array(A)
scaler = preprocessing.StandardScaler().fit(B)
B= scaler.transform(B)

y = to_categorical(y);
#y=list(map(lambda x :x-1,y));

fid  = open('./hw_red.csv','r')
r = csv.reader(fid)
C = []
yt = []
for ii in r:
    C.append([float(jj) for jj in ii[:2]])
    yt.append(int(ii[2]))
fid.close()
D = np.array(C)
D=scaler.transform(D)
yt = to_categorical(yt);
#yt=list(map(lambda x :x-1,yt));

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(B, y,epochs=20,batch_size=128)
#score = model.evaluate(D, yt, batch_size=128)
score = model.predict_classes(D)
model.predict_classes(scaler.transform(np.array([5.0,120.0]).reshape(1,-1)))
model.predict_classes(scaler.transform(np.array([6.0,150.0]).reshape(1,-1)))
model.predict_classes(scaler.transform(np.array([6.0,300.0]).reshape(1,-1)))