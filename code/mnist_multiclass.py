# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:02:16 2014

@author: jkeshet
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# read data
data = np.loadtxt("training_data_1_vs_8_vs_4.rs.dat.gz");
X = data[:,1:]
Y = data[:,0]
m, d = X.shape;  # m number of examples, d instance dimesion
k = max(Y)+1  # k number of classes

#%%

# show some images
plt.figure(1);
for i in range (1,26):
    ax = plt.subplot(5,5,i);
    ax.axis('off');
    ax.imshow(255-X[i,:].reshape(28,28),cmap="gray");
plt.draw();

#%%    

# weight vector
w = np.zeros((k, d));
M = 0; # counts mistakes

# Support vector machines
T = 6000
alpha = np.zeros((m,))
for t in range(0, T):
    # choose example
    i = random.randint(0,m-1)
    # predict
    Yhat = np.argmax(np.dot(w, X[1,:]))
    # compute hinge loss
    loss1 = max([0.0, 1.0 - np.dot(w[Y[i],:], X[i,:]) + np.dot(w[Yhat,:], X[i,:])])
    if loss1 > 0.0:
        M = M + 1
        # update w
        tau = loss1/np.dot(X[i,:], X[i,:])
        w[Y[i],:] = w[Y[i],:] + tau*X[i,:]
        w[Yhat,:] = w[Yhat,:] - tau*X[i,:]

#%%

# show the mask learnt by multiclass SVM

plt.figure(2);
classes = [1, 4, 8]
for i, c in enumerate(classes):
    print i, c
    ax1 = plt.subplot(1,3,i+1);
    ax1.axis('off');
    ax1.imshow(w[c,:].reshape(28,28),cmap="gray");
plt.draw();

#%%

# check performence on test data
test_data = np.loadtxt("test_data_1_vs_8_vs_4.dat.gz");
X = test_data[:,1:]
Y = test_data[:,0]
m, d = X.shape;

M = 0
for i in range(0, m):
    Yhat = np.argmax(np.dot(w, X[i,:]))
    if Y[i] != Yhat:
        M = M + 1
print "test err=", float(M)/m

