# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:02:16 2014

@author: jkeshet
"""


import numpy as np
import matplotlib.pyplot as plt
import random

# read data
training_data = np.loadtxt("training_data_1_vs_8.rs.dat.gz");
X = training_data[:,1:]
Y = training_data[:,0]

# convert 3 to +1 and 8 to -1
Y_binary = np.where(Y == 1, 1, -1)
m, d = X.shape;

#%%

# show some images
plt.figure(1);
for i in range (1,26):
    ax = plt.subplot(5,5,i);
    ax.axis('off');
    if Y_binary[i] > 0:
        ax.imshow(X[i,:].reshape(28,28), cmap="gray");
    else:
        ax.imshow(255-X[i,:].reshape(28,28), cmap="gray");
plt.draw();

#%%    

# weight vector
w = np.zeros((d,));
M = 0; # counts mistakes

# Perceptron
T = 2000
for t in range(0, T):
    # choose example
    i = random.randint(0, m-1)
    # predict
    Yhat = np.sign(np.dot(w, X[i,:]))
    if Y_binary[i] != Yhat:
        M = M + 1
        # update
        w = w + Y_binary[i]* X[i,:]
w_perceptron = w

#%%

# show the mask learnt by Perceptron
plt.figure(2);
ax1 = plt.subplot(1,2,1);
ax1.axis('off'); # no need for axis marks
ax2 = plt.subplot(1,2,2);
ax2.axis('off'); # no need for axis marks
ax1.imshow(w.reshape(28,28),cmap="gray");
tmp = 1/(1+np.exp(-10*w/w.max()));
ax2.imshow(tmp.reshape(28,28),cmap="gray");
plt.draw();

#%%    

# weight vector
w = np.zeros((d,));
M = 0; # counts mistakes

# Support vector machines
T = 2000
alpha = np.zeros((m,))
for t in range(0, T):
    # choose example
    i = random.randint(0,m-1)
    # predict
    Yhat = np.sign(np.dot(w, X[i,:]))
    # compute hinge loss
    loss = max([0.0, 1.0- Y_binary[i]*np.dot(w, X[i,:])])
    if loss > 0.0:
        M = M + 1
        # update w
        tau = max([-alpha[i], loss/np.dot(X[i,:], X[i,:])])
        alpha[i] = alpha[i] + tau 
        w = w + tau*Y_binary[i]* X[i,:]
w_svm = w

#%%

# show the mask learnt by SVM
plt.figure(3);
ax1 = plt.subplot(1,2,1);
ax1.axis('off'); # no need for axis marks
ax2 = plt.subplot(1,2,2);
ax2.axis('off'); # no need for axis marks
ax1.imshow(w.reshape(28,28),cmap="gray");
tmp = 1/(1+np.exp(-10*w/w.max()));
ax2.imshow(tmp.reshape(28,28),cmap="gray");
plt.draw();

#%%

# check performence on test data
test_data = np.loadtxt("test_data_1_vs_8.dat.gz");
X = test_data[:,1:]
Y = test_data[:,0]

# convert 3 to +1 and 8 to -1
Y_binary = np.where(Y == 1, 1, -1)
m, d = X.shape;

M_perceptron = 0
for t in range(0, m):
    Yhat = np.sign(np.dot(w_perceptron, X[t,:]))
    if Y_binary[t] != Yhat:
        M_perceptron = M_perceptron + 1
print "perceptron err=", float(M_perceptron)/m
        
M_svm = 0
for t in range(0, m):
    Yhat = np.sign(np.dot(w_svm, X[t,:]))
    if Y_binary[t] != Yhat:
        M_svm = M_svm + 1
print "SVM err=", float(M_svm)/m
        