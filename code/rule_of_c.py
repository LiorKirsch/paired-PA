# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:02:16 2014

@author: jkeshet
"""


import numpy as np
import matplotlib.pyplot as plt
import random

# generate data
m = 1000
d = 2
desired_num_pos = 10
w_opt = np.array([1,2])
# random numbers in [0,1]
X = np.random.rand(m,d)  
# convert to numbers in [-1,1]
X = 2.0*X - 1.0
# set the label
Y  = np.zeros((m,))
num_pos = 0
indices_to_delete = list()
for i in range(m):
    Y[i] = np.sign(np.dot(w_opt, X[i,:]))
#    # set margin
#    if Y[i]*np.dot(w_opt, X[i,:]) < 0.5:
#        indices_to_delete.append(i)
# 
#X = np.delete(X, indices_to_delete, 0)
#Y = np.delete(Y, indices_to_delete, 0)
m, d = X.shape;  # updated number of examples

# add noise
for i in range(m):
    if i % 10 == 0:
        Y[i] = -Y[i]


#%%

# show some images
plt.figure(1);
index_pos = np.where(Y > 0)
index_neg = np.where(Y < 0)
plt.plot(X[index_pos, 0], X[index_pos, 1], 'ro')
plt.plot(X[index_neg, 0], X[index_neg, 1], 'gs')
plt.draw();

#%%    

# weight vector
w = np.zeros((d,));
M = 0; # counts mistakes
C = 1.0

# Support vector machines
T = 5000
alpha = np.zeros((m,))
for t in range(0, T):
    # choose example
    i = random.randint(0,m-1)
    # predict
    Yhat = np.sign(np.dot(w, X[i,:]))
    # compute hinge loss
    loss = max([0.0, 1.0- Y[i]*np.dot(w, X[i,:])])
    if loss > 0.0:
        M = M + 1
        # update w
        tau = min(C-alpha[i], max([-alpha[i], loss/np.dot(X[i,:], X[i,:])]))
        alpha[i] = alpha[i] + tau 
        w = w + tau*Y[i]* X[i,:]
w_svm = w

#%%

plt.plot([0, 1, -1],[0, -w_svm[0]/w_svm[1], w_svm[0]/w_svm[1]], 'b-')
plt.draw();

#%%    

# weight vector
w = np.zeros((d,));
M = 0; # counts mistakes

# Perceptron
T = 5000
for t in range(0, T):
    # choose example
    i = random.randint(0, m-1)
    # predict
    Yhat = np.sign(np.dot(w, X[i,:]))
    if Y[i]*np.dot(w, X[i,:]) <= 0.0:
        M = M + 1
        # update
        w = w + Y[i]* X[i,:]
w_perceptron = w

#%%

plt.plot([0, 1, -1],[0, -w_perceptron[0]/w_perceptron[1], w_perceptron[0]/w_perceptron[1]], 'y-')
plt.draw();

#%%    

