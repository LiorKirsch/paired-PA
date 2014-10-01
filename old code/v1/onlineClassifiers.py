'''
Created on Jul 9, 2014

@author: lior
'''
from __future__ import division
import numpy as np
import random

def svm_dual(X, Y, C, repeat= 5000, seed = 42):
    
    random.seed(seed)    
    m, d = X.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Support vector machines
    alpha = np.zeros((m,))
    for t in range(0, repeat):
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

    return w

def svm_dual_pairs(X, Y, C, repeat= 5000, seed = 42):
    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    m = n_pos * n_neg  # number of pairs
    X_pairs = np.zeros( (m,d) )
    Y_pairs = np.ones( (m,1) )
    ind = 0
    for i in range(0, n_pos):
        for j in range(0, n_neg):
            X_pairs[ind] = X_pos[i] - X_neg[j]
            Y_pairs[ind] = 1
            ind = ind + 1
            
    w = svm_dual(X_pairs, Y_pairs, C, repeat, seed)
    
    return w

#    random.seed(seed)    
#    X_pos = X[ Y>0,:]
#    X_neg = X[ Y<0,:]
#    
#    n_pos, d = X_pos.shape;  # updated number of examples
#    n_neg, d = X_neg.shape;  # updated number of examples
#    m = n_pos * n_neg  # number of pairs
#        
#    # weight vector
#    w = np.zeros((d,));
#    M = 0; # counts mistakes
#
#    random.seed(seed)    
#
#
#    # Support vector machines
#    alpha = np.zeros((m,))
#    for t in range(0, repeat):
#        # choose example
#        i = random.randint(0,n_pos-1)
#        j = random.randint(0,n_neg-1)
#        sample = X_pos[i,:] - X_neg[j,:]
#        
#        a = i * n_neg + j
#        # predict
#        Yhat = np.sign(np.dot(w, sample))
#        # compute hinge loss
#        loss = max([0.0, 1.0- np.dot(w, sample)])
#        if loss > 0.0:
#            M = M + 1
#            # update w
#            tau = min(C-alpha[a], max([-alpha[a], loss/np.dot(sample, sample)]))
#            alpha[a] = alpha[a] + tau 
#            w = w + tau*sample
#
#    return w

def perceptron(X,Y, repeat= 5000, seed = 42):
    
    random.seed(seed)    
    m, d = X.shape;  
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
    return w