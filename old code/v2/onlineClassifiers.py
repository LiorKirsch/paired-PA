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
#        loss = max([0.0, 1.0- Y[i]*np.dot(w, X[i,:])])
        loss = 1.0- Y[i]*np.dot(w, X[i,:])
        if loss > 0.0:
            M = M + 1
            # update w
            tau = min(C-alpha[i], max([-alpha[i], loss/np.dot(X[i,:], X[i,:])]))
            alpha[i] = alpha[i] + tau 
            w = w + tau*Y[i]* X[i,:]

    return w

def svm_compute_auc(X, Y, C, repeat= 5000, seed = 42):
    
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

def svm_dual_double_step(X, Y, C, repeat= 5000, seed = 42):
    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    alpha_pos = np.zeros((n_pos,))
    alpha_neg = np.zeros((n_neg,))
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        
        # compute hinge loss
        loss = max([0.0, 1.0- np.dot(w, X_pos[i_pos,:])])
        if loss > 0.0:
            M = M + 1
            # update w
            tau = min(C-alpha_pos[i_pos], max([alpha_pos[i_pos], loss/np.dot(X_pos[i_pos,:], X_pos[i_pos,:] ) ]))
            alpha_pos[i_pos] = alpha_pos[i_pos] + tau 
            w = w + tau* X_pos[i_pos,:]
            
        loss = max([0.0, 1.0 + np.dot(w, X_neg[i_neg,:])])
        if loss > 0.0:
            M = M + 1
            # update w
            tau = min(C-alpha_neg[i_neg], max([alpha_neg[i_neg], loss/np.dot(X_neg[i_neg,:], X_neg[i_neg,:])]))
            alpha_neg[i_neg] = alpha_neg[i_neg] + tau 
            w = w - tau* X_neg[i_neg,:]
      
    
    return w

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