# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:56:20 2014

@author: Lior Kirsch

Using guidelines from http://scikit-learn.org/stable/developers/
"""

from __future__ import division
import numpy as np
import random
from numpy.linalg import norm
import sklearn
from numpy import double, ones

class basePA(sklearn.base.BaseEstimator):
  
    def __init__(self, C =1, repeat = 5000, seed =42):
        self.repeat = repeat
        self.seed = seed
        self.C = C
        self.w_ = None
        self.classes_ = None
  
    def decision_function(self,X):
        return np.dot(X, self.w_)
    
#     def predict_proba(self,X):
#         n_samples,d = X.shape;
#         decisions =  self.decision_function(X)
#         probabilties = 1 / (1 + np.exp( -decisions ))
#         classes_probabilties = np.zeros((n_samples,2));
#         classes_probabilties[:,1] = probabilties
#         classes_probabilties[:,0] = 1 - probabilties
#         return classes_probabilties
#     
    def predict(self, X):
#         D = self.decision_function(X)
#         return self.classes_[np.argmax(D, axis=1)]

        predictions =  self.decision_function(X)
        postive_predictions = predictions >= 0
        predictions[ postive_predictions ] = 1
        predictions[ ~postive_predictions ] = -1
        return predictions



    
class multiClassPairedPA(basePA):
    
    def __init__(self, C =1, repeat = 5000, seed =42, early_stopping = 10):
        self.early_stopping = early_stopping
        basePA.__init__(self, C =1, repeat = 5000, seed =42)
        
    def fit(self, X, Y):
        # transforms the classes into indices
        self.classes_, Y = np.unique(Y, return_inverse=True) 
        num_classes = len(self.classes_)
        
        # divide X into the different classes
        X_classes = []
        X_sizes = []
        for j in range(num_classes):
            X_classes[j]  =  X[ Y == j,:] 
            X_sizes[j], d  =  X_classes[j].shape  
            
            
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
#        print('running with %s' % self.get_params(False) )
        random.seed(self.seed)
        
        # weight vector for each class ( num_features, num_classes)
        w = np.zeros((d,num_classes));
    
        for t in range(0, self.repeat):
            # choose examples
            example_ind = np.ndarray((num_classes,),int)
            X_at_time_t = np.array( (num_classes,d) )
            losses = np.ndarray((num_classes,), double)
            for j in range(num_classes):
                example_ind[j] = random.randint(0, X_sizes[j] -1)
                X_at_time_t[j,:] = X_classes[j][example_ind]
            
            # Train a one vs all classifier for each class    
            for j in range(num_classes):
                Y_at_time_t = -1 * np.ones( (num_classes,1) ) 
                Y_at_time_t[j] = 1
                
                losses = max([0.0, 1.0 - np.dot(X_at_time_t * Y_at_time_t, w[:,j])  ])
                
                  
                if np.any(losses > 0.0 ):
                    # if any of the samples has an error in classification
                    # solve a mini problem with samples
                    
                    w[:,j] = self.dca_with_memory(X_at_time_t, Y_at_time_t, self.C / 2.0, w[:,j] ,early_stopping = self.early_stopping)
                
        self.w_ = w
        return self
       
    def dca_with_memory(self, X, Y, C, w_memory, early_stopping = np.Inf, minimum_gap = np.power(10.0,-14) ):
        
        m, d = X.shape;  # updated number of examples
    
        # init w as the w from the former steps 
        w = w_memory
        M = 0; # counts mistakes
    
    
        # Support vector machines
        alpha = np.zeros((m,))
    #    losses = np.zeros((m,))
        memory_losses = np.zeros((m,))
        
        for i in range(0,m):
                memory_losses[i] = 1.0- Y[i]*np.dot(w, X[i,:])
            
        #primal = 0.5*np.dot(w, w) + C* np.sum( 1 - 0.5*np.dot(w_memory, w_memory)
        t = 0
        dual_diff = np.Inf
        dual = np.sum(alpha) -0.5*np.dot(w , w) + 0.5*np.dot(w_memory, w_memory)
        
        
        while t < early_stopping and dual_diff > minimum_gap:
            for i in range(0,m):
                # predict
                Yhat = np.sign(np.dot(w, X[i,:]))
                # compute hinge loss
    #            loss = max([0.0, 1.0- Y[i]*np.dot(w , X[i,:])])
                loss = 1.0- Y[i]*np.dot(w , X[i,:])
                if loss > 0.0:
                    M = M + 1
                    # update w
                    tau = min(C-alpha[i], max([-alpha[i], loss/np.dot(X[i,:], X[i,:])]))
                    alpha[i] = alpha[i] + tau 
                    w = w + tau*Y[i]* X[i,:]
                    
            t = t +1
    
            dual_new = np.sum(alpha) -0.5*np.dot(w , w) + 0.5*np.dot(w_memory, w_memory)
            tmp = np.prod([alpha, memory_losses], 0)
            dual_new2 = np.sum(tmp) -0.5*np.dot(w + w_memory , w + w_memory) 
            dual_diff = np.abs(dual_new - dual)
    #        print('dual step differance:%g\n' % dual_diff)
            dual = dual_new
      
    #        losses = np.zeros((m,))
    #        for i in range(0,m):
    #            losses[i] = max([0 , 1.0- Y[i]*np.dot(w, X[i,:]) ])
    #        
    #        primal = 0.5*np.dot(w - w_memory, w - w_memory) + C* np.sum( losses)
    #        print('dual gap:%g\n' % (primal - dual) )
        return w