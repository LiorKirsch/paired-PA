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


    def dca_with_memory(self, X, Y, C, w_memory, early_stopping = np.Inf, minimum_gap = np.power(10.0,-14) ,balanced = False):
        
        m, d = X.shape;  # updated number of examples
        pos_indcies = Y>0
        neg_indcies = Y<0
        num_pos = np.count_nonzero(pos_indcies)
        num_neg = np.count_nonzero(neg_indcies)
        if balanced == 'problem':
            samples_C = np.ndarray( (m,1) )
            samples_C[pos_indcies] = C / num_pos
            samples_C[neg_indcies] = C / num_neg
        elif balanced == 'samples':
            X_new = np.ndarray( (m,d) )
            X_new[pos_indcies,:] = X[pos_indcies,:]  / num_pos
            X_new[neg_indcies,:] = X[neg_indcies,:]  / num_neg
            X = X_new
            samples_C = C * np.ones( m )
        else:
            samples_C = C * np.ones( m )
    
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

                loss = 1.0- Y[i]*np.dot(w , X[i,:])
                if loss > 0.0:
                    M = M + 1
                    # update w
                    tau = min(samples_C[i]-alpha[i], max([-alpha[i], loss/np.dot(X[i,:], X[i,:])]))
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

class classicPA(basePA):
     
    def fit(self, X, Y):
        self.classes_, Y = np.unique(Y, return_inverse=True) # transforms the classes into indices
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
        random.seed(self.seed)     
        m, d = X.shape;  # updated number of examples
        
        # weight vector
        w = np.zeros((d,));
        M = 0; # counts mistakes
    
        # Passive aggressive
        for t in range(0, self.repeat):
            # choose example
            i = random.randint(0,m-1)
            # predict
            Yhat = np.sign(np.dot(w, X[i,:]))
            # compute hinge loss
            loss = max([0.0, 1.0- Y[i]*np.dot(w, X[i,:])])
            if loss > 0.0:
                M = M + 1
                # update w
                tau = min(self.C , loss/np.dot(X[i,:], X[i,:]) )
                w = w + tau*Y[i]* X[i,:]

        self.w_ = w
        return self


class aucPA(basePA):
    def fit(self, X, Y):
        
        self.classes_, Y = np.unique(Y, return_inverse=True) # transforms the classes into indices
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
        random.seed(self.seed)    
        X_pos = X[ Y>0,:]
        X_neg = X[ Y<0,:]
        
        n_pos, d = X_pos.shape;  # updated number of examples
        n_neg, d = X_neg.shape;  # updated number of examples
        
        # weight vector
        w = np.zeros((d,));
        M = 0; # counts mistakes
    
        # Every step choose a positive and a negative sample, 
        # construct a new vector which is x(pos) - x(neg) with a positive classification y=1
        # then run passive aggressive on that set
        
        for t in range(0, self.repeat):
            # choose example
            i_pos = random.randint(0,n_pos-1)
            i_neg = random.randint(0,n_neg-1)
            
            X_diff = X_pos[i_pos,:] - X_neg[i_neg,:]
            # predict
            Yhat = np.sign(np.dot(w, X_diff))
            # compute hinge loss
            loss = max([0.0, 1.0 - np.dot(w, X_diff )])
            if loss > 0.0:
                M = M + 1
                # update w
                tau = min(self.C , loss/np.dot(X_diff, X_diff ) )
                w = w + tau* X_diff
        
        self.w_ = w
        return self
 
    
class pairedPA(basePA):
    
    def __init__(self, C =1, repeat = 5000, seed =42, early_stopping = 10):
        self.early_stopping = early_stopping
        basePA.__init__(self, C =1, repeat = 5000, seed =42)
        
    def fit(self, X, Y):
        self.classes_, Y = np.unique(Y, return_inverse=True) # transforms the classes into indices
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
#        print('running with %s' % self.get_params(False) )
        random.seed(self.seed)
        X_pos = X[ Y>0,:]
        X_neg = X[ Y<0,:]
        
        n_pos, d = X_pos.shape;  # updated number of examples
        n_neg, d = X_neg.shape;  # updated number of examples
        
        # weight vector
        w = np.zeros((d,));
        N = 0; # counts mistakes
        P = 0; # counts mistakes
    
        for t in range(0, self.repeat):
            # choose example
            i_pos = random.randint(0,n_pos-1)
            i_neg = random.randint(0,n_neg-1)
            # predict
            Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
            Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
            
            loss_pos = max([0.0, 1.0 - np.dot(w,  X_pos[i_pos,:] )  ])
            loss_neg = max([0.0, 1.0 + np.dot(w,  X_neg[i_neg,:] )  ])
            
                  
            if loss_pos > 0.0 or loss_neg > 0.0:
                # if any of the samples has an error in classification
                # solve a mini problem with two samples
                
                X_pair = np.array( [X_pos[i_pos,:],X_neg[i_neg,:]] )
                Y_pair = np.array( (1,-1) )
                w = self.dca_with_memory(X_pair, Y_pair, self.C / 2.0, w ,early_stopping = self.early_stopping)
                
        self.w_ = w
        return self