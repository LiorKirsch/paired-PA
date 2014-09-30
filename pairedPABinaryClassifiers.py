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
from scipy import sparse
import copy
# import fast_sparse

class basePA(sklearn.base.BaseEstimator):
  
    def __init__(self, C =1, repeat = 5000, seed =42):
        self.repeat = repeat
        self.seed = seed
        self.C = C
        self.w_ = None
        self.w_mean_ = None
        self.classes_ = None
        self.w_progress_ = []
        self.w_progress_mean_ = []
        
    def check_tracking(self,t,track_every_n_steps,w, w_mean):
        if (not track_every_n_steps == 0) and (t % track_every_n_steps == 1):
            self.w_progress_.append( copy.copy(w) )
            self.w_progress_mean_.append( copy.copy(w_mean) )
          
    def evaulate_tracking(self, evaluate_function, X_test, y_test):
        num_tracking_steps = len(self.w_progress_)
        
        results_w = []
        results_w_mean = []
        
        estimatorCopy = copy.deepcopy(self)
        for i in range(0, num_tracking_steps):
            estimatorCopy.w_ = self.w_progress_[i]
            results_w.append( evaluate_function(estimatorCopy, X_test, y_test) )
            
            estimatorCopy.w_ = self.w_progress_mean_[i]   
            results_w_mean.append( evaluate_function(estimatorCopy, X_test, y_test) )
            
        
        return (results_w, results_w_mean)
            
        
            
    def decision_function(self,X):
        return X.dot( self.w_.transpose())
    
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


    def dca_with_memory(self, X, Y, C, w_memory, early_stopping = np.Inf, minimum_gap = np.power(10.0,-14) ,balanced = False, X_norm = None):
        
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
            balanced_weights = np.ones(m)
            balanced_weights[pos_indcies] = balanced_weights[pos_indcies]  / num_pos
            balanced_weights[neg_indcies] = balanced_weights[neg_indcies]  / num_neg
            
            X =  sparse.dia_matrix( (balanced_weights, 0) , shape=(m, m)) * X
            samples_C = C * np.ones( m )
        else:
            samples_C = C * np.ones( m )
    
        # init w as the w from the former steps 
        w = w_memory
        M = 0; # counts mistakes
    
        flat_w = w.flatten()
        # Support vector machines
        alpha = np.zeros((m,))
        t = 0
        
        while t < early_stopping :
        
            i = 0    
            for i in range(0,m):
                current_X = X.getrow(i)
                X_nonzeros = current_X.nonzero()[1] 
                # predict
                prediction = current_X.dot(w.T)
#                 prediction = prediction.todense()
                prediction = prediction.item()
                
#                 prediction2 = fast_sparse.t_dot(X_nonzeros,current_X.data, flat_w)
                
                loss = 1.0- Y[i]*  prediction
                if loss > 0.0:
                    M = M + 1
                    # update w
                    if X_norm is None:
                        current_x_norm = (current_X.data**2).sum()
                    else:
                        current_x_norm = X_norm[i]
                          
                    tau = min(samples_C[i]-alpha[i], max([-alpha[i], loss/ current_x_norm]))
                    alpha[i] = alpha[i] + tau 
#                     w1 = w + tau*Y[i]* current_X
                    w[ X_nonzeros ] +=  tau*Y[i]*current_X.data
                    
#                     flat_w = fast_sparse.update_w(X_nonzeros,current_X.data, flat_w,tau, Y[i])
                    
                i += 1
                
            t = t +1
    
        return w

class classicPA(basePA):
    def fit(self, X, Y, track_every_n_steps = 0):
        self.classes_, Y = np.unique(Y, return_inverse=True) # transforms the classes into indices
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
        random.seed(self.seed)     
        m, d = X.shape;  # updated number of examples

        # weight vector
        w = np.zeros((1,d))
        w_mean = np.zeros((1,d))
        M = 0; # counts mistakes
    
        # Passive aggressive
        for t in range(0, self.repeat):
#             print(t)
            # choose example
            i = random.randint(0,m-1)
            current_X = X[i,:]
            # predict
            Yhat = np.sign( current_X.dot(w.T) )
            # compute hinge loss
            loss = max([0.0, 1.0- Y[i]* current_X.dot(w.T) ])
            if loss > 0.0:
                M = M + 1
                # update w
                try:
                    x_norm = (current_X.data**2).sum()
                except:
                    x_norm = current_X.dot(current_X)
                tau = min(self.C , loss/x_norm)
                
                w = w + tau*Y[i]* current_X
#                 X_nonzeros = current_X.nonzero()[1] 
#                 w[ 0,X_nonzeros ] +=  tau*Y[i]*current_X.data
                    
                    
                    
                    
            
            w_mean = (w_mean * (t) + w ) / (t+1)
            self.check_tracking(t,track_every_n_steps,w,w_mean)
                
        self.w_ = w
        self.w_mean_ = w_mean
        return self


class aucPA(basePA):
#     @profile
    def fit(self, X, Y, track_every_n_steps = 0):
        
        self.classes_, Y = np.unique(Y, return_inverse=True) # transforms the classes into indices
        Y = 2*Y -1  # transform  (0,1) to (-1,1)
        
        random.seed(self.seed)    
        X_pos = X[ Y>0,:]
        X_neg = X[ Y<0,:]
        
        n_pos, d = X_pos.shape;  # updated number of examples
        n_neg, d = X_neg.shape;  # updated number of examples
        
        # weight vector
        w = np.zeros((1,d))
        w_mean = np.zeros((1,d))
        M = 0; # counts mistakes
    
        # Every step choose a positive and a negative sample, 
        # construct a new vector which is x(pos) - x(neg) with a positive classification y=1
        # then run passive aggressive on that set
        
        for t in range(0, self.repeat):
            # choose example
            i_pos = random.randint(0,n_pos-1)
            i_neg = random.randint(0,n_neg-1)
            
            X_diff = X_pos[i_pos,:] - X_neg[i_neg,:]
            
            try:  
                X_diff_norm = (X_diff.data**2).sum()
            except:
                X_diff_norm = X_diff.dot(X_diff)
                
            # predict
            Yhat = np.sign( X_diff.dot(w.T) )
            # compute hinge loss
            loss = max([0.0, 1.0 - X_diff.dot(w.T) ])
            if loss > 0.0 and X_diff_norm > 0.0:
                M = M + 1
                # update w
              
                    
                tau = min(self.C , loss/X_diff_norm )
                w = w + tau* X_diff
            
            w_mean = (w_mean * (t) + w ) / (t+1)
            self.check_tracking(t,track_every_n_steps,w,w_mean)
            
        self.w_ = w
        self.w_mean_ = w_mean
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