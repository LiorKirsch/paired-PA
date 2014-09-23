# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:56:20 2014

@author: Lior Kirsch

Using guidelines from http://scikit-learn.org/stable/developers/
"""

from __future__ import division
import numpy as np
import random
import sklearn
import pairedPABinaryClassifiers
import scipy.sparse as sparse


class baseMultiClassPA(pairedPABinaryClassifiers.basePA):
 
    def predict(self, X):
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]


class oneVsAllClassifier(baseMultiClassPA):
    def _fit(self, binary_classifier, X, Y, **kwags):
        num_samples, d = X.shape
        
        # transforms the classes into indices
        self.classes_, Y = np.unique(Y, return_inverse=True) 
        num_classes = len(self.classes_)
        
        # weight vector for each class ( num_features, num_classes)
        w = np.zeros((num_classes,d))
        w_mean = np.zeros((num_classes,d))
        
        for j in range(num_classes):
            Y_current = -1 * np.ones( num_samples , np.double)
            Y_current[ Y==j ] = 1.0
            
            binary_classical_PA = binary_classifier( **self.get_params() )
            binary_classical_PA.fit(X, Y_current, **kwags)
            w[j,:] = binary_classical_PA.w_
            w_mean[j,:] = binary_classical_PA.w_mean_
            
            self.w_progress_mean_.append( binary_classical_PA.w_progress_mean_ )
            self.w_progress_.append( binary_classical_PA.w_progress_ )
            
        self.w_ = w
        self.w_mean_ = w_mean
        return self
    
        
class oneVsAllClassicPA(oneVsAllClassifier):
    def fit(self, X, Y, **kwags):
        binary_classifier = pairedPABinaryClassifiers.classicPA
        return self._fit(binary_classifier, X, Y, **kwags)
       
         
class oneVsAllAucPA(oneVsAllClassifier):
    def fit(self, X, Y, **kwags):
        binary_classifier = pairedPABinaryClassifiers.aucPA
        return self._fit(binary_classifier, X, Y, **kwags)
                  
             
    
class multiClassPairedPA(baseMultiClassPA):
    
    def __init__(self, C =1, repeat = 5000, seed =42, early_stopping = 10, balanced_weight = None):
        self.early_stopping = early_stopping
        self.balanced_weight = balanced_weight
        baseMultiClassPA.__init__(self, C =C, repeat = repeat, seed =seed)
    

    def fit(self, X, Y):
        # transforms the classes into indices
        self.classes_, Y = np.unique(Y, return_inverse=True) 
        num_classes = len(self.classes_)
        num_samples,d = X.shape
        # divide X into the different classes
        try:
            X_norms_squared = X.multiply(X).sum(1) 
            X_norms_squared = np.array(X_norms_squared).flatten()
        except:
            X_norms_squared = (X**2).sum(1)
        

        X_classes_indices = [None]*num_classes
        X_sizes = np.ndarray( (num_classes,1), np.int64 )
        for j in range(num_classes):
            X_classes_indices[j]  =  np.where( Y == j )[0] 
            X_sizes[j] = len(X_classes_indices[j])  
            
 
            
        random.seed(self.seed)
        
        
        # weight vector for each class ( num_features, num_classes)
        w = np.zeros((num_classes,d))
        w = [None]*num_classes
        for j in range(num_classes):
            w[j] = sparse.csr_matrix((1,d) )
            w[j] = np.zeros((1,d))
        
        for t in range(0, self.repeat):
            # choose examples
            hinge_loss = np.ndarray((num_classes,), np.double)
            example_ind = np.ndarray( (num_classes), np.int64 )
            for j in range(num_classes):
                current_X_classes_indices = X_classes_indices[j]
                random_index = random.randint(0, X_sizes[j] -1)
                example_ind[j] = current_X_classes_indices[random_index]
                
            X_at_time_t = X[example_ind,:]
            X_norms_squared_time_t = X_norms_squared[example_ind]

            # Train a one vs all classifier for each class    
            for j in range(num_classes):
                Y_at_time_t = -1 * np.ones( num_classes ) 
                Y_at_time_t[j] = 1
                
                classifier_score =  (X_at_time_t.dot( w[j].T )).flatten() 
                classifier_score = np.multiply(Y_at_time_t ,classifier_score  )
                  
                if np.any( classifier_score < 1.0 ):
                    # if any of the samples has an error in classification
                    # solve a mini problem with the samples
                    w[j] = self.dca_with_memory(X_at_time_t, Y_at_time_t, self.C , w[j] ,early_stopping = self.early_stopping, balanced=self.balanced_weight,X_norm = X_norms_squared_time_t)
#                     passive = sklearn.linear_model.PassiveAggressiveClassifier(C=self.C, n_iter=1)
#                     passive.fit(X_at_time_t, Y_at_time_t, coef_init=w[j])
#                     w[j] = passive.coef_

        w = np.vstack(w)
        self.w_ = w
        return self