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
    def _fit(self, binary_classifier, X, Y):
        num_samples, d = X.shape
        
        # transforms the classes into indices
        self.classes_, Y = np.unique(Y, return_inverse=True) 
        num_classes = len(self.classes_)
        
        # weight vector for each class ( num_features, num_classes)
        w = np.zeros((d,num_classes))
        
        for j in range(num_classes):
            Y_current = -1 * np.ones( num_samples , np.double)
            Y_current[ Y==j ] = 1.0
            
            binary_classical_PA = binary_classifier( self.get_params() )
            binary_classical_PA.fit(X, Y_current)
            w[:,j] = binary_classical_PA.w_
            
        self.w_ = w
        return self
    
        
class oneVsAllClassicPA(oneVsAllClassifier):
    def fit(self, X, Y):
        binary_classifier = pairedPABinaryClassifiers.classicPA
        return self._fit(binary_classifier, X, Y)
       
         
class oneVsAllAucPA(oneVsAllClassifier):
    def fit(self, X, Y):
        binary_classifier = pairedPABinaryClassifiers.aucPA
        return self._fit(binary_classifier, X, Y)
                  
             
    
class multiClassPairedPA(baseMultiClassPA):
    
    def __init__(self, C =1, repeat = 5000, seed =42, early_stopping = 10, balanced_weight = None):
        self.early_stopping = early_stopping
        self.balanced_weight = balanced_weight
        baseMultiClassPA.__init__(self, C =1, repeat = 5000, seed =42)
        
    def fit(self, X, Y):
        # transforms the classes into indices
        self.classes_, Y = np.unique(Y, return_inverse=True) 
        num_classes = len(self.classes_)
        
        # divide X into the different classes
        X_classes = [None]*num_classes
        X_sizes = np.ndarray( (num_classes,1), np.double )
        for j in range(num_classes):
            X_classes[j]  =  X[ Y == j,:] 
            X_sizes[j], d  =  X_classes[j].shape  
            
        random.seed(self.seed)
        
        # weight vector for each class ( num_features, num_classes)
        w = np.zeros((d,num_classes))
    
        for t in range(0, self.repeat):
            # choose examples
            
            X_at_time_t = []
                 
            hinge_loss = np.ndarray((num_classes,), np.double)
            for j in range(num_classes):
                current_X = X_classes[j]
                example_ind = random.randint(0, X_sizes[j] -1)
                X_at_time_t.append( current_X[example_ind,:] )

                
            if X.getformat() == 'csr':
                X_at_time_t = sparse.vstack( X_at_time_t,  format='csr' )
            else:
                X_at_time_t = np.vstack( X_at_time_t )
          
            # Train a one vs all classifier for each class    
            for j in range(num_classes):
                Y_at_time_t = -1 * np.ones( num_classes ) 
                Y_at_time_t[j] = 1
                
                hinge_loss = 1.0 - Y_at_time_t * X_at_time_t.dot( w[:,j] ) 
                hinge_loss[ hinge_loss < 0.0] = 0.0
                
                  
                if np.any(hinge_loss > 0.0 ):
                    # if any of the samples has an error in classification
                    # solve a mini problem with the samples
                    w[:,j] = self.dca_with_memory(X_at_time_t, Y_at_time_t, self.C , w[:,j] ,early_stopping = self.early_stopping, balanced=self.balanced_weight)
                
        self.w_ = w
        return self