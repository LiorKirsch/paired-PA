'''
Created on Sep 28, 2014

@author: lior
'''



import cython
cimport cython

import numpy as np
cimport numpy as np

from scipy import sparse

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def balanceSamples(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=1] Y, double C, balanced):

    cdef unsigned int m = array.shape[0]
    cdef np.ndarray[np.uint8, cast=True] pos_indcies = (Y > 0)
    cdef np.ndarray[np.uint8, cast=True] neg_indcies = (Y < 0)
    cdef unsigned int num_pos = np.sum(pos_indcies, axis=0)
    cdef unsigned int num_neg = np.sum(pos_indcies, axis=0)
    
    cdef np.ndarray[DTYPE_t, ndim=1] samples_C = np.ones(m)
    
    if balanced == 'problem':
        samples_C[pos_indcies] = C / num_pos
        samples_C[neg_indcies] = C / num_neg
    elif balanced == 'samples':
    
    cdef unsigned int cols = array.shape[1]
    cdef unsigned int row, col, row2
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((rows, cols))

    for row in range(rows):
        for row2 in range(rows):
            for col in range(cols):
                out[row, col] += array[row2, col] - array[row, col]

    return out

 
def cython_dca_with_memory(p.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=1] Y, double C, np.ndarray[DTYPE_t, ndim=1] w_memory, unsigned int early_stopping = np.Inf, minimum_gap = np.power(10.0,-14), np.ndarray[DTYPE_t, ndim=1] X_norm = None):
    
        cdef unsigned int m = array.shape[0]
    
        # init w as the w from the former steps
        cdef np.ndarray[DTYPE_t, ndim=1] w = w_memory
        
        # balance the positive and negative samples
#         X, samples_C = balanceSamples(X, Y, C, balanced)
            
        # Support vector machines
        cdef np.ndarray[DTYPE_t, ndim=1] alpha = np.zeros(m)
        cdef unsigned int t = 0
        
        while t < early_stopping :
            for i in range(0,m):
                current_X = X.getrow(i)
                # predict
                prediction = current_X.dot(w.T)
#                 prediction = prediction.todense()
                prediction = prediction.item()
                
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
                    
#                     w = w + tau*Y[i]* current_X
                    w[ 0,current_X.nonzero()[1] ] +=  tau*Y[i]*current_X.data
                    
            
            t = t +1
    
        return w
    
def balanceSamples(X, Y, C, balanced = False):
    '''
    balanced == 'problem': uses C_positive for positive samples and C_negative.
    balanced == 'samples': normalizes the features according to the number of samples in each class.
    '''
     
    m = X.shape[0]
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
        
    return (X, samples_C)
 
 
def dca_with_memory(X, Y, C, w_memory, early_stopping = np.Inf, minimum_gap = np.power(10.0,-14) ,balanced = False, X_norm = None):
        
        m = X.shape[0];  # updated number of examples
    
        # init w as the w from the former steps 
        w = w_memory
        M = 0; # counts mistakes
    
        # balance the positive and negative samples
        X, samples_C = balanceSamples(X, Y, C, balanced)
            
        # Support vector machines
        alpha = np.zeros((m,))
        t = 0
        
        while t < early_stopping :
            for i in range(0,m):
                current_X = X.getrow(i)
                # predict
                prediction = current_X.dot(w.T)
#                 prediction = prediction.todense()
                prediction = prediction.item()
                
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
                    
#                     w = w + tau*Y[i]* current_X
                    w[ 0,current_X.nonzero()[1] ] +=  tau*Y[i]*current_X.data
                    
            
            t = t +1
    
        return w