'''
Created on Jul 8, 2014

@author: lior
'''


from __future__ import division
import numpy as np
import random
from numpy.linalg import norm


def pairedPA_one_loss(X, Y, C, repeat = 5000, seed = 42):
    
    random.seed(seed)
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    N = 0; # counts mistakes
    P = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        # predict
        Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
        Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
        
        x_pos_norm = norm( X_pos[i_pos,:] )
        x_neg_norm = norm( X_neg[i_neg,:] )
        x_diff_norm = norm( X_pos[i_pos,:] - X_neg[i_neg,:] )
        loss_pos = max([0.0, 1.0 - np.dot(w,  X_pos[i_pos,:] )  ])
        loss_neg = max([0.0, 1.0 + np.dot(w,  X_neg[i_neg,:] )  ])
        
              
        if loss_pos > 0.0 :
            
            if loss_neg > 0.0:
                #update w using both the positive and the negative
                P = P + 1
                N = N + 1
                loss_auc = 1.0 - np.dot(w,  X_pos[i_pos,:] - X_neg[i_neg,:] )  
                tau_auc = min(C,     loss_auc / (x_diff_norm*x_diff_norm)  ) 
                w = w + tau_auc * ( X_pos[i_pos,:] - X_neg[i_neg,:] )
            else:
                #update using only the positive
                P = P + 1
                tau_pos = min(C/2.0,     loss_pos / (x_pos_norm*x_pos_norm)  ) 
                w = w + tau_pos * X_pos[i_pos,:]
        else:    
            if loss_neg > 0.0:
                # update w using the negative sample
                N = N + 1
                tau_neg = min(C/2.0,     loss_neg / (x_neg_norm*x_neg_norm)  ) 
                w = w - tau_neg *  X_neg[i_neg,:]
            # else: no update is needed 

    return w

def pairedPA_exact(X, Y, C, repeat = 5000, seed = 42, early_stopping = np.Inf):
    
    random.seed(seed)
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    N = 0; # counts mistakes
    P = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
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
            w = dca_with_memory(X_pair, Y_pair, C/2.0, w ,early_stopping=early_stopping)
            

    return w

def dca_with_memory(X, Y, C, w_memory, early_stopping = np.Inf, minimum_gap = np.power(10.0,-14) ):
    
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