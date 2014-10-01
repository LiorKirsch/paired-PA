'''
Created on Jul 8, 2014

@author: lior
'''


from __future__ import division
import numpy as np
import random
from numpy.linalg import norm


def pairedPA_new(X, Y, C_pos, C_neg, repeat = 5000, seed = 42):
    
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
        cos_between = np.dot(X_pos[i_pos,:], X_neg[i_neg,:]) / x_pos_norm / x_neg_norm
        
        projection_on_neg = cos_between * x_pos_norm / x_neg_norm
        projection_on_pos = cos_between * x_neg_norm / x_pos_norm
        loss_pos = max([0.0, 1.0 + projection_on_neg - np.dot(w,  X_pos[i_pos,:] - X_neg[i_neg,:] * projection_on_neg )])
        loss_neg = max([0.0, 1.0 + projection_on_pos + np.dot(w,  X_neg[i_neg,:] - X_pos[i_pos,:] * projection_on_pos )])
        
        
        
        if loss_pos > 0.0 :
            # update w using the positive sample
            P = P + 1
            tau_pos = min(C_pos,     loss_pos / (1 - cos_between*cos_between) / (x_pos_norm*x_pos_norm)  ) 
            w = w + tau_pos * X_pos[i_pos,:]
            
        if loss_neg > 0.0:
            # update w using the negative sample
            N = N + 1
            tau_neg = min(C_neg,     loss_neg / (1 - cos_between*cos_between) / (x_neg_norm*x_neg_norm)  ) 
            w = w - tau_neg *  X_neg[i_neg,:] 

    return w, P, N

def pairedPA(X, Y, C_pos, C_neg, repeat = 5000, seed = 42):
    
    random.seed(seed)
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        # predict
        Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
        Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
        
        # compute hinge loss
        loss_pos = max([0.0, 1.0 - np.dot(w, X_pos[i_pos,:])])
        loss_neg = max([0.0, 1.0 + np.dot(w, X_neg[i_neg,:])])
        
        
        
        if loss_pos > 0.0 or loss_neg > 0.0:
            M = M + 1
            # update w
            x_pos_norm = np.dot(X_pos[i_pos,:], X_pos[i_pos,:]);
            x_neg_norm = np.dot(X_neg[i_neg,:], X_neg[i_neg,:]);
            cross = np.dot(X_pos[i_pos,:], X_neg[i_neg,:]);
            
   
            tau_pos = max(0,   min(C_pos,     (loss_pos*x_neg_norm + loss_neg*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
            tau_neg = max(0,   min(C_neg,     (loss_neg*x_pos_norm + loss_pos*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
                
            w = w + tau_pos * X_pos[i_pos,:] - tau_neg *  X_neg[i_neg,:] 

    return w, M


def pairedPA2(X, Y, C_pos, C_neg, repeat = 5000, seed = 42):

    random.seed(seed)    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        # predict
        Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
        Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
        
        # compute hinge loss
        loss_pos = 1.0 - np.dot(w, X_pos[i_pos,:])
        loss_neg = 1.0 + np.dot(w, X_neg[i_neg,:])
        
        
        
        if loss_pos > 0.0 or loss_neg > 0.0:
            M = M + 1
            # update w
            x_pos_norm = np.dot(X_pos[i_pos,:], X_pos[i_pos,:]);
            x_neg_norm = np.dot(X_neg[i_neg,:], X_neg[i_neg,:]);
            cross = np.dot(X_pos[i_pos,:], X_neg[i_neg,:]);
            
   
            tau_pos = max(0,   min(C_pos,     (loss_pos*x_neg_norm + loss_neg*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
            tau_neg = max(0,   min(C_neg,     (loss_neg*x_pos_norm + loss_pos*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
                
            w = w + tau_pos * X_pos[i_pos,:] - tau_neg *  X_neg[i_neg,:] 

    return w, M

def pairedPA_alt(X, Y, C_pos, C_neg, repeat = 5000, seed = 42):

    random.seed(seed)    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes

    
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        # predict
        Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
        Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
        
        # compute hinge loss
        loss_pos = max([0.0, 1.0 - np.dot(w, X_pos[i_pos,:])])
        loss_neg = max([0.0, 1.0 + np.dot(w, X_neg[i_neg,:])])
        
        
        
        if loss_pos > 0.0 or loss_neg > 0.0:
            M = M + 1
            # update w
            x_pos_norm = np.dot(X_pos[i_pos,:], X_pos[i_pos,:]);
            x_neg_norm = np.dot(X_neg[i_neg,:], X_neg[i_neg,:]);
            cross = np.dot(X_pos[i_pos,:], X_neg[i_neg,:]);
            
            if loss_pos > 0.0:
                if loss_neg > 0.0:
                    tau_pos = max(0,   min(C_pos,     (loss_pos*x_neg_norm + loss_neg*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
                    tau_neg = max(0,   min(C_neg,     (loss_neg*x_pos_norm + loss_pos*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
                else:
                    tau_pos =   min(C_pos,     loss_pos / x_pos_norm  ) 
                    tau_neg = 0
            else:
                tau_pos = 0 
                tau_neg =   min(C_neg,     loss_neg / x_neg_norm  )
                
            w = w + tau_pos * X_pos[i_pos,:] - tau_neg *  X_neg[i_neg,:] 

    return w, M


def PA(X, Y, C, repeat = 5000, seed = 42):
   
    random.seed(seed)     
    m, d = X.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Passive aggressive
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
            tau = min(C , loss/np.dot(X[i,:], X[i,:]) )
            w = w + tau*Y[i]* X[i,:]

    return w, M

def PA_auc(X, Y, C, repeat=5000, seed = 42):
    
    random.seed(seed)    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
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
            tau = min(C , loss/np.dot(X_diff, X_diff ) )
            w = w + tau* X_diff

    return w, M

def pairedPA_without_cross(X, Y, C_pos, C_neg, repeat = 5000, seed = 42):
    
    random.seed(seed)    
    X_pos = X[ Y>0,:]
    X_neg = X[ Y<0,:]
    
    n_pos, d = X_pos.shape;  # updated number of examples
    n_neg, d = X_neg.shape;  # updated number of examples
    
    # weight vector
    w = np.zeros((d,));
    M = 0; # counts mistakes


    # Support vector machines
    
    for t in range(0, repeat):
        # choose example
        i_pos = random.randint(0,n_pos-1)
        i_neg = random.randint(0,n_neg-1)
        # predict
        Yhat_pos = np.sign(np.dot(w, X_pos[i_pos ,:]))
        Yhat_neg = np.sign(np.dot(w, X_neg[i_neg ,:]))
        
        # compute hinge loss
        loss_pos = max([0.0, 1.0 - np.dot(w, X_pos[i_pos,:])])
        loss_neg = max([0.0, 1.0 + np.dot(w, X_neg[i_neg,:])])
        
        
        
        if loss_pos > 0.0 or loss_neg > 0.0:
            M = M + 1
            # update w
            x_pos_norm = np.dot(X_pos[i_pos,:], X_pos[i_pos,:]);
            x_neg_norm = np.dot(X_neg[i_neg,:], X_neg[i_neg,:]);
            cross = 0.0;
            
   
            tau_pos = max(0,   min(C_pos,     (loss_pos*x_neg_norm + loss_neg*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
            tau_neg = max(0,   min(C_neg,     (loss_neg*x_pos_norm + loss_pos*cross)/ (x_pos_norm*x_neg_norm - cross*cross)  ) )
                
            w = w + tau_pos * X_pos[i_pos,:] - tau_neg *  X_neg[i_neg,:] 

    return w, M