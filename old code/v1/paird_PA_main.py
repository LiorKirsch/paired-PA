# -*- coding: utf-8 -*-
'''
Created on Jul 8, 2014

@author: lior
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import passive_aggresive
import onlineClassifiers
import scipy.stats as stats




def generate_data():
    # generate data
    m = 10000
    d = 2
    desired_num_pos = 10
    w_opt = np.array([1.0,2.0])
    # random numbers in [0,1]
    X = np.random.rand(m,d)  
    # convert to numbers in [-1,1]
    X = 2.0*X - 1.0
    # set the label
    Y  = np.zeros((m,))
    num_pos = 0
    indices_to_delete = list()
    for i in range(m):
        Y[i] = np.sign(np.dot(w_opt, X[i,:]))
#        # set margin
#        if Y[i]*np.dot(w_opt, X[i,:]) < 0.5:
#            indices_to_delete.append(i)
#     
    X = np.delete(X, indices_to_delete, 0)
    Y = np.delete(Y, indices_to_delete, 0)
    m, d = X.shape;  # updated number of examples
    
    # add noise
    for i in range(m):
        if i % 10 == 0:
            Y[i] = -Y[i]
    
    
    # make unbalance remove samples from the positive set
    percent_to_remove = 0.99
    pos_ind = Y > 0
    remove_ind = np.random.rand(m) < percent_to_remove
    to_remove = remove_ind & pos_ind 
    X = X[~to_remove,:]
    Y = Y[~to_remove,:]
    
        
    
    return (X,Y, w_opt)




def plotW(w, colorString):
    p_plot = plt.plot([0, 1, -1],[0, -w[0]/w[1], w[0]/w[1]], colorString)
    return p_plot


def calc_AUC(scores, true_labels):
    
    num_samples = float(len(scores)) 
    positive_labels = true_labels > 0
    num_pos = sum(positive_labels)
    num_neg = num_samples - num_pos
    all_ranks = stats.rankdata(  scores  )
    pos_ranks = all_ranks[positive_labels]
    bigger_than_neg = pos_ranks.sum() - num_pos * (num_pos + 1.0) / 2.0
    auc = bigger_than_neg / (num_pos * num_neg)
    
    return auc

def normalizeX(X):
    
    n, d = X.shape    
    for j in range(0, n):
        X[j,:] = X[j,:] / np.linalg.norm( X[j,:] )
         
    return X    
        
def predict(X,w):
    return np.array(   np.dot(   np.matrix(X), np.matrix(w).transpose() ) )

def showOutput(X,Y,w,name):
    prediction = np.array(   np.dot(   np.matrix(X), np.matrix(w).transpose() ) )
    pos_ind = Y  > 0
    neg_ind = ~pos_ind
    auc = calc_AUC( predict(X,w) , pos_ind)
    
    pos_correct = (prediction[pos_ind] > 0).sum() / pos_ind.sum()
    neg_correct = (prediction[neg_ind] < 0).sum() / neg_ind.sum()
    
    stringToPrint = '=== %s  \t\t pos %g,  neg %g, auc %g' % (name, pos_correct, neg_correct, auc)
    print(stringToPrint)
    
    
if __name__ == '__main__':
    
    X,Y, w_opt = generate_data()
    #%%
    repeat = 100000
    
    # show some images
    plt.figure(1);
    index_pos = Y > 0
    index_neg = Y < 0
    plt.plot(X[index_neg, 0], X[index_neg, 1], 'gs')
    plt.plot(X[index_pos, 0], X[index_pos, 1], 'yo')
    p_opt = plotW(w_opt, 'g-')
     
     
    X = normalizeX(X)
   #%%
           
    C_pos = 1.0
    C_neg = 1.0
    w_pairedPA_new , pos_pairedPA_new, neg_pairedPA_new = passive_aggresive.pairedPA_new(X, Y, C_pos, C_neg, repeat)
    #p_pairedPA_new = plotW(w_pairedPA_new, 'b.')
    showOutput(X,Y,w_pairedPA_new,'pairedPA_new')
    auc_pairedPA2 = calc_AUC(predict(X, w_pairedPA_new) , Y > 0)
    
    
   #%%    
    
    C = 1.0
    w_svm = onlineClassifiers.svm_dual(X, Y, C, repeat)
    p_svm = plotW(w_svm, 'r-')
    showOutput(X,Y,w_svm,'svm')
    auc_svm = calc_AUC( predict(X,w_svm) , Y > 0 )
   #%%    
    C = 1.0
    w_svm_pairs = onlineClassifiers.svm_dual_pairs(X, Y, C, repeat)
    p_svm_pairs = plotW(w_svm_pairs, 'r-')
    showOutput(X,Y,w_svm_pairs,'svm pairs')
    auc_svm_pairs = calc_AUC( predict(X,w_svm_pairs) , Y > 0 )
   #%%
    
    C = 1.0
    w_PA, M_PA = passive_aggresive.PA(X, Y, C, repeat)
    p_PA = plotW(w_PA, 'p--')
    showOutput(X,Y,w_PA,'PA ')
    auc_PA = calc_AUC(predict(X, w_PA) , Y > 0)
    
    #%%    
    
    C1 = 1.0
    C2 = 1.0
    w_PA_auc, M_PA_auc = passive_aggresive.PA_auc(X, Y, C1, repeat)
    p_PA_auc = plotW(w_PA_auc, 'g--')
    showOutput(X,Y,w_PA_auc,'PA auc')
    auc_PA_auc = calc_AUC(predict(X, w_PA_auc) , Y > 0)
    
    #%%    
    
    w_perceptron = onlineClassifiers.perceptron(X,Y, repeat)
    p_perceptron = plotW(w_perceptron, 'y-')
    showOutput(X,Y,w_perceptron,'percept')
    auc_perceptron = calc_AUC(predict(X, w_perceptron) , Y > 0)
    #%%
    
    C_pos = 1.0
    C_neg = 1.0
    w_pairedPA , M_paired = passive_aggresive.pairedPA(X, Y, C_pos, C_neg, repeat)
    p_pairedPA = plotW(w_pairedPA, 'b-')
    showOutput(X,Y,w_pairedPA,'pairedPA')
    auc_pairedPA = calc_AUC(predict(X, w_pairedPA) , Y > 0)

    C_pos = 1.0
    C_neg = 1.0
    w_pairedPA2 , M_paired2 = passive_aggresive.pairedPA2(X, Y, C_pos, C_neg, repeat)
    p_pairedPA2 = plotW(w_pairedPA2, 'b.')
    showOutput(X,Y,w_pairedPA2,'pairedPA2')
    auc_pairedPA2 = calc_AUC(predict(X, w_pairedPA2) , Y > 0)
    
 
    
    plt.legend([p_opt, p_svm, p_svm_pairs, p_perceptron, p_PA, p_PA_auc, p_pairedPA, p_pairedPA2], ["original"  , "SVM %g" % auc_svm, "SVM pairs %g" % auc_svm_pairs, "perceptron %g" % auc_perceptron,"PA %g" % auc_PA ,"PA_auc %g" % auc_PA_auc,"paired PA %g" % auc_pairedPA, "paired PA2 %g" % auc_pairedPA2])
    plt.draw()
    plt.show() 
    
    #%%    