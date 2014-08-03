# -*- coding: utf-8 -*-
'''
Created on Jul 8, 2014

@author: lior
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import multi_step_passive_aggressive
import passive_aggressive_varients
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
    percent_to_remove = 0.1
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

def normalizeData(X):
    n, d = X.shape;
    for j in range(0, n):
        # choose example
        
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
    repeat = 10000
    
    # show some images
    plt.figure(1);
    index_pos = Y > 0
    index_neg = Y < 0
    plt.plot(X[index_neg, 0], X[index_neg, 1], 'gs')
    plt.plot(X[index_pos, 0], X[index_pos, 1], 'yo')
    p_opt = plotW(w_opt, 'g-')
    
    X = normalizeData(X)
     
     
    algs = [{'alg_func': onlineClassifiers.svm_dual,    'alg_name':'svm DCA','line_properties':'r-','args':{'C':1.0,'repeat':repeat}},
            {'alg_func': onlineClassifiers.svm_dual_double_step, 'alg_name':'svm double step','line_properties':'r--','args':{'C':1.0,'repeat':repeat} },
            {'alg_func': multi_step_passive_aggressive.pairedPA_exact, 'alg_name':'PA dual exact','line_properties':'','args':{'C':1.0,'repeat':repeat} },
            {'alg_func': multi_step_passive_aggressive.pairedPA_exact, 'alg_name':'PA sequantial','line_properties':'','args':{'C':1.0,'repeat':repeat,'early_stopping':1} },
            {'alg_func': multi_step_passive_aggressive.pairedPA_one_loss, 'alg_name':'PA one loss','line_properties':'','args':{'C':1.0,'repeat':repeat} },
            {'alg_func': passive_aggressive_varients.PA, 'alg_name':'PA','line_properties':'b-','args':{'C':1.0,'repeat':repeat} },
            {'alg_func': passive_aggressive_varients.PA_auc, 'alg_name':'PA auc','line_properties':'g--','args':{'C':1.0,'repeat':repeat} },
            {'alg_func': passive_aggressive_varients.pairedPA_projected_constrains, 'alg_name':'PA projected_constrains','line_properties':'','args':{'C_pos':1.0,'C_neg':1.0,'repeat':repeat} },
            ]


    plots = []
    plots_info = []
    
    for alg in algs:
        func = alg['alg_func']
        args = alg['args']
        w = func(X,Y, **args)
        showOutput(X,Y,w,  alg['alg_name']  )
        plots.append(    plotW(w, alg['line_properties'])    )
        plots_info.append( "%s %g" %( alg['alg_name'] , calc_AUC( predict(X,w) , Y > 0 ))  )
        
    
    plt.legend(plots, plots_info)
    plt.draw()
    plt.show() 
    
    #%%    