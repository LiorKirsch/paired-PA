# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:49:14 2014

@author: liorlocal
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm, grid_search, datasets, metrics, linear_model

import predictionMetrics
import pairedPABinaryClassifiers
import multiClassPaPa
# Generate data, run cross validation with hyperparms tuning
import scipy.io
import scipy.sparse as sparse


def loadDataSet(dataset_name, appendOnesColumn=False):
    
    if dataset_name == "iris":
        '''
        --------- Multiclass - IRIS dataset -------------
        contains two variables:
        data: holds the data matrix
        target:   a number between that indicates the class 0,1,2

        '''
        iris = datasets.load_iris()
        iris.data.shape, iris.target.shape
    
        X_all = np.array(iris.data) 
        Y_all = np.array(iris.target)
        if appendOnesColumn:
            X_all = np.append( X_all, np.ones( (X_all.shape[0] ,1) ),1 )    # add a 1 coulmn for the bias

    elif dataset_name == "reuters":
        '''
        --------- Multilabel - Reuters dataset -------------
        contains two variables:
        target: holds a 2000 x 7 matrix which hold the class of the object (Multilabel , 7 classes)
        bags:   a cell array of size 2000, in each cell there is a different number of bag-of-words vector of size 243

        '''
        
        reutersFile = 'datasets/Reuters-21578.mat'
        mat = scipy.io.loadmat(reutersFile)
        bag_of_words = mat['bags']
        belongs_to = mat['target'] == 1
        if appendOnesColumn:
            X_all = np.append( X_all, np.ones( (X_all.shape[0] ,1) ),1 )    # add a 1 coulmn for the bias
#         X_all = np.array(iris.data) 
#         Y_all = np.array(iris.target)
        
    elif dataset_name == "20_news_groups":
        '''
        --------- Multiclass - 20 news groups dataset -------------
        contains two variables:
        Xtfidf_normalized: holds tfidf normalized representation of the data (sparse)
        Y:   the labels, an integer 1 to 20 representing the news group the article was taken from

        '''
        
        matfile = 'datasets/20_newsgroups_50Kfeatures.mat'
        mat = scipy.io.loadmat(matfile)
        X_all = mat['Xtfidf_normalized']
        Y_all = mat['Y'].flatten()   # flatten the (n,1) matrix to a (n,) vector
        if appendOnesColumn:
            ones_column = sparse.csr_matrix( np.ones( (X_all.shape[0] ,1) ) )
            X_all = sparse.hstack( [X_all,ones_column] )
    
    return X_all , Y_all
    
if __name__ == '__main__':
    
    num_folds = 5
    
    pa_alg_parms = {'C':[0.01,0.1,1,10], 'repeat' : [5000], 'seed' :[0] }
    algs = [{'name':'pairedPA1', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [1], 'balanced_weight' : ['samples','problem',None]}.items() + pa_alg_parms.items() )},
            {'name':'pairedPA10', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [10], 'balanced_weight' : ['samples','problem',None]}.items() + pa_alg_parms.items() ) },
            {'name':'classicPA', 'alg': multiClassPaPa.oneVsAllClassicPA, 'parameters' : pa_alg_parms },
            {'name':'aucPA', 'alg': multiClassPaPa.oneVsAllAucPA, 'parameters' : pa_alg_parms },
            ]
    
#     algs = [{'name':'pairedPA1', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [1]}.items() + pa_alg_parms.items() ) },
#         {'name':'pairedPA10', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [10]}.items() + pa_alg_parms.items() ) },
#         {'name':'classicPA', 'alg': pairedPABinaryClassifiers.classicPA, 'parameters' : pa_alg_parms },
#         {'name':'aucPA', 'alg': pairedPABinaryClassifiers.aucPA, 'parameters' : pa_alg_parms },
#         ]
    
    
    X_all, Y_all = loadDataSet("20_news_groups", appendOnesColumn=False)
    
#     first_class =  Y_all == 1
#     Y_all[ first_class ] = 1
#     Y_all[ ~first_class ] = -1
#     
    skf = StratifiedKFold(Y_all, num_folds)
    
    results_balanced = {}
    results_auc = {}
    results_accuracy = {}
    for algo in algs:
        results_balanced[ algo['name'] ] = np.empty(num_folds) * np.nan
        results_auc[ algo['name'] ] = np.empty(num_folds) * np.nan
        results_accuracy[ algo['name'] ] = np.empty(num_folds) * np.nan
    
    i = 0
    for train, test in skf:
        
        X_train = X_all[train,:]
        y_train = Y_all[train]
        X_test = X_all[test,:]
        y_test = Y_all[test]

        validationCV = StratifiedKFold(y_train, num_folds)
        
        for algo in algs:
            #alg = linear_model.PassiveAggressiveClassifier()
            #alg = svm.SVC()
           
            alg = algo['alg']()
            parameters = algo['parameters']
    
            print('running %s (%d):  ' %(algo['name'],i) ) 
            clf = grid_search.GridSearchCV(alg, parameters, cv=validationCV, scoring= predictionMetrics.balancedAccuracy, n_jobs=-2)
            clf.fit(X_train, y_train)

            clf.score(X_test, y_test)
            y_predictions = clf.best_estimator_.predict(X_test)
            results_accuracy[ algo['name'] ][i] = metrics.accuracy_score(y_test, y_predictions)
            results_auc[ algo['name'] ][i] = predictionMetrics.oneVsAllAUC(clf.best_estimator_, X_test, y_test) 
            results_balanced[ algo['name'] ][i] = predictionMetrics.balancedAccuracy(clf.best_estimator_, X_test, y_test)
            print('\t%s  \t\t  ( %g, %g, %g)' %(clf.best_params_, results_accuracy[ algo['name'] ][i], results_auc[ algo['name'] ][i], results_balanced[ algo['name'] ][i]) )                        
        i = i +1
        
    for algo in algs:
        print('===== %s =====' % algo['name'])
        print('accuracy: %g (%g)' % ( np.mean( results_accuracy[ algo['name'] ] ), np.std( results_accuracy[ algo['name'] ] )))
        print('auc: %g (%g)' % ( np.mean( results_auc[ algo['name'] ] ), np.std( results_auc[ algo['name'] ] )))
        print('balanced acc: %g (%g)' % ( np.mean( results_balanced[ algo['name'] ] ), np.std( results_balanced[ algo['name'] ] )))

#    X,Y, w_opt = generate_data()
    #%%