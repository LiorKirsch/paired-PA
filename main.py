# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:49:14 2014

@author: liorlocal
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import multi_step_passive_aggressive
import passive_aggressive_varients
import onlineClassifiers
import scipy.stats as stats


import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm, grid_search, datasets, metrics, linear_model


import pairedPAClassifiers
# Generate data, run cross validation with hyperparms tuning

def balancedAccuracy(estimator, X, y_true):
    y_pred = estimator.predict(X)
    pos = y_true > 0
    return  0.5 * metrics.accuracy_score( y_true[pos], y_pred[pos]) + 0.5 * metrics.accuracy_score( y_true[~pos], y_pred[~pos])
   
if __name__ == '__main__':
    
    num_folds = 5
    
    pa_alg_parms = {'C':[0.1,1,10], 'repeat' : [500], 'seed' :[0] }
    algs = [{'name':'pairedPA', 'alg': pairedPAClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [10]}.items() + pa_alg_parms.items() ) },
            {'name':'classicPA', 'alg': pairedPAClassifiers.classicPA, 'parameters' : pa_alg_parms },
            {'name':'aucPA', 'alg': pairedPAClassifiers.aucPA, 'parameters' : pa_alg_parms },
            ]
    
    
    iris = datasets.load_iris()
    iris.data.shape, iris.target.shape

    X_all = np.array(iris.data) 
    Y_all = np.array(iris.target)
    
    X_all = np.append( X_all, np.ones( (X_all.shape[0] ,1) ),1 )    # add a 1 coulmn for the bias
    first_class =  Y_all == 1
    Y_all[ first_class ] = 1
    Y_all[ ~first_class ] = -1
    
    skf = StratifiedKFold(Y_all, num_folds)
    
    results_balanced = {}
    results_auc = {}
    results_accuracy = {}
    for algo in algs:
        results_balanced[ algo['alg'] ] = np.empty(num_folds) * np.nan
        results_auc[ algo['alg'] ] = np.empty(num_folds) * np.nan
        results_accuracy[ algo['alg'] ] = np.empty(num_folds) * np.nan
    
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
    
            clf = grid_search.GridSearchCV(alg, parameters, cv=validationCV, scoring= balancedAccuracy )
            print('running grid search...\n')
            clf.fit(X_train, y_train)
            
            clf.score(X_test, y_test)
            y_predictions = clf.best_estimator_.predict(X_test)
            results_auc[ algo['alg'] ][i] = metrics.roc_auc_score(y_test, clf.best_estimator_.decision_function(X_test))
            results_accuracy[ algo['alg'] ][i] = metrics.accuracy_score(y_test, y_predictions)
            results_balanced[ algo['alg'] ][i] = balancedAccuracy(clf.best_estimator_, X_test, y_test)
                               
        i = i +1
        
    for algo in algs:
        print('===== %s =====' % algo['name'])
        print('accuracy: %g (%g)' % ( np.mean( results_accuracy[ algo['alg'] ] ), np.std( results_accuracy[ algo['alg'] ] )))
        print('auc: %g (%g)' % ( np.mean( results_auc[ algo['alg'] ] ), np.std( results_auc[ algo['alg'] ] )))
        print('balanced acc: %g (%g)' % ( np.mean( results_balanced[ algo['alg'] ] ), np.std( results_balanced[ algo['alg'] ] )))

#    X,Y, w_opt = generate_data()
    #%%