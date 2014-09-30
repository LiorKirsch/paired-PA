# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:49:14 2014

@author: liorlocal
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab

from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search

import predictionMetrics
import multiClassPaPa
import pickle
import datasets

def plot_tracking(results, tracking_step,file_name):
    max_val = len(results) * tracking_step
    steps = range(0,max_val, tracking_step)
    fig = plt.figure()
    plt.plot(steps, results)
    plt.plot(steps, [   results[-1]] *len(results),'--')
    plt.axis([0, max_val, 0, 1])
    plt.yticks( list(plt.yticks()[0]) + [results[-1]])
    pylab.savefig(file_name)
    plt.close(fig)
    
def showResults(best_estimator, X_test, y_test, metrics_to_test, results):
    
    print('\t%s  \t\t  (' %(clf.best_params_)),
    for metric_name, metric_to_use in metrics_to_test.iteritems():
        tracking_results, mean_tracking_results = best_estimator.evaulate_tracking(metric_to_use, X_test, y_test)
        plot_tracking(tracking_results, track_every_n_steps, 'figures/%s/%s(%d).png' %(metric_name,algo['name'],i) )
        plot_tracking(mean_tracking_results, track_every_n_steps, 'figures/%s/%s(%d).mean.png' %(metric_name,algo['name'],i) )
        
        metric_score = metric_to_use(best_estimator, X_test, y_test)
        print(metric_score),
        results[ metric_name ].append(metric_score)
        
    
    print(')')
    return results

if __name__ == '__main__':
    
    num_folds = 3
    metrics_to_test = {'AUC':predictionMetrics.oneVsAllAUC, 'ACC':predictionMetrics.accuracy, 'BalancedACC':predictionMetrics.balancedAccuracy}
    
    parms = { 'track_every_n_steps' :100, 'repeat_on_test': 40000,'n_jobs':-2}
    hyper_parms = {'C':[0.01, 0.1, 1,10], 'repeat' : [5000], 'seed' :[0] }

#     parms = { 'track_every_n_steps' :20, 'repeat_on_test': 50,'n_jobs':1}
#     hyper_parms = {'C':[10], 'repeat' : [50], 'seed' :[0] }
    
    algs = [{'name':'pairedPA1', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [1], 'balanced_weight' : [None]}.items() + hyper_parms.items() )},
            {'name':'pairedPA10', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [10], 'balanced_weight' : [None]}.items() + hyper_parms.items() ) },
            {'name':'aucPA', 'alg': multiClassPaPa.oneVsAllAucPA, 'parameters' : hyper_parms },
            {'name':'classicPA', 'alg': multiClassPaPa.oneVsAllClassicPA, 'parameters' : hyper_parms },
           ]

#     algs = [{'name':'pairedPA1', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [1]}.items() + pa_alg_parms.items() ) },
#         {'name':'pairedPA10', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [10]}.items() + pa_alg_parms.items() ) },
#         {'name':'classicPA', 'alg': pairedPABinaryClassifiers.classicPA, 'parameters' : pa_alg_parms },
#         {'name':'aucPA', 'alg': pairedPABinaryClassifiers.aucPA, 'parameters' : pa_alg_parms },
#         ]
    
    
    X_all, Y_all = datasets.loadDataSet("20_news_groups", appendOnesColumn=False)
    skf = StratifiedKFold(Y_all, num_folds)
    
    results_balanced = {}
    results_auc = {}
    results_accuracy = {}
    results = {}
    for algo in algs:
        results[ algo['name'] ] = {}
        for metric in metrics_to_test.keys():
            results[ algo['name'] ][metric] = []
        
    
    i = 0
    for train, test in skf:
        
        X_train = X_all[train,:]
        y_train = Y_all[train]
        X_test = X_all[test,:]
        y_test = Y_all[test]

        validationCV = StratifiedKFold(y_train, num_folds)
        
        for algo in algs:
            alg = algo['alg']()
            parameters = algo['parameters']
    
            print('running %s (%d):  ' %(algo['name'],i) ) 
            clf = grid_search.GridSearchCV(alg, parameters, cv=validationCV, scoring= predictionMetrics.balancedAccuracy, refit=False, n_jobs=parms[ 'n_jobs'])
            clf.fit(X_train, y_train)
            
            track_every_n_steps = parms[ 'track_every_n_steps']
            clf.best_params_['repeat'] = parms[ 'repeat_on_test']
            best_estimator = algo['alg'](**clf.best_params_)
            best_estimator.fit(X_train, y_train, track_every_n_steps = track_every_n_steps)
            y_predictions = best_estimator.predict(X_test)
            
#             with open('results/weights_%s_%d.pickle' % (algo['name'],i), 'wb') as handle:
#                 pickle.dump(best_estimator, handle)

            results[ algo['name'] ] = showResults(best_estimator, X_test, y_test,metrics_to_test, results[ algo['name'] ])
            
        i = i +1
        
        
    for algo in algs:
        print('===== %s =====' % algo['name'])
        for metric in metrics_to_test.keys():
            print('%s: %g (%g)' % (metric, np.mean( results[ algo['name'] ][metric] ), np.std( results[ algo['name'] ][metric] )))
        
        