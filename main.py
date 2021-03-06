# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:49:14 2014

@author: Lior Kirsch
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab

from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search

import predictionMetrics
import multiClassPaPa, pairedPABinaryClassifiers
import pickle
import datasets
import figures

    
def showResults(best_estimator, X_test, y_test, metrics_to_test, results, track_every_n_steps):
    
    print('\t%s  \t\t  (' %(clf.best_params_)),
    for metric_name, metric_to_use in metrics_to_test.iteritems():
        tracking_results, mean_tracking_results = best_estimator.evaulate_tracking(metric_to_use, X_test, y_test)
        
        figures.plot_tracking(tracking_results, track_every_n_steps, 'figures/%s/%s(%d).png' %(metric_name,algo['name'],i) )
#        figures.plot_tracking(mean_tracking_results, track_every_n_steps, 'figures/%s/%s(%d).mean.png' %(metric_name,algo['name'],i) )

        figures.plot_tracking(tracking_results, track_every_n_steps * best_estimator.samples_per_timepoint, 'figures/%s/num_samples_%s(%d).png' %(metric_name,algo['name'],i) )
        
        filename = 'results/%s_%s_%s.pickle' % ( metric_name,algo['name'],i )
        with open(filename, 'wb') as output:
            pickle.dump({'tracking_results': tracking_results, 'mean_tracking_results': mean_tracking_results}, output, pickle.HIGHEST_PROTOCOL)
            
        metric_score = metric_to_use(best_estimator, X_test, y_test)
        print(' %s:%g ' % (metric_name,metric_score)),
        results[ metric_name ].append(metric_score)
        
    
    print(')')
    return results

if __name__ == '__main__':
    
    num_folds = 5
    
    
    
    parms = { 'track_every_n_steps' :100, 'repeat_on_test': 40000,'n_jobs':-1}
    hyper_parms = {'C':[0.01, 0.1, 1,10], 'repeat' : [5000], 'seed' :[0] }

#     parms = { 'track_every_n_steps' :20, 'repeat_on_test': 50,'n_jobs':1}
#     hyper_parms = {'C':[10], 'repeat' : [50], 'seed' :[0] }

    algs = [{'name':'pairedPA1_single_negative', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'choose_single_negative':[True], 'early_stopping' : [1], 'balanced_weight' : [ None, 'samples' ]}.items() + hyper_parms.items() )},
        {'name':'pairedPA1', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [1], 'balanced_weight' : [ None, 'samples' ]}.items() + hyper_parms.items() )},
            {'name':'pairedPA10', 'alg': multiClassPaPa.multiClassPairedPA, 'parameters' : dict( {'early_stopping' : [10], 'balanced_weight' : [None]}.items() + hyper_parms.items() ) },
            {'name':'aucPA', 'alg': multiClassPaPa.oneVsAllAucPA, 'parameters' : hyper_parms },
            {'name':'classicPA', 'alg': multiClassPaPa.oneVsAllClassicPA, 'parameters' : hyper_parms },
           ]
    metrics_to_test = {'AUC':predictionMetrics.oneVsAllAUC, 'ACC':predictionMetrics.accuracy, 'BalancedACC':predictionMetrics.balancedAccuracy}
    
#     algs = [{'name':'pairedPA1', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [1]}.items() + hyper_parms.items() ) },
#         {'name':'pairedPA10', 'alg': pairedPABinaryClassifiers.pairedPA, 'parameters' : dict( {'early_stopping' : [10]}.items() + hyper_parms.items() ) },
#         {'name':'classicPA', 'alg': pairedPABinaryClassifiers.classicPA, 'parameters' : hyper_parms },
#         {'name':'aucPA', 'alg': pairedPABinaryClassifiers.aucPA, 'parameters' : hyper_parms },
#         ]
#     metrics_to_test = {'AUC':predictionMetrics.twoClassAUC, 'ACC':predictionMetrics.accuracy, 'BalancedACC':predictionMetrics.balancedAccuracy}
    
    
    X_all, Y_all = datasets.loadDataSet("20_news_groups", appendOnesColumn=False, seed=0)
    
    subset = (7 < Y_all) & (Y_all < 16)
    X_all = X_all[subset,:]
    Y_all = Y_all[subset]
    
#     figures.showCorrelationBetweenClasses(X_all,Y_all)
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

        print('positives: %d    negatives: %d' %( (y_test > 0).sum(),  (y_test < 0).sum() ))
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

            results[ algo['name'] ] = showResults(best_estimator, X_test, y_test,metrics_to_test, results[ algo['name'] ], track_every_n_steps)
            
        i = i +1
        
        
    for algo in algs:
        print('===== %s =====' % algo['name'])
        for metric in metrics_to_test.keys():
            print('%s: %g (%g)' % (metric, np.mean( results[ algo['name'] ][metric] ), np.std( results[ algo['name'] ][metric] )))
        
        
