'''
Created on Sep 15, 2014

@author: lior
'''

from sklearn import metrics
import numpy as np

def balancedAccuracy(estimator, X, y_true):
    num_classes = len(estimator.classes_)
    y_pred = estimator.predict(X)
    accuracy_in_class = np.ndarray( num_classes )
    for j in range(num_classes):
        current_class_indcies = y_true == estimator.classes_[j]
        accuracy_in_class[j] =  metrics.accuracy_score( y_true[current_class_indcies], y_pred[current_class_indcies])
    
    return  np.mean(accuracy_in_class)



def oneVsAllAUC(estimator, X, y_true):
    num_classes = len(estimator.classes_)
    num_samples,d = X.shape
    y_descisions = estimator.decision_function(X)
    assert( y_descisions.shape == (num_samples,num_classes), 'in oneVsAll estimator.decision_function(X) should be a matrix of shape (num_samples,num_classes)')
    
    auc_in_class = np.ndarray( num_classes )
    for j in range(num_classes):
        current_classifier_descision = y_descisions[:,j]
        current_class_indcies = y_true == estimator.classes_[j]
        current_labels = -1 * np.ones(num_samples)
        current_labels[current_class_indcies] = 1
        auc_in_class[j] = metrics.roc_auc_score( current_labels, current_classifier_descision )
    
    return  np.mean(auc_in_class)
