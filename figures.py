'''
Created on Sep 22, 2014

@author: liorlocal
'''
from __future__ import division
import numpy as np
from pylab import *
import predictionMetrics, pickle

def showClassesPie(Y):
    num_samples = len(Y)
    classes_names, Y = np.unique(Y, return_inverse=True) 
    num_classes = len(classes_names)
    percent = np.zeros(num_classes, np.double)
    for j in range(num_classes):
        percent[j] = ( Y == j ).sum()
        
    
    percent = percent / num_samples
    drawPie(percent, classes_names)
    

def drawPie(fracs, labels):    
    """
    Make a pie chart - see
    http://matplotlib.sf.net/matplotlib.pylab.html#-pie for the docstring.
    
    This example shows a basic pie chart with labels optional features,
    like autolabeling the percentage, offsetting a slice with "explode",
    adding a shadow, and changing the starting angle.
    
    """
    
    
    # make a square figure and axes
    figure(1, figsize=(6,6))
    ax = axes([0.1, 0.1, 0.8, 0.8])
    
    pie(fracs, labels=labels,
                    autopct='%1.1f%%', shadow=True, startangle=90)
                    # The default startangle is 0, which would start
                    # the Frogs slice on the x-axis.  With startangle=90,
                    # everything is rotated counter-clockwise by 90 degrees,
                    # so the plotting starts on the positive y-axis.
    
        
    show()
        
def plot_tracking(results, tracking_step,file_name):
    max_val = len(results) * tracking_step
    steps = range(0,max_val, tracking_step)
    fig = plt.figure()
    plt.plot(steps, results, linestyle='-', marker='.')
    plt.plot(steps, [   results[-1]] *len(results),'--')
    plt.axis([0, max_val, 0, 1])
    plt.yticks( list(plt.yticks()[0]) + [results[-1]])
    savefig(file_name)
    plt.close(fig)
    
def createFigureFromResults(algs,x_axis_label):
    
    num_folds = 5
    
    
    tracking_step = 100
    metrics_to_test = {'AUC':predictionMetrics.oneVsAllAUC, 'ACC':predictionMetrics.accuracy, 'BalancedACC':predictionMetrics.balancedAccuracy}

    for i in range(num_folds):
        for metric in metrics_to_test.keys():
            fig = plt.figure()
    
            for algo in algs:        
                filename = 'results/%s_%s_%d.pickle' % ( metric,algo['name'],i )
                with open(filename, 'r') as output:
                    fileData = pickle.load(output)
                tracking_results = fileData['tracking_results']
                
                num_samples_steps = tracking_step * algo['samples_at_each_step']
                max_val = len(tracking_results) * num_samples_steps
                steps = range(0,max_val, num_samples_steps)
                
                plt.plot(steps, tracking_results, linestyle='-', marker='.', color= algo['plot_color'], label=algo['name'])
                plt.axis([0, max_val, 0.5, 1])
            
            legend(loc='lower right')
            plt.xlabel(x_axis_label)
            plt.ylabel(metric)
            file_name = 'figures/%s_%s%d.png' % (x_axis_label,metric, i)
            savefig(file_name)
            plt.close(fig)
    
            
            
def showCorrelationBetweenClasses(X,Y):
    n, d = X.shape
    classes, Y = np.unique(Y, return_inverse=True)
    num_classes = len(classes)
#         classes_corr = np.zeros( (num_classes,num_classes) )
    data  = np.zeros( (num_classes,d) )
     
    for i in range(num_classes):
        in_i = (Y == i)
#             mean_i = X[in_i,:].mean()
#         data[i,:] = np.mean(X[in_i,:], axis=0) 
#         
        current_X = X[in_i,:]
        data[i,:] = current_X.data.mean() 
        


#             for j in range(num_classes):
#                 in_j = (Y == j)
#                 mean_j = X[in_j,:].mean()
#                 classes_corr[i,j] = np.dot(mean_i,mean_j)/norm(mean_j)/norm(mean_j)
#                 classes_corr[j,i] = classes_corr[i,j]
    
    fig = plt.figure()
    R = corrcoef(data)
    pcolor(R)
    colorbar()
    yticks(arange(0.5,num_classes + 0.5),range(num_classes))
    xticks(arange(0.5,num_classes + 0.5),range(num_classes))
    plt.show()        
    plt.close(fig)  
        
                

if __name__ == '__main__':
    num_classes = 2   
    algs = [{'name':'classicPA' ,'samples_at_each_step':1,'plot_color':'y'},
            {'name':'pairedPA1' ,'samples_at_each_step':num_classes,'plot_color':'c'},
#             {'name':'pairedPA1_single_negative', 'samples_at_each_step':2,'plot_color':'b'},
            {'name':'pairedPA10', 'samples_at_each_step':num_classes,'plot_color':'b'},
            {'name':'aucPA' , 'samples_at_each_step':2,'plot_color':'g'},
           ] 
    createFigureFromResults(algs, 'sample seen')    
                
    
    
    algs = [{'name':'classicPA' ,'samples_at_each_step':1,'plot_color':'y'},
            {'name':'pairedPA1' ,'samples_at_each_step':1,'plot_color':'c'},
#             {'name':'pairedPA1_single_negative', 'samples_at_each_step':1,'plot_color':'b'},
            {'name':'pairedPA10', 'samples_at_each_step':1,'plot_color':'b'},
            {'name':'aucPA' , 'samples_at_each_step':1,'plot_color':'g'},
           ] 
    createFigureFromResults(algs, 'time steps')    
        
    
    algs = [{'name':'classicPA' ,'samples_at_each_step':1,'plot_color':'y'},
            {'name':'pairedPA1' ,'samples_at_each_step':num_classes,'plot_color':'c'},
#             {'name':'pairedPA1_single_negative', 'samples_at_each_step':2,'plot_color':'b'},
            {'name':'pairedPA10', 'samples_at_each_step':num_classes,'plot_color':'b'},
            {'name':'aucPA' , 'samples_at_each_step':1,'plot_color':'g'},
           ] 
    createFigureFromResults(algs, 'updates to w')                