'''
Created on Sep 22, 2014

@author: liorlocal
'''
from __future__ import division
import numpy as np
from pylab import *

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
        
        