'''
Created on Sep 28, 2014

@author: lior

Before compiling make sure to install python-dev libxml2-dev libxslt-dev:
sudo apt-get install python-dev libxml2-dev libxslt-dev

cython fast_sparse.pyx
python setup.py build_ext --inplace

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -lm -I/usr/include/python2.7/ -o fast_sparse.so fast_sparse.c

'''


import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
def t_dot( np.ndarray[int, ndim=1] x_indices, 
           np.ndarray[np.float64_t, ndim=1] x_data,
           np.ndarray[np.float64_t, ndim=1] y):   
    """
    Dot product of the transpose.
    """
    #Using this type is important, this gives us exactly the same values as it
    #will give exact same values as numpy's equivalent X*W.T
    cdef np.float64_t sum
    cdef int idx, ptr
    sum = 0.0
    for idx, ptr in enumerate(x_indices):
        sum += x_data[idx] * y[ptr]    
    return sum

@cython.boundscheck(False)
def update_w( np.ndarray[int, ndim=1] x_indices, 
           np.ndarray[np.float64_t, ndim=1] x_data,
           np.ndarray[np.float64_t, ndim=1] w,
           float tau, int y):
    
    for idx, ptr in enumerate(x_indices):
        w[ptr] += tau*y*x_data[idx]    
    
    return w
    
