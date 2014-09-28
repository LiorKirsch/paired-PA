'''
Created on Sep 28, 2014

@author: liorlocal
'''

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='fast_sparse',
    ext_modules = cythonize('fast_sparse.pyx'),
)