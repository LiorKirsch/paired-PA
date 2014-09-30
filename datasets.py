'''
Created on Sep 30, 2014

@author: liorlocal
'''

from sklearn import svm, grid_search, datasets, metrics, linear_model
import numpy as np
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
        X_all =  mat['Xtfidf_normalized'] # matlab default sparse type format is using column wise indexing csc
        X_all = sparse.csr_matrix(X_all)  # change to csr type of sparse because we access we need access to entire rows
        Y_all = mat['Y'].flatten()   # flatten the (n,1) matrix to a (n,) vector
        
        if appendOnesColumn:
            ones_column = sparse.csr_matrix( np.ones( (X_all.shape[0] ,1) ) )
            X_all = sparse.hstack( [X_all,ones_column] )
    
    samples_classes = np.unique(Y_all) 
    print('loaded %s: %d sample, %d features, %d classes' % (dataset_name, X_all.shape[0], X_all.shape[1] , len(samples_classes) ))
    return X_all , Y_all