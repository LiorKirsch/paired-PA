'''
Created on Sep 30, 2014

@author: Lior Kirsch
'''

from __future__ import division
from sklearn import svm, grid_search, datasets, metrics, linear_model
import numpy as np
import scipy.io
import scipy.sparse as sparse

def loadDataSet(dataset_name, appendOnesColumn=False, seed = 0):
    
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
    
    elif dataset_name == "synthetic":
        X_all, Y_all = generate_data(seed = seed)
        
    samples_classes = np.unique(Y_all) 
    print('loaded %s: %d sample, %d features, %d classes' % (dataset_name, X_all.shape[0], X_all.shape[1] , len(samples_classes) ))
    return X_all , Y_all






def generate_data(seed, m = 1000000, d=2, positive_neg_ratio = 0.00005, flip_percentage = 0, gaussian_noise_std = 0.7):
    ''' 
    generate data by choosing a classification vector and then generate random samples 
    and apply w to these samples to get the correct label.
    To add noise: 
        the labels of some of the samples are flipped.
        a gaussian noise is added to the features.
    The percent of positive sample can be controlled to create an unbalanced dataset
    '''   
    
    np.random.seed(seed)
    w_opt = np.random.rand(1,d)  #  np.array([1.0,2.0])
    # random numbers in [0,1]
    X = np.random.rand(m,d)  
    # convert to numbers in [-1,1]
    X = 2.0*X - 1.0
    # set the label
    Y  = np.zeros((m,))
 
    indices_to_delete = list()
    for i in range(m):
        Y[i] = np.sign(np.dot(w_opt, X[i,:]))
#        # set margin
#        if Y[i]*np.dot(w_opt, X[i,:]) < 0.5:
#            indices_to_delete.append(i)
#     
    X = np.delete(X, indices_to_delete, 0)
    Y = np.delete(Y, indices_to_delete, 0)
    m, d = X.shape;  # updated number of examples
    
    # add flip noise
    if flip_percentage > 0.0:
        flip_every_n_steps = round(flip_percentage * m)
        for i in range(m):
            if i % flip_every_n_steps == 0:
                Y[i] = -Y[i]
    
    # add gaussian noise
    X = X + np.random.normal(0, gaussian_noise_std, X.shape)
    
    
    # make an unbalanced dataset by removing samples from the positive set
    pos_ind = Y > 0
    num_pos = pos_ind.sum()
    num_neg = m - num_pos
    
    percent_pos_to_remove = 1 - positive_neg_ratio 
    remove_ind = np.random.rand(m) < percent_pos_to_remove
    to_remove = remove_ind & pos_ind 
    X = X[~to_remove,:]
    Y = Y[~to_remove]
    
    pos_neg_ratio = float( (Y > 0).sum()) / len(Y) 
    new_num_pos = (Y > 0).sum()
    print('positives: %d , negatives: %d' %(new_num_pos, num_neg ))
    print( 'positive ratio in data - %f' % pos_neg_ratio )
    
    return (X,Y)

