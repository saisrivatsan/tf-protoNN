# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.sparse as ss

def normalize(a, order = 2, axis=-1):
    """ Normalize numpy array. Order= "inf" for infinity norm """
    
    if order is 'inf':
        norm = a.max(axis)
    else:
        norm = np.atleast_1d(np.linalg.norm(a, order, axis))
        
    norm[norm==0] = 1.0
    return a / np.expand_dims(norm, axis)

def smat_to_dmat(X):
    """ Converts sparse scipy matrix to dense numpy array """
    return np.asarray(X.todense())
    
def dmat_to_smat(X):
    """ Converts dense numpy array to csr_matrix """ 
    return ss.csr_matrix(X)

def smat_to_sparseTensor(X): 
    """ Converts sparse matrix to sparse tensor """
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)

def data_loader(path, config):
    """ Loads data @ path. Cross-checks with configs
    Args:
        path: location of .mat file to be loaded
        config: config dict corresponding to the current experiment run
    Retuns:
        X: (N, D) numpy array/scipy matrix
        Y: (N, L) sparse csr_matrix
    """

    data = sio.loadmat(path)    
    X = data['X'].T.astype(np.float32)
    Y = data['Y'].T.astype(np.float32)

    " Asserts to check data is loaded correctly "
    assert(len(X.shape) == 2), "dim(X) != 2" 
    assert(len(Y.shape) == 2), "dim(Y) != 2"
    assert(X.shape[1] == config.D), "Number of features is X doesn't match config.D"
    assert(Y.shape[0] == X.shape[0]), "Number of instances in X != number of test instances in Y"
    assert(Y.shape[1] == config.L), "Number of labels in Y doesn't match config.L"

    " Convert to the correct format if required "
    if config.is_sparse_input:
        if not ss.issparse(X): X = dmat_to_smat(X)
    else:
        if ss.issparse(X): X = smat_to_dmat(X)

    # Output is always sparse
    if not ss.issparse(Y): Y = dmat_to_smat(Y)

    " Normalize data if required "
    if config.normalize: X = normalize(X)

    return X, Y


class TrainGenerator:  
    def __init__(self, config):        
        self.config = config
        self.batch_size = self.config.train_batch_size
        self.path = os.path.join(self.config.dir_name, "train_data")
        self.get_data()
        self.train_gen = self.train_generator()
        self.val_gen = self.val_generator()
        
    def get_data(self):
        " Gets train data and creates train and test splits "       
        X, Y = data_loader(self.path, self.config)      
        self.N = int(self.config.N * self.config.train_val_split_ratio)
        self.Nv = self.config.N - self.N
        self.X = X[:self.N]
        self.X_v = X[self.N:] 
        self.Y = Y[:self.N]
        self.Y_v = Y[self.N:]
                   
    def train_generator(self):        
        """ Train generator function """
        i, perm = 0, np.random.permutation(self.N)       
        while True:                
            idx = perm[i * self.batch_size: (i + 1) * self.batch_size]
            X_batch = smat_to_sparseTensor(self.X[idx]) if self.config.is_sparse_input else self.X[idx]
            yield X_batch, smat_to_sparseTensor(self.Y[idx]), self.Y[idx] 
            i += 1
            # We ignore the last few indices. Reset and shuffle indices
            if( (i+1) * self.batch_size  > self.N): i, perm = 0, np.random.permutation(self.N)
                                                    
    def val_generator(self):
        """ Val generator function """
        i = 0            
        while True:           
            X_batch = self.X_v[i * self.batch_size: (i + 1) * self.batch_size]
            Y_batch = self.Y_v[i * self.batch_size: (i + 1) * self.batch_size]
            if self.config.is_sparse_input: X_batch = smat_to_sparseTensor(X_batch)
            yield X_batch, smat_to_sparseTensor(Y_batch), Y_batch            
            i += 1
            # We don't ignore the last few indices. Reset and shuffle indices for next call
            if( i * self.batch_size  > self.Nv): i = 0
                
    def reset_generator(self):
        """ Resets generator functions """
        self.train_gen = self.train_generator()
        self.val_gen = self.val_generator()
                
                
class TestGenerator:   
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.test_batch_size
        self.path = os.path.join(self.config.dir_name, "test_data")
        self.get_data()
        self.test_gen = self.test_generator()
        
    def get_data(self):
        """ Loads test data """      
        self.X, self.Y = data_loader(self.path, self.config)      
                                                    
    def test_generator(self):
        """ Test Generator function """
        i = 0            
        while True:           
            X_batch = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            Y_batch = self.Y[i * self.batch_size: (i + 1) * self.batch_size]
            if self.config.is_sparse_input: X_batch = smat_to_sparseTensor(X_batch)
            yield X_batch, smat_to_sparseTensor(Y_batch), Y_batch            
            i += 1
            # We don't ignore the last few indices. Reset and shuffle indices for next call
            if( i * self.batch_size  > self.config.Nt): i = 0

                