# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def create_var(name, shape, dtype = tf.float32, init = None, wd = None, summaries = False, on_cpu = False):
    """ 
    Helper to create a Variable and summary if required
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        wd: weight decay (adds regularizer)
        summaries: attach summaries
    Returns:
        Variable Tensor
    """
    
    if on_cpu:
        with tf.device('/cpu:0'):        
            var = tf.get_variable(name, shape = shape, dtype = dtype, initializer = init)
        print("%s variable created on cpu"%(name))
    else:
        var = tf.get_variable(name, shape = shape, dtype = dtype, initializer = init)
        print("%s variable created on gpu"%(name))
        
    """ Regularization """
    if wd is not None:
        reg = tf.multiply(tf.nn.l2_loss(var), wd, name = "{}/wd".format(var.op.name))
        tf.add_to_collection('reg_losses', reg)
   
    """ Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if summaries:
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    return var


def _activation_summary(x):
    """ 
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    
class Graph:
    
    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        """ setup parameters W,B,Z and gamma """
        W_shape = [self.config.D, self.config.d]
        B_shape = [self.config.d, self.config.m]
        Z_shape = [self.config.m, self.config.L]
        
        on_cpu = True if (self.config.num_gpus > 1) else False
            
        
        self.gamma = tf.Variable(self.config.gamma, trainable = False)
        self.W = create_var("W", W_shape, init = self.config.initW, wd = self.config.wd, summaries = self.config.summaries, on_cpu = on_cpu)
        self.B = create_var("B", B_shape, init = self.config.initB, wd = self.config.wd, summaries = self.config.summaries, on_cpu = on_cpu)
        self.Z = create_var("Z", Z_shape, init = self.config.initZ, wd = self.config.wd, summaries = self.config.summaries, on_cpu = on_cpu)

        
    def forward(self, x):
        """ Builds forward pass of the graph """
        
        # Projection Layer: [N x D] --> [N x d]
        if self.config.is_sparse_input:
            Wx = tf.sparse_tensor_dense_matmul(x, self.W) 
        else:
            Wx = tf.matmul(x, self.W)
                              
        # RBF Layer: [N x d] --> [N x m]                                       
        Wx_sumsq = tf.reduce_sum(Wx**2, axis = -1, keepdims=True)
        B_sumsq = tf.reduce_sum(self.B**2, axis = 0, keepdims=True)
        v = Wx_sumsq + B_sumsq - 2*tf.matmul(Wx, self.B)
        D = tf.exp( -(self.gamma**2) * v) 
        if self.config.normalize_D:
            D = tf.nn.l2_normalize(D, axis = -1)
        
        # Z layer: [N x m] --> [m x L]
        y_pred = tf.matmul(D, self.Z) 
        if self.config.normalize_output:
            y_pred = tf.nn.l2_normalize(y_pred, axis = -1)
        
        return D, y_pred